# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Spark DSG receiver for ROS."""

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from hflex_eqa_python import EQAInput, EQAPlannerInput, HighLevelPlannerInput
from hydra_msgs.msg import DsgUpdate
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from spark_config import Config
from spark_dsg._dsg_bindings import DynamicSceneGraph
from std_srvs.srv import Trigger

from hflex_eqa_msgs.msg import ObjectsViews
from hflex_eqa_ros.ros_conversions import Conversions


@dataclass
class DsgImageRosReceiverConfig(Config):
    """Configuration for DSG ROS receiver."""

    dsg_topic: str = "dsg_update"
    objects_views_topic: str = "objects_views"
    img_topic: str = "image"
    force_full_load: bool = False
    clear_change: bool = True
    queue_size: int = 500
    slop_s: float = 0.1

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class DsgImageRosReceiver:
    """Receive a scene graph over ROS."""

    def __init__(self, node: Node, config: DsgImageRosReceiverConfig) -> None:
        """Initialize the ROS subscriber.
        :param node: The ROS node to use for subscribing.
        :param config: The configuration for the receiver.
        """
        self._node = node
        self._config = config
        self._has_change = False
        self._graph = None
        self._img = None
        self._objects_views = None
        self._last_time_ = None
        self._last_frame_id = None
        self._dsg_sub = Subscriber(node, DsgUpdate, self._config.dsg_topic)
        self._objects_views_sub = Subscriber(
            node, ObjectsViews, self._config.objects_views_topic
        )
        self._img_sub = Subscriber(node, Image, self._config.img_topic)
        self._sync = ApproximateTimeSynchronizer(
            [self._dsg_sub, self._objects_views_sub, self._img_sub],
            queue_size=self._config.queue_size,
            slop=self._config.slop_s,
        )

        self._sync.registerCallback(self._callback)

        self._mutex = threading.Lock()
        self._should_shutdown = threading.Event()

    def _callback(
        self, dsg: DsgUpdate, objects_views: ObjectsViews, img: Image
    ) -> None:
        """ROS callback for receiving a scene graph update.
        :param dsg: The received DSG update message.
        :param objects_views: The received objects views message.
        :param img: The received image message.
        """
        self._node.get_logger().info("Received new DSG update, processing...")
        with self._mutex:
            self._last_time_ = dsg.header.stamp
            self._last_frame_id = dsg.header.frame_id
            if self._last_frame_id == "":
                self._last_frame_id = None
                self._node.get_logger().error(
                    "Received DSG update with empty frame_id, ignoring"
                )
                return
            try:
                if self._graph is None or self._config.force_full_load:
                    self._graph = DynamicSceneGraph.from_binary(
                        bytes(dsg.layer_contents)
                    )
                else:
                    self._graph.update_from_binary(bytes(dsg.layer_contents))
                self._img = Conversions.to_image(img)
                if len(objects_views.object_ids) == 0:
                    self._objects_views = None
                else:
                    self._objects_views = {
                        obj_id: Conversions.to_image(image)
                        for obj_id, image in zip(
                            objects_views.object_ids, objects_views.images
                        )
                    }
                self._has_change = True
            except Exception as e:
                self._node.get_logger().error(f"Failed to parse DSG update: {e}")
                self._graph = None

    def has_change(self) -> bool:
        """Check if a new graph has been received
        since the last call to this function."""
        with self._mutex:
            return self._has_change

    def clear_change(self) -> None:
        """Clear the change flag."""
        with self._mutex:
            if self._config.clear_change:
                self._has_change = False

    def get(self) -> tuple[DynamicSceneGraph, np.ndarray, str, Time]:
        """Get the latest received graph, along with its frame_id and timestamp.
        :return: A tuple containing the latest graph, image,
                 its frame_id, and its timestamp.
        """
        with self._mutex:
            if self._graph is None:
                self._node.get_logger().error(
                    "No DSG graph received yet, cannot get graph"
                )
                return None, None, None, None
            if self._config.clear_change:
                self._has_change = False
            return (
                self._graph,
                self._img,
                self._objects_views,
                self._last_frame_id,
                self._last_time_,
            )


@dataclass
class DsgTriggeredRosReceiverWorkerConfig(Config):
    """Configuration for DSG triggered receiver worker."""

    dsg_config: DsgImageRosReceiverConfig = field(
        default_factory=DsgImageRosReceiverConfig
    )
    planner_srv_name: str = "trigger_planner"
    qa_srv_name: str = "trigger_qa"

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class DsgTriggeredRosReceiverWorker:
    """Worker to process DSG updates in a separate thread,
    triggered by new DSG updates."""

    def __init__(
        self,
        node: Node,
        config: DsgTriggeredRosReceiverWorkerConfig,
        planner_queue: queue.Queue | None,
        qa_queue: queue.Queue | None,
    ) -> None:
        """Initialize the worker.
        :param node: The ROS node to use for subscribing.
        :param config: The configuration for the worker.
        :param planner_queue: The queue to put planner inputs in
                              when a new DSG update is received.
        :param qa_queue: The queue to put QA inputs in when a
                         new DSG update is received.
        """
        self._node = node

        self._config = config
        self._planner_queue = planner_queue
        self._qa_queue = qa_queue

        self._receiver = DsgImageRosReceiver(self._node, config.dsg_config)

        # Create services for triggering planner and QA
        self._planner_srv = self._node.create_service(
            Trigger, config.planner_srv_name, self._handle_planner_trigger
        )
        self._qa_srv = self._node.create_service(
            Trigger, config.qa_srv_name, self._handle_qa_trigger
        )

    def _handle_planner_trigger(
        self, _: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle planner trigger service call.
        :param _: The service request (not used).
        :param response: The service response to fill.
        :return: The filled service response."""
        self._node.get_logger().info(
            "Received planner trigger service call, checking for new DSG update..."
        )
        if self._planner_queue is None:
            self._node.get_logger().error(
                "Planner queue is not set, cannot trigger planner"
            )
            response.success = False
            return response
        if self._receiver.has_change():
            graph, img, objects_views, frame_id, timestamp = self._receiver.get()
            ts_ns = timestamp.sec * 1_000_000_000 + timestamp.nanosec
            self._node.get_logger().info(
                f"Triggering planner with new DSG update at "
                f"time {ts_ns} and frame_id {frame_id}"
            )
            self._planner_queue.put(
                HighLevelPlannerInput(graph, img, objects_views, ts_ns, frame_id)
            )
            response.success = True
        else:
            self._node.get_logger().info(
                "No new DSG update available, not triggering planner."
            )
            response.success = False
        return response

    def _handle_qa_trigger(
        self, _: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle QA trigger service call.
        :param _: The service request (not used).
        :param response: The service response to fill.
        :return: The filled service response."""
        self._node.get_logger().info(
            "Received QA trigger service call, checking for new DSG update..."
        )
        if self._qa_queue is None:
            self._node.get_logger().error("QA queue is not set, cannot trigger QA")
            response.success = False
            return response
        if self._receiver.has_change():
            graph, img, objects_views, frame_id, timestamp = self._receiver.get()
            ts_ns = timestamp.sec * 1_000_000_000 + timestamp.nanosec
            self._node.get_logger().info(
                f"Triggering QA with new DSG update at "
                f"time {ts_ns} and frame_id {frame_id}"
            )
            self._qa_queue.put(
                EQAPlannerInput(graph, img, objects_views, ts_ns, frame_id)
            )
            response.success = True
        else:
            self._node.get_logger().info(
                "No new DSG update available, not triggering QA."
            )
            response.success = False
        return response


@dataclass
class DsgRosReceiverWorkerConfig(Config):
    """Configuration for DSG receiver worker."""

    dsg_config: DsgImageRosReceiverConfig = field(
        default_factory=DsgImageRosReceiverConfig
    )
    queue_size: int = 1
    min_separation_s: float = 0.0

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


class DsgRosReceiverWorker:
    """Worker to process DSG updates in a separate thread."""

    def __init__(
        self, node: Node, config: DsgRosReceiverWorkerConfig, callback: callable
    ) -> None:
        """Initialize the worker."""
        self._node = node
        self._node.context.on_shutdown(self.stop)

        self._config = config
        self._callback = callback

        self._started = False
        self._should_shutdown = False
        self._last_stamp = None

        self._queue = queue.Queue(maxsize=config.queue_size)
        self._receiver = DsgImageRosReceiver(self._node, config.dsg_config)
        self.start()

    def start(self):
        """Start worker processing queue."""
        if not self._started:
            self._started = True
            self._worker_thread = threading.Thread(target=self._spin)
            self._receiver_thread = threading.Thread(target=self._spin_receiver)
            self._worker_thread.start()
            self._receiver_thread.start()

    def stop(self):
        """Stop the worker."""
        if self._started:
            self._should_shutdown = True
            self._worker_thread.join()
            self._receiver_thread.join()

        self._started = False
        self._should_shutdown = False

    def _spin_receiver(self):
        """Spin the receiver thread."""
        while not self._should_shutdown:
            if not self._receiver.has_change():
                time.sleep(0.1)
                continue
            graph, img, objects_views, frame_id, timestamp = self._receiver.get()
            ts_ns = timestamp.sec * 1_000_000_000 + timestamp.nanosec
            if graph is None:
                continue
            if self._last_stamp is not None:
                ts_ns = timestamp.sec * 1_000_000_000 + timestamp.nanosec
                last_ns = (
                    self._last_stamp.sec * 1_000_000_000 + self._last_stamp.nanosec
                )
                diff_s = (ts_ns - last_ns) * 1.0e-9
                if diff_s < self._config.min_separation_s:
                    continue

            self._last_stamp = timestamp
            self._queue.put(
                (graph, img, objects_views, frame_id, ts_ns), block=False, timeout=False
            )

    def _spin(self):
        """Spin the worker thread."""
        while not self._should_shutdown:
            try:
                graph, img, objects_views, frame_id, timestamp = self._queue.get(
                    timeout=0.1
                )
                self._callback(graph, img, objects_views, frame_id, timestamp)
            except queue.Empty:
                continue
