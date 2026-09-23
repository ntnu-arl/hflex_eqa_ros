#!/usr/bin/env python3
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
"""Node that wraps the high-level planner."""

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import rclpy
import spark_config as sc
import vlms_ros
import yaml
from hflex_eqa_python import (
    EQAPlanner,
    EQAPlannerConfig,
    EQAPlannerOutput,
)
from rclpy.node import Node
from std_msgs.msg import String
from vlm_msgs.msg import TaskParsingOutput

import hflex_eqa_ros
from hflex_eqa_msgs.msg import EQAPlannerOutput as EQAPlannerOutputMsg
from hflex_eqa_msgs.msg import QuestionRoomsEmbeddings
from hflex_eqa_ros import (
    Conversions,
    DsgTriggeredRosReceiverWorker,
    DsgTriggeredRosReceiverWorkerConfig,
)

import semantic_inference_python


@dataclass
class VLMsNodeConfig(sc.Config):
    """Configuration for VLMsNode."""

    worker: DsgTriggeredRosReceiverWorkerConfig = field(
        default_factory=DsgTriggeredRosReceiverWorkerConfig
    )
    eqa_planner: EQAPlannerConfig = field(default_factory=EQAPlannerConfig)
    embedding_model: Any = sc.config_field("clip", default="open_clip")
    question: str = ""


class VLMsNode(Node):
    """Node that wraps the high-level planner."""

    TRANSITION_PROMPT_TEMPLATES = (
        "doorway to a {room_label}",
        "entrance to a {room_label}",
        "hallway leading to a {room_label}",
        "opening into a {room_label}",
    )

    def __init__(self):
        """Initialize the node."""
        super().__init__("vlms_node")
        self.context.on_shutdown(self.stop)

        # Load configuration
        ros_config_params = (
            self.declare_parameter("config", "").get_parameter_value().string_value
        )
        config_path = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )
        ros_config_path = (
            self.declare_parameter("ros_config_path", "")
            .get_parameter_value()
            .string_value
        )
        question = (
            self.declare_parameter("question", "").get_parameter_value().string_value
        )
        config_path = pathlib.Path(config_path).expanduser().absolute()
        ros_config_path = pathlib.Path(ros_config_path).expanduser().absolute()
        if not config_path.exists() or not config_path.is_file():
            self.get_logger().warn(
                f"config path '{config_path}' does not exist or is not a file!"
            )
            self.config = VLMsNodeConfig()
        else:
            self.config = sc.Config.load(VLMsNodeConfig, config_path)
            self.config.update(yaml.safe_load(ros_config_params))

        if not ros_config_path.exists() or not ros_config_path.is_file():
            self.get_logger().warn(
                f"ROS config path '{ros_config_path}' does not exist or is not a file!"
            )
        else:
            with ros_config_path.open() as f:
                self.config.update(yaml.safe_load(f))

        if question != "":
            self.config.update({"question": question})

        try:
            self.config.eqa_planner.resolve_floorplan_graph()
        except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
            self.get_logger().error(f"Failed to load floorplan configuration: {exc}")
            raise

        # Initialize EQA planner module and add sink to publish its output
        self._eqa_planner = EQAPlanner(self.config.eqa_planner)
        self._eqa_planner.add_sink(self._publish_eqa_planner_output)
        if self.config.question != "":
            self._eqa_planner.question = self.config.question
            self._eqa_planner._image_sampler.set_question(self.config.question)

        self.get_logger().info(f"Initializing with {self.config.show()}")

        # Initialize ROS receiver worker
        self._dsg_receiver = DsgTriggeredRosReceiverWorker(
            self,
            self.config.worker,
            None,
            self._eqa_planner.input_queue,
        )
        # ROS subscribers
        self.create_subscription(String, "question", self._question_callback, 1)
        self.create_subscription(String, "choices", self._choices_callback, 1)
        self.create_subscription(
            TaskParsingOutput, "task_parsing_output", self._task_parsing_callback, 1
        )
        # Initialize ROS publishers
        self._eqa_planner_publisher = self.create_publisher(
            EQAPlannerOutputMsg, "eqa_planner_output", 1
        )
        self._embeddings_publisher = self.create_publisher(
            QuestionRoomsEmbeddings, "question_rooms_embeddings", 1
        )
        self._publish_room_question_embeddings()

        self.get_logger().info("Finished initializing!")

    def _publish_room_question_embeddings(self) -> None:
        """Publish room and question embeddings to ROS."""
        if self._eqa_planner._unique_room_names is None:
            self.get_logger().warn(
                "Unique room names are not available yet. "
                "Cannot publish room and question embeddings."
            )
            return

        room_embeddings = (
            self._eqa_planner._image_sampler._encoder.embed_text(
                self._eqa_planner._unique_room_names
            )
            .cpu()
            .numpy()
        )
        question_embedding = (
            self._eqa_planner._image_sampler._encoder.embed_text(
                self._eqa_planner.question
            )
            .cpu()
            .numpy()
            .squeeze()
        )

        msg = Conversions.question_rooms_embeddings_to_msg(
            room_embeddings=room_embeddings,
            room_labels=self._eqa_planner._unique_room_names,
            question_embedding=question_embedding,
            question=self._eqa_planner.question,
            header=Conversions.create_header(
                frame_id="", timestamp=self.get_clock().now()
            ),
        )
        self._embeddings_publisher.publish(msg)

    def _add_find_room_embeddings(self, output: EQAPlannerOutput) -> None:
        """Add CLIP embeddings needed by the find_room exploration strategy."""
        if not output.valid:
            return
        if output.mode.value != "find_room":
            return
        if output.target_room_label is None:
            return

        room_labels = output.floorplan_room_labels
        if room_labels is None:
            room_labels = self._eqa_planner._unique_room_names
        if not room_labels:
            return

        target_room_label = output.target_room_label
        output.transition_prompts = [
            template.format(room_label=target_room_label)
            for template in self.TRANSITION_PROMPT_TEMPLATES
        ]

        room_prompts = [
            [
                template.format(room_label=room_label)
                for template in self.TRANSITION_PROMPT_TEMPLATES
            ]
            for room_label in room_labels
        ]
        prompts = output.transition_prompts + [
            prompt for prompts_for_room in room_prompts for prompt in prompts_for_room
        ]
        embeddings = (
            self._eqa_planner._image_sampler._encoder.embed_text(prompts).cpu().numpy()
        )

        output.transition_prompt_embeddings = [
            embeddings[i].copy() for i in range(len(output.transition_prompts))
        ]

        room_embeddings = embeddings[len(output.transition_prompts) :]
        output.floorplan_room_embeddings = [
            room_embeddings[i : i + len(self.TRANSITION_PROMPT_TEMPLATES)].mean(axis=0)
            for i in range(
                0, len(room_embeddings), len(self.TRANSITION_PROMPT_TEMPLATES)
            )
        ]

    def start(self):
        """Start the node."""
        self.get_logger().info("Starting node...")
        self._eqa_planner.start()

    def _question_callback(self, msg: String) -> None:
        """
        Callback for incoming questions.
        :param msg: The incoming question message.
        """
        question = msg.data
        self.get_logger().info(f"Received question: '{question}'")
        self._eqa_planner.question = question
        self._eqa_planner._image_sampler.set_question(question)

    def _choices_callback(self, msg: String) -> None:
        """
        Callback for incoming answer choices.
        :param msg: JSON array string containing answer choices.
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Ignoring invalid choices JSON: {exc}")
            return

        if not isinstance(payload, list):
            self.get_logger().warn("Ignoring choices message that is not a JSON list")
            return

        choices = [str(choice).strip() for choice in payload if str(choice).strip()]
        self.config.eqa_planner.choices = choices
        self.config.eqa_planner.use_choices = bool(choices)
        self._eqa_planner._config.choices = choices
        self._eqa_planner._config.use_choices = bool(choices)
        self.get_logger().info(f"Received {len(choices)} answer choices")

    def _task_parsing_callback(self, msg: TaskParsingOutput) -> None:
        """
        Callback for incoming task parsing outputs.
        :param msg: The incoming task parsing output message.
        """
        features = Conversions.features_from_msg(msg.objects.features)
        if features is not None:
            self.get_logger().info(
                f"Received task parsing output with {len(features)} object features"
            )
            self._eqa_planner._image_sampler.set_objects(features)

    def _publish_eqa_planner_output(self, output: EQAPlannerOutput) -> None:
        """
        Publish EQA planner output to ROS.
        :param output: The output from the EQA planner to publish.
        """
        if not output.valid:
            self.get_logger().error("EQA planner output is invalid")

        self._add_find_room_embeddings(output)
        msg = Conversions.eqa_planner_output_to_msg(
            output,
            header=Conversions.create_header(output.frame, self.get_clock().now()),
            use_timestamp=False,
        )
        self._eqa_planner_publisher.publish(msg)

    def stop(self):
        """Stop the node."""
        self._eqa_planner.stop()


def main():
    """Start a node."""
    rclpy.init()

    node = None
    try:
        node = VLMsNode()
        node.start()
        hflex_eqa_ros.setup_ros_log_forwarding(node)
        vlms_ros.setup_ros_log_forwarding(node)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
        if node is not None:
            node.stop()


if __name__ == "__main__":
    main()
