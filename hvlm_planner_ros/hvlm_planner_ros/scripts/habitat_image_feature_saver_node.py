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
"""Save Habitat RGB/depth/detection images and dense semantic features."""

from __future__ import annotations

import pathlib
import re
import sys
from collections import defaultdict
from collections.abc import Callable

import cv2
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from semantic_inference_msgs.msg import FullFeatures


class HabitatImageFeatureSaverNode(Node):
    """ROS node that records image streams and pixelwise feature maps to disk."""

    _ENCODING_PATTERN = re.compile(
        r"^(?P<bits>8|16|32|64)(?P<kind>[USF])C(?P<channels>\d+)$"
    )
    _ENCODING_DTYPES = {
        ("8", "U"): np.uint8,
        ("8", "S"): np.int8,
        ("16", "U"): np.uint16,
        ("16", "S"): np.int16,
        ("32", "S"): np.int32,
        ("32", "F"): np.float32,
        ("64", "F"): np.float64,
    }

    def __init__(self) -> None:
        """Initialize subscribers and output directories."""
        super().__init__("habitat_image_feature_saver_node")
        self._bridge = cv_bridge.CvBridge()
        self._counts = defaultdict(int)
        self._saved_counts = defaultdict(int)

        output_dir = self.declare_parameter(
            "output_dir", "/developer/ros2_hydra_ws/images/hvlm_planner_ros_captures"
        ).value
        self._output_dir = pathlib.Path(output_dir).expanduser().absolute()
        self._stream_dirs = {
            "rgb": self._output_dir / "rgb",
            "depth": self._output_dir / "depth",
            "detections": self._output_dir / "detections",
            "pixelwise_feature_map": self._output_dir / "pixelwise_feature_map",
        }
        for stream_dir in self._stream_dirs.values():
            stream_dir.mkdir(parents=True, exist_ok=True)

        self._save_every_n = max(
            1, int(self.declare_parameter("save_every_n", 1).value)
        )
        self._max_samples_per_topic = int(
            self.declare_parameter("max_samples_per_topic", 0).value
        )
        queue_size = int(self.declare_parameter("queue_size", 10).value)

        rgb_topic = self.declare_parameter("rgb_topic", "/habitat/rgb/image_raw").value
        depth_topic = self.declare_parameter(
            "depth_topic", "/habitat/depth/image_raw"
        ).value
        detections_topic = self.declare_parameter(
            "detections_topic", "/semantic_inference/detections"
        ).value
        full_features_topic = self.declare_parameter(
            "full_features_topic", "/semantic_inference/semantic/full_features"
        ).value

        self.create_subscription(
            Image,
            rgb_topic,
            lambda msg: self._save_color_image(msg, "rgb"),
            queue_size,
        )
        self.create_subscription(Image, depth_topic, self._save_depth_image, queue_size)
        self.create_subscription(
            Image,
            detections_topic,
            lambda msg: self._save_color_image(msg, "detections"),
            queue_size,
        )
        self.create_subscription(
            FullFeatures,
            full_features_topic,
            self._save_pixelwise_feature_map,
            queue_size,
        )

        self.get_logger().info(f"Saving Habitat captures to '{self._output_dir}'")
        self.get_logger().info(
            "Subscribed to "
            f"'{rgb_topic}', '{depth_topic}', '{detections_topic}', "
            f"and '{full_features_topic}'"
        )

    def _should_save(self, stream: str) -> bool:
        self._counts[stream] += 1
        if self._counts[stream] % self._save_every_n != 0:
            return False
        return (
            self._max_samples_per_topic <= 0
            or self._saved_counts[stream] < self._max_samples_per_topic
        )

    def _next_path(
        self, msg: Image | FullFeatures, stream: str, suffix: str
    ) -> pathlib.Path:
        self._saved_counts[stream] += 1
        stamp = msg.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns == 0:
            now = self.get_clock().now().nanoseconds
            stamp_ns = now
        filename = f"{stamp_ns}_{self._saved_counts[stream]:06d}.{suffix}"
        return self._stream_dirs[stream] / filename

    def _save_with_errors(
        self, stream: str, callback: Callable[[], pathlib.Path | None]
    ) -> None:
        if not self._should_save(stream):
            return
        saved_count = self._saved_counts[stream]
        try:
            saved_path = callback()
        except Exception as err:  # noqa: BLE001
            self.get_logger().error(f"Failed to save {stream}: {err}")
            self._saved_counts[stream] = saved_count
            return
        if saved_path is not None:
            self.get_logger().debug(f"Saved {stream} to '{saved_path}'")

    def _save_color_image(self, msg: Image, stream: str) -> None:
        """Save an RGB-like image topic as a PNG."""

        def save() -> pathlib.Path:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            path = self._next_path(msg, stream, "png")
            if not cv2.imwrite(str(path), img):
                raise RuntimeError(f"cv2.imwrite returned false for '{path}'")
            return path

        self._save_with_errors(stream, save)

    def _save_depth_image(self, msg: Image) -> None:
        """Save a depth image as a PNG, using millimeters for float depths."""

        def save() -> pathlib.Path:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if msg.encoding == "32FC1" or depth.dtype.kind == "f":
                depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                depth_png = np.clip(
                    depth_mm * 1000.0, 0, np.iinfo(np.uint16).max
                ).astype(np.uint16)
            elif msg.encoding == "16UC1" or depth.dtype == np.uint16:
                depth_png = depth
            else:
                depth_png = np.asarray(depth)

            path = self._next_path(msg, "depth", "png")
            if not cv2.imwrite(str(path), depth_png):
                raise RuntimeError(f"cv2.imwrite returned false for '{path}'")
            return path

        self._save_with_errors("depth", save)

    def _save_pixelwise_feature_map(self, msg: FullFeatures) -> None:
        """Save the FullFeatures pixelwise_feature_map field as a NPY array."""

        def save() -> pathlib.Path | None:
            feature_msg = msg.pixelwise_feature_map
            if feature_msg.height == 0 or feature_msg.width == 0:
                self.get_logger().warn("Skipping empty pixelwise_feature_map")
                return None
            feature_map = self._image_msg_to_array(feature_msg)
            path = self._next_path(msg, "pixelwise_feature_map", "npy")
            np.save(path, feature_map)
            return path

        self._save_with_errors("pixelwise_feature_map", save)

    def _image_msg_to_array(self, msg: Image) -> np.ndarray:
        """Convert generic channel-packed sensor_msgs/Image data to a numpy array."""
        parsed_encoding = self._parse_image_encoding(msg.encoding)
        dtype = np.dtype(parsed_encoding[0])
        channels = parsed_encoding[1]
        if msg.is_bigendian:
            dtype = dtype.newbyteorder(">")
        else:
            dtype = dtype.newbyteorder("<")

        row_items = msg.step // dtype.itemsize
        expected_items = msg.height * row_items
        data = np.frombuffer(msg.data, dtype=dtype, count=expected_items)
        rows = data.reshape((msg.height, row_items))
        image = rows[:, : msg.width * channels].reshape(
            (msg.height, msg.width, channels)
        )
        return np.asarray(image, dtype=parsed_encoding[0])

    def _parse_image_encoding(self, encoding: str) -> tuple[np.dtype, int]:
        match = self._ENCODING_PATTERN.match(encoding)
        if match is None:
            raise ValueError(f"Unsupported image encoding '{encoding}'")

        bits = match.group("bits")
        kind = match.group("kind")
        dtype = self._ENCODING_DTYPES.get((bits, kind))
        if dtype is None:
            raise ValueError(f"Unsupported image encoding '{encoding}'")

        channels = int(match.group("channels"))
        if channels <= 0:
            raise ValueError(f"Invalid channel count in image encoding '{encoding}'")
        return np.dtype(dtype), channels


def main(args: list[str] | None = None) -> None:
    """Run the saver node."""
    rclpy.init(args=args)
    node = HabitatImageFeatureSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
