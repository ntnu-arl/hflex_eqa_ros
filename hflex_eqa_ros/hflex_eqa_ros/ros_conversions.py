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
"""Module containing ROS message conversions."""

import struct

import cv2
import cv_bridge
import numpy as np
from hflex_eqa_python import EQAOutput, EQAPlannerOutput, HighLevelPlannerOutput
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header

from hflex_eqa_msgs.msg import EQAOutput as EQAOutputMsg
from hflex_eqa_msgs.msg import EQAPlannerOutput as EQAPlannerOutputMsg
from hflex_eqa_msgs.msg import FeatureVector as HflexEqaFeatureVector
from hflex_eqa_msgs.msg import HighLevelPlannerOutput as HighLevelPlannerOutputMsg
from hflex_eqa_msgs.msg import QuestionRoomsEmbeddings

from semantic_inference_msgs.msg import FeatureVector


class Conversions:
    """Conversion namespace."""

    bridge = cv_bridge.CvBridge()

    @staticmethod
    def to_hflex_eqa_feature(
        feature: np.ndarray | list[float],
    ) -> HflexEqaFeatureVector:
        """Create an HFLEX-EQA feature vector message."""
        return HflexEqaFeatureVector(feature=np.asarray(feature).flatten().tolist())

    @staticmethod
    def compressed_depth_to_cv2(msg: CompressedImage, depth_fmt: str):
        # remove header from raw data
        depth_header_size = 12
        raw_data = msg.data[depth_header_size:]

        depth_img = cv2.imdecode(
            np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
        )
        if depth_img is None:
            # probably wrong header size
            raise Exception(
                "Could not decode compressed depth image."
                "You may need to change 'depth_header_size'!"
            )

        if depth_fmt == "32FC1":
            raw_header = msg.data[:depth_header_size]
            # header: int, float, float
            [_, depthQuantA, depthQuantB] = struct.unpack("iff", raw_header)
            depth_img_scaled = depthQuantA / (
                depth_img.astype(np.float32) - depthQuantB
            )
            # filter max values
            depth_img_scaled[depth_img == 0] = 0

            # depth_img_scaled provides distance in meters as f32
            # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
            depth_img = (depth_img_scaled * 1000).astype(np.uint16)

        return (depth_img / 1000.0).astype(np.float32)

    @classmethod
    def to_image(cls, msg):
        """Convert sensor_msgs.Image to numpy array."""
        if isinstance(msg, Image):
            # Color images are expected to be in RGB format
            # Depth images are expected to be in 32FC1 format
            image = cls.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if msg.encoding == "16UC1":
                # Convert depth image from 16-bit unsigned int to float32
                image = image.astype(np.float32) / 1000.0
            elif msg.encoding == "32FC1":
                # Ensure depth image is in meters
                image = image.astype(np.float32)
            elif msg.encoding == "bgr8":
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        elif isinstance(msg, CompressedImage):
            format, _ = msg.format.split(";")
            format = format.strip()
            if format == "rgb8" or format == "bgr8":
                # Convert BGR to RGB
                image = cls.bridge.compressed_imgmsg_to_cv2(
                    msg, desired_encoding="passthrough"
                )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif format == "32FC1" or format == "16UC1":
                # Convert depth image to float32
                image = Conversions.compressed_depth_to_cv2(msg, format)
            return image
        else:
            raise ValueError(f"Message type '{type(msg)}' not supported!")

    @classmethod
    def to_image_msg(cls, header, img, encoding="passthrough"):
        msg = cls.bridge.cv2_to_imgmsg(img, encoding=encoding)
        msg.header = header
        return msg

    @staticmethod
    def hlp_output_to_msg(
        output: HighLevelPlannerOutput, header: Header, use_timestamp: bool = False
    ) -> HighLevelPlannerOutputMsg:
        """Convert HighLevelPlannerOutput to HighLevelPlannerOutputMsg.
        :param output: The HighLevelPlannerOutput to convert.
        :param header: The header to use for the message.
        :param use_timestamp: Whether to include the timestamp in the message.
        :return: The converted HighLevelPlannerOutputMsg.
        """
        msg = HighLevelPlannerOutputMsg()
        msg.header = header
        msg.mode = output.mode.value
        msg.target_room_id = (
            output.target_room_id if output.target_room_id is not None else 0
        )
        msg.target_object_ids = (
            output.target_object_ids if output.target_object_ids is not None else []
        )
        msg.sg_description = output.sg_description
        msg.images_description = (
            output.images_description if output.images_description is not None else []
        )
        msg.reasoning = output.reasoning
        msg.confidence = output.confidence
        msg.valid = output.valid
        if output.timestamp is not None and use_timestamp:
            # Timestamp is in nanoseconds, convert to ROS time
            msg.header.stamp = Time(nanoseconds=output.timestamp).to_msg()

        return msg

    @staticmethod
    def eqa_output_to_msg(
        output: EQAOutput, header: Header, use_timestamp: bool = False
    ) -> EQAOutputMsg:
        """Convert EQAOutput to EQAOutputMsg.
        :param output: The EQAOutput to convert.
        :param header: The header to use for the message.
        :param use_timestamp: Whether to include the timestamp in the message.
        :return: The converted EQAOutputMsg.
        """
        msg = EQAOutputMsg()
        msg.header = header
        msg.answered = output.answered
        msg.answer = output.answer
        msg.confidence = output.confidence
        msg.image_descriptions = (
            [
                image_description
                for image_description in output.image_descriptions
                if image_description is not None
            ]
            if output.image_descriptions is not None
            else []
        )
        msg.sg_description = output.sg_description
        msg.reasoning = output.reasoning
        msg.valid = output.valid
        if output.timestamp is not None and use_timestamp:
            # Timestamp is in nanoseconds, convert to ROS time
            msg.header.stamp = Time(nanoseconds=output.timestamp).to_msg()

        return msg

    @staticmethod
    def eqa_planner_output_to_msg(
        output: EQAPlannerOutput, header: Header, use_timestamp: bool = False
    ) -> EQAPlannerOutputMsg:
        """Convert EQAPlannerOutput to EQAPlannerOutputMsg.
        :param output: The EQAPlannerOutput to convert.
        :param header: The header to use for the message.
        :param use_timestamp: Whether to include the timestamp in the message.
        :return: The converted EQAPlannerOutputMsg.
        """
        msg = EQAPlannerOutputMsg()
        msg.header = header
        msg.answered = output.answered
        msg.answer = output.answer if output.answer is not None else ""
        msg.confidence = output.confidence
        msg.image_descriptions = (
            [
                image_description
                for image_description in output.image_descriptions
                if image_description is not None
            ]
            if output.image_descriptions is not None
            else []
        )
        msg.sg_description = (
            output.sg_description if output.sg_description is not None else ""
        )
        msg.floorplan_description = (
            output.floorplan_description
            if output.floorplan_description is not None
            else ""
        )
        msg.reasoning = output.reasoning if output.reasoning is not None else ""
        msg.mode = output.mode.value

        if output.target_room_id is not None and output.target_room_id > 0:
            msg.target_room_id = output.target_room_id
        msg.target_room_label = (
            output.target_room_label if output.target_room_label is not None else ""
        )
        msg.target_object_ids = (
            output.target_object_ids if output.target_object_ids is not None else []
        )
        msg.transition_prompts = (
            output.transition_prompts if output.transition_prompts is not None else []
        )
        msg.transition_prompt_embeddings = [
            Conversions.to_hflex_eqa_feature(embedding)
            for embedding in (
                output.transition_prompt_embeddings
                if output.transition_prompt_embeddings is not None
                else []
            )
        ]
        msg.floorplan_room_labels = (
            output.floorplan_room_labels
            if output.floorplan_room_labels is not None
            else []
        )
        msg.floorplan_progress_scores = (
            output.floorplan_progress_scores
            if output.floorplan_progress_scores is not None
            else []
        )
        msg.floorplan_room_embeddings = [
            Conversions.to_hflex_eqa_feature(embedding)
            for embedding in (
                output.floorplan_room_embeddings
                if output.floorplan_room_embeddings is not None
                else []
            )
        ]
        msg.new_labels = output.new_labels if output.new_labels is not None else []
        msg.valid = output.valid
        if output.timestamp is not None and use_timestamp:
            # Timestamp is in nanoseconds, convert to ROS time
            msg.header.stamp = Time(nanoseconds=output.timestamp).to_msg()

        return msg

    @staticmethod
    def create_header(frame_id: str, timestamp: Time) -> Header:
        """Create a ROS header with the given frame ID and timestamp.
        :param frame_id: The frame ID to use for the header.
        :param timestamp: The timestamp to use for the header.
        :return: The created Header.
        """
        header = Header()
        header.frame_id = frame_id
        header.stamp = timestamp.to_msg()
        return header

    @staticmethod
    def features_from_msg(features_msg: list[FeatureVector]) -> np.ndarray | None:
        """Convert a list of FeatureVector messages to a numpy array.
        :param features_msg: The list of FeatureVector messages to convert.
        :return: A numpy array containing the features.
        """
        if not features_msg:
            return None

        feature_length = len(features_msg[0].data)
        features_array = np.zeros((len(features_msg), feature_length), dtype=np.float32)

        for i, feature in enumerate(features_msg):
            features_array[i] = np.array(feature.data, dtype=np.float32)

        return features_array

    @staticmethod
    def question_rooms_embeddings_to_msg(
        room_embeddings: np.ndarray,
        room_labels: list[str],
        question_embedding: np.ndarray,
        question: str,
        header: Header,
    ) -> QuestionRoomsEmbeddings:
        """Convert room and question embeddings to a QuestionRoomsEmbeddings message.
        :param room_embeddings: A numpy array containing the room embeddings.
        :param room_labels: A list of room labels corresponding to the embeddings.
        :param question_embedding: A numpy array containing the question embedding.
        :param question: The original question string.
        :param header: The header to use for the message.
        :return: The created QuestionRoomsEmbeddings message.
        """
        msg = QuestionRoomsEmbeddings()
        msg.header = header
        msg.question = question
        msg.question_embedding.feature = question_embedding.flatten().tolist()
        msg.room_labels = room_labels
        msg.room_embeddings = [
            Conversions.to_hflex_eqa_feature(room_embedding)
            for room_embedding in room_embeddings
        ]
        return msg
