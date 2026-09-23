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
"""Node that manages the Habitat EQA simulation."""

import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
import spark_config as sc
import yaml
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from rclpy.node import Node
from std_srvs.srv import Trigger

from hvlm_planner_msgs.msg import EQAPlannerOutput, Monitor
from simulation_manager_ros import (
    DataConfig,
    State,
    load_eqa_data,
    state_from_string,
    state_to_string,
)


@dataclass
class HabitatSimulationManagerNodeConfig(sc.Config):
    max_episodes: int = 50
    data: DataConfig = field(default_factory=DataConfig)
    frequency: float = 1.0
    launch_pkg: str = ""
    launch_file: str = ""
    launch_wait_s: float = 5.0
    log_folder: str = "logs"
    skip_existing: bool = True
    trigger: bool = True
    use_floorplan_prior: bool = True
    use_choices: bool = True
    gt_semantics: bool = False


class HabitatSimulationManagerNode(Node):
    """Node that manages the Habitat EQA simulation."""

    def __init__(self) -> None:
        """Initialize the HabitatSimulationManagerNode."""
        super().__init__("habitat_eqa_manager_node")

        ros_config_params = (
            self.declare_parameter("config", "").get_parameter_value().string_value
        )
        config_path = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )

        config_path = pathlib.Path(config_path).expanduser().absolute()

        if config_path.exists():
            self.config = sc.Config.load(
                HabitatSimulationManagerNodeConfig, config_path
            )
        else:
            self.config = HabitatSimulationManagerNodeConfig()

        self.config.update(yaml.safe_load(ros_config_params) or {})

        self.get_logger().info(f"Initializing with {self.config.show()}")

        self._episodes = load_eqa_data(
            self.config.data, self.config.use_floorplan_prior
        )
        self.get_logger().info(
            f"Loaded {len(self._episodes)} episodes from {self.config.data.dataset}."
        )

        if self.config.data.load_test_split_only:
            self.get_logger().info(
                "Test split mode enabled with manifest "
                f"{self.config.data.test_split_manifest_path}"
            )

        self._idx = 0
        self._num_iters = 0
        self._path_length = 0
        self._path = np.empty((0, 3))
        self._launch_episode = True
        self._gt_answer = None
        self._log_folder = pathlib.Path(self.config.log_folder)
        self._log_folder.mkdir(parents=True, exist_ok=True)
        if not self.config.skip_existing:
            shutil.rmtree(self._log_folder)
            self._log_folder.mkdir(parents=True, exist_ok=True)
        else:
            self._get_next_idx()

        self._launch_process = None
        self._launch_timer = None
        self._pending_launch_args = None

        self.create_subscription(Monitor, "monitor", self._monitor_callback, 10)

        self.create_subscription(EQAPlannerOutput, "eqa_output", self._eqa_callback, 10)

        self.create_subscription(Path, "agent_path", self._path_callback, 10)

        self.create_subscription(
            PoseArray, "habitat_agent_path", self._habitat_path_callback, 10
        )

        # Create service client for triggering episodes
        self._trigger_client = self.create_client(Trigger, "trigger_episode")

        self.create_timer(
            1.0 / self.config.frequency,
            self._main_loop,
        )

        self.get_logger().info("Finished initializing!")

    def _start_launch(self, launch_args: dict) -> None:
        """Start the launch file with the given arguments.
        :param launch_args: A dictionary of arguments to pass to the launch file.
        """
        if self._launch_process is not None:
            self.get_logger().warn("Launch already running.")
            return

        if not self.config.launch_file:
            self.get_logger().error("No launch_file configured.")
            return

        ros_cmd = [
            "ros2",
            "launch",
            self.config.launch_pkg,
            self.config.launch_file,
        ]

        for key, value in launch_args.items():
            if key not in [
                "initial_T_HB",
                "use_floorplan_prior",
                "floorplan_nodes",
                "floorplan_edges",
                "choices",
                "use_choices",
                "gt_semantics",
            ]:
                ros_cmd.append(f"{key}:='{value}'")
            else:
                ros_cmd.append(f"{key}:={value}")

        env = os.environ.copy()

        # Remove only the Qt vars that break rviz2
        env.pop("QT_PLUGIN_PATH", None)
        env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

        # Usually helpful in containers
        env["QT_QPA_PLATFORM"] = "xcb"

        self.get_logger().info(f"Starting launch process:\n  {' '.join(ros_cmd)}")

        # Create new process group so we can kill entire tree
        self._launch_process = subprocess.Popen(
            ros_cmd,
            preexec_fn=os.setsid,
            env=env,
        )

    def _stop_launch(self) -> None:
        """Stop the currently running launch process."""
        if self._launch_process is None:
            return

        self.get_logger().info("Stopping launch process...")

        try:
            os.killpg(
                os.getpgid(self._launch_process.pid),
                signal.SIGINT,
            )
            self._launch_process.wait(timeout=10)
        except Exception as e:
            self.get_logger().warn(f"Graceful shutdown failed: {e}")
            os.killpg(
                os.getpgid(self._launch_process.pid),
                signal.SIGKILL,
            )

        self._launch_process = None
        self.get_logger().info("Launch process stopped.")

    def _get_next_idx(self) -> None:
        if not self.config.skip_existing:
            self._idx += 1
            return
        while True:
            if self._idx >= len(self._episodes):
                break
            episode = self._episodes[self._idx]
            question_id = episode["question_id"]
            if (self._log_folder / f"{question_id}" / "log.json").exists():
                self.get_logger().info(f"Skipping existing question {question_id}")
                self._idx += 1
            elif (self._log_folder / f"{question_id}").exists():
                # Remove incomplete log folder
                self.get_logger().info(
                    f"Removing incomplete log folder for question {question_id}"
                )
                shutil.rmtree(self._log_folder / f"{question_id}")
                break
            else:
                break

    def _log_episode_data(self, state: str, answer: str = "") -> None:
        """Log data from the episode to a file
        :param state: The state of the episode (e.g. "finished", "stuck", etc.)
        :param answer: The answer given by the agent (if applicable)
        """
        self.get_logger().info(
            f"Logging episode data for state: {state}, answer: {answer}"
        )
        episode = self._episodes[self._idx]
        log = {
            "question": episode["question"],
            "question_category": episode.get("category", "unknown"),
            "choices": episode.get("choices", []),
            "gt_answer": self._gt_answer,
            "answer": answer,
            "final_state": state,
            "num_iters": self._num_iters,
            "path_length": self._path_length,
            "path": self._path.tolist(),
        }
        self._path_length = 0
        self._path = np.empty((0, 3))

        question_id = episode["question_id"]
        log_folder = self._log_folder / f"{question_id}"
        log_folder.mkdir(parents=True, exist_ok=True)
        log_path = log_folder / "log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        self.get_logger().info(f"Logged episode data to {log_path}")

    def _monitor_callback(self, msg: Monitor) -> None:
        """Callback for monitor messages to track episode state and iterations.
        :param msg: The incoming monitor message containing state and iteration info.
        """
        if self._launch_episode:
            return
        self._num_iters = msg.iteration
        state = state_from_string(msg.state)

        if (
            state in [State.STUCK, State.HOMING]
            or self._num_iters >= self.config.max_episodes
        ):
            self.get_logger().info("Episode ended by monitor.")
            self._stop_launch()
            self.get_logger().info(
                f"Episode ended with state: {state}, iterations: {self._num_iters}"
            )
            self._log_episode_data(state_to_string(state))
            self._num_iters = 0
            self._get_next_idx()
            self.get_logger().info(f"Next question index: {self._idx}")
            self._launch_episode = True
        elif self.config.trigger and state == State.READY:
            self.get_logger().info("Episode ready, triggering next episode.")
            trigger = Trigger.Request()
            self._trigger_client.call_async(trigger)
            time.sleep(1)  # Give some time for the episode to start

    def _eqa_callback(self, msg: EQAPlannerOutput) -> None:
        """Callback for EQA output messages to check for answers.
        :param msg: The incoming EQA output message containing the
                    answer and whether it was correct.
        """
        if not msg.answered or self._launch_episode:
            return
        self.get_logger().info(f"Received answer: {msg.answer}, GT: {self._gt_answer}")

        self._stop_launch()
        self._log_episode_data("finished", msg.answer)
        self._num_iters = 0
        self._get_next_idx()
        self.get_logger().info(f"Next question index: {self._idx}")
        self._launch_episode = True

    def _path_callback(self, msg: Path) -> None:
        """Callback for agent path messages to track path length.
        :param msg: The incoming Path message containing the agent's path.
        """
        # Stack the path of points to the path array
        new_path = np.array(
            [
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                for pose in msg.poses
            ]
        )
        self._path = np.vstack((self._path, new_path))
        # Sum the distances between consecutive poses to get total path length
        self._path_length = np.sum(np.linalg.norm(np.diff(self._path, axis=0), axis=1))
        self.get_logger().info(f"Updated path length: {self._path_length}")

    def _habitat_path_callback(self, msg: PoseArray) -> None:
        """Callback for habitat agent path messages to track path length.
        :param msg: The incoming PoseArray message containing the agent's path.
        """
        new_path = np.array(
            [[pose.position.x, pose.position.y, pose.position.z] for pose in msg.poses]
        )
        self._path = np.vstack((self._path, new_path))
        self._path_length = np.sum(np.linalg.norm(np.diff(self._path, axis=0), axis=1))
        self.get_logger().info(
            f"Updated path length from habitat-sim: {self._path_length}"
        )

    def _main_loop(self) -> None:
        """Main loop to manage episode launches based
        on the current state and configuration."""
        if not self._launch_episode:
            return

        if self._idx >= len(self._episodes):
            self.get_logger().info("All episodes completed.")
            self.cleanup()
            return

        episode = self._episodes[self._idx]
        question_id = episode["question_id"]
        question = episode["question"]
        # Remove ' carachters from question and choices
        # to avoid issues with command line arguments
        question = question.replace("'", "")
        choices = [choice.replace("'", "") for choice in episode.get("choices", [])]

        floorplan_data = self._load_prior_floorplan(episode["floorplan_path"])
        self._gt_answer = episode.get("gt_answers", [])

        launch_args = {
            "scene_file": episode["scene_file"],
            "question": question,
            "choices": choices,
            "use_choices": self.config.use_choices,
            "gt_semantics": self.config.gt_semantics,
            "log_path": str(self._log_folder / f"{question_id}"),
            "initial_T_HB": episode["initial_pose"],
            "floorplan_nodes": floorplan_data.get("nodes", []),
            "floorplan_edges": floorplan_data.get("edges", []),
            "use_floorplan_prior": self.config.use_floorplan_prior,
        }

        self._launch_episode = False

        time.sleep(self.config.launch_wait_s)
        self._start_launch(launch_args)

    def _load_prior_floorplan(
        self, floorplan_path: str
    ) -> dict[str, list[str] | list[list[str]]]:
        """Load a prior floorplan from the given path and return
        its contents as a dictionary.
        :param floorplan_path: The file path to the floorplan to load.
        :return: The contents of the floorplan file as a dictionary.
        """
        if not os.path.exists(floorplan_path):
            self.get_logger().warn(f"Floorplan file {floorplan_path} does not exist.")
            return {}

        with open(floorplan_path) as f:
            floorplan_data = json.load(f)
        return floorplan_data

    def cleanup(self) -> None:
        """Clean up resources and stop any running processes."""
        self._stop_launch()
        rclpy.shutdown()


def main() -> None:
    """Start the HabitatSimulationManagerNode."""
    rclpy.init()

    node = HabitatSimulationManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()


if __name__ == "__main__":
    main()
