/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2026, NTNU Autonomous Robots Lab
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */
#pragma once
#include <hflex_eqa/common/input.h>
#include <ianvs/node_handle.h>

#include <string>
#include <unordered_map>

#include <hflex_eqa_msgs/msg/eqa_output.hpp>
#include <hflex_eqa_msgs/msg/eqa_planner_output.hpp>
#include <hflex_eqa_msgs/msg/high_level_planner_output.hpp>
#include <hflex_eqa_msgs/msg/objects_views.hpp>
#include <hydra_msgs/msg/dsg_update.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/client.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/service.hpp>
#include <rclcpp/subscription.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace hflex_eqa {

class ManagerRosInterface {
 public:
  struct Config {
    std::string ns = "~/search_manager";
  } const config;

  explicit ManagerRosInterface(const Config& config);

  virtual ~ManagerRosInterface() = default;

  void publishEQAInfo(const input::EQAInput::Ptr& eqa_input,
                      bool publish_graph = true) const;

  void publishHLPInfo(const input::HLPInput::Ptr& hlp_input,
                      bool publish_graph = false) const;

  void publishMissionComplete(const bool complete) const;

 protected:
  void publishGraph(const spark_dsg::DynamicSceneGraph& dsg,
                    const uint64_t timestamp_ns) const;

  void publishObjectsViews(
      const std::unordered_map<spark_dsg::NodeId, std::optional<cv::Mat>>& object_views,
      const uint64_t timestamp_ns) const;

  void callbackEQAOutput(const hflex_eqa_msgs::msg::EQAOutput::SharedPtr msg) const;

  void callbackHLPOutput(
      const hflex_eqa_msgs::msg::HighLevelPlannerOutput::SharedPtr msg) const;

  void callbackEQAPlannerOutput(
      const hflex_eqa_msgs::msg::EQAPlannerOutput::SharedPtr msg) const;

  void callbackFailedSimGoal(const std_msgs::msg::Bool::SharedPtr msg) const;

  void handleStepTrigger(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) const;

  void handleStopTrigger(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) const;

  void handleNextObjectTrigger(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) const;

  const std::string printInfo() const;

  mutable ianvs::NodeHandle nh_;
  rclcpp::Publisher<hydra_msgs::msg::DsgUpdate>::SharedPtr dsg_publisher_;
  rclcpp::Publisher<hflex_eqa_msgs::msg::ObjectsViews>::SharedPtr
      objects_views_publisher_;
  rclcpp::Subscription<hflex_eqa_msgs::msg::EQAOutput>::SharedPtr
      eqa_output_subscriber_;
  rclcpp::Subscription<hflex_eqa_msgs::msg::HighLevelPlannerOutput>::SharedPtr
      hlp_output_subscriber_;
  rclcpp::Subscription<hflex_eqa_msgs::msg::EQAPlannerOutput>::SharedPtr
      eqa_planner_output_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr failed_sim_goal_subscriber_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_eqa_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_hlp_client_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr mission_complete_publisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_step_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_stop_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_next_object_service_;
};

void declare_config(ManagerRosInterface::Config& config);

}  // namespace hflex_eqa
