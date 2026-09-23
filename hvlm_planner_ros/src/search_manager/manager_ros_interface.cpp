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
#include "hvlm_planner_ros/search_manager/manager_ros_interface.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <glog/logging.h>
#include <hvlm_planner/common/agent_state.h>
#include <hvlm_planner/common/global_info.h>
#include <hvlm_planner/common/pipeline_queues.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>

#include <cv_bridge/cv_bridge.hpp>

#include "hvlm_planner_ros/utils/conversions.h"

namespace hvlm_planner {

void declare_config(ManagerRosInterface::Config& config) {
  using namespace config;
  name("ManagerRosInterfaceConfig");
  field(config.ns, "ns");
}

ManagerRosInterface::ManagerRosInterface(const Config& config)
    : config(config),
      nh_(ianvs::NodeHandle::this_node(config.ns)),
      dsg_publisher_(nh_.create_publisher<hydra_msgs::msg::DsgUpdate>("dsg", 1)),
      objects_views_publisher_(
          nh_.create_publisher<hvlm_planner_msgs::msg::ObjectsViews>("objects_views",
                                                                     1)),
      eqa_output_subscriber_(nh_.create_subscription<hvlm_planner_msgs::msg::EQAOutput>(
          "eqa_output", 1, &ManagerRosInterface::callbackEQAOutput, this)),
      hlp_output_subscriber_(
          nh_.create_subscription<hvlm_planner_msgs::msg::HighLevelPlannerOutput>(
              "hlp_output", 1, &ManagerRosInterface::callbackHLPOutput, this)),
      eqa_planner_output_sub_(
          nh_.create_subscription<hvlm_planner_msgs::msg::EQAPlannerOutput>(
              "eqa_planner_output",
              1,
              &ManagerRosInterface::callbackEQAPlannerOutput,
              this)),
      failed_sim_goal_subscriber_(nh_.create_subscription<std_msgs::msg::Bool>(
          "failed_sim_goal",
          rclcpp::QoS(1).transient_local(),
          &ManagerRosInterface::callbackFailedSimGoal,
          this)),
      trigger_eqa_client_(nh_.create_client<std_srvs::srv::Trigger>("trigger_eqa")),
      trigger_hlp_client_(nh_.create_client<std_srvs::srv::Trigger>("trigger_hlp")),
      mission_complete_publisher_(
          nh_.create_publisher<std_msgs::msg::Bool>("mission_complete", 1)),
      trigger_step_service_(nh_.create_service<std_srvs::srv::Trigger>(
          "trigger_step", &ManagerRosInterface::handleStepTrigger, this)),
      trigger_stop_service_(nh_.create_service<std_srvs::srv::Trigger>(
          "trigger_stop", &ManagerRosInterface::handleStopTrigger, this)),
      trigger_next_object_service_(nh_.create_service<std_srvs::srv::Trigger>(
          "trigger_next_object", &ManagerRosInterface::handleNextObjectTrigger, this)) {
  RCLCPP_INFO(nh_.logger(),
              "ManagerRosInterface initialized with config:\n%s",
              printInfo().c_str());
  // Wait for service clients to be available
  while (!trigger_eqa_client_->wait_for_service(std::chrono::seconds(3))) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(nh_.logger(),
                   "Interrupted while waiting for trigger_eqa service. Exiting.");
      throw std::runtime_error("ROS shutdown while waiting for trigger_eqa service");
    }
    RCLCPP_INFO(nh_.logger(), "trigger_eqa service not available, retrying...");
  }
  RCLCPP_INFO(nh_.logger(), "trigger_eqa service is available.");
  while (!trigger_hlp_client_->wait_for_service(std::chrono::seconds(3))) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(nh_.logger(),
                   "Interrupted while waiting for trigger_hlp service. Exiting.");
      throw std::runtime_error("ROS shutdown while waiting for trigger_hlp service");
    }
    RCLCPP_INFO(nh_.logger(), "trigger_hlp service not available, retrying...");
  }
  RCLCPP_INFO(nh_.logger(), "trigger_hlp service is available.");
}

void ManagerRosInterface::publishEQAInfo(const input::EQAInput::Ptr& eqa_input,
                                         bool publish_graph) const {
  publishObjectsViews(eqa_input->object_views, eqa_input->base_input->getTimestamp());
  if (publish_graph) {
    publishGraph(*(eqa_input->base_input->getDSG()),
                 eqa_input->base_input->getTimestamp());
  }
  LOG(INFO) << "Publishing and triggering EQA info with question: "
            << GlobalInfo::instance().getQuestion();
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  trigger_eqa_client_->async_send_request(request);
}

void ManagerRosInterface::publishHLPInfo(const input::HLPInput::Ptr& hlp_input,
                                         bool publish_graph) const {
  if (publish_graph) {
    publishGraph(*(hlp_input->base_input->getDSG()),
                 hlp_input->base_input->getTimestamp());
  }

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  trigger_hlp_client_->async_send_request(request);
}

void ManagerRosInterface::publishMissionComplete(const bool complete) const {
  std_msgs::msg::Bool msg;
  msg.data = complete;
  mission_complete_publisher_->publish(msg);
}

void ManagerRosInterface::publishGraph(const spark_dsg::DynamicSceneGraph& dsg,
                                       const uint64_t timestamp_ns) const {
  auto msg = std::make_unique<hydra_msgs::msg::DsgUpdate>();
  msg->header.stamp = rclcpp::Time(timestamp_ns);
  msg->header.frame_id = GlobalInfo::instance().getFrames().odom;
  spark_dsg::io::binary::writeGraph(dsg, msg->layer_contents, false);
  msg->full_update = true;
  dsg_publisher_->publish(std::move(msg));
  // Wait for the message to be published before returning to ensure that the latest
  // graph is sent before any triggers are sent that rely on it.
  rclcpp::sleep_for(std::chrono::milliseconds(500));
}

void ManagerRosInterface::publishObjectsViews(
    const std::unordered_map<spark_dsg::NodeId, std::optional<cv::Mat>>& object_views,
    const uint64_t timestamp_ns) const {
  auto msg = std::make_unique<hvlm_planner_msgs::msg::ObjectsViews>();
  msg->header.stamp = rclcpp::Time(timestamp_ns);
  msg->header.frame_id = GlobalInfo::instance().getFrames().odom;
  for (const auto& [node_id, view_opt] : object_views) {
    if (!view_opt) {
      continue;  // Skip if there is no view for this object
    }
    msg->object_ids.push_back(node_id);
    sensor_msgs::msg::Image ros_view;
    const auto img_msg =
        cv_bridge::CvImage(
            msg->header, sensor_msgs::image_encodings::RGB8, view_opt.value())
            .toImageMsg();
    msg->images.push_back(*img_msg);
  }
  objects_views_publisher_->publish(std::move(msg));
}

void ManagerRosInterface::callbackEQAOutput(
    const hvlm_planner_msgs::msg::EQAOutput::SharedPtr msg) const {
  input::TriggerHLP::Ptr trigger = std::make_shared<input::TriggerHLP>();
  trigger->timestamp_ns = msg->header.stamp.nanosec;
  trigger->answered = msg->answered;
  trigger->answer = msg->answer;
  trigger->reasoning = msg->reasoning;
  trigger->confidence = msg->confidence;
  trigger->valid = msg->valid;
  auto& queue = PipelineQueues::instance().hlp_input_queue;
  if (!queue.push(trigger)) {
    LOG(WARNING) << "Failed to push HLP trigger to queue.";
  }
  VLOG(2) << "Pushed HLP trigger to queue with answer: " << trigger->answer;
}

void ManagerRosInterface::callbackHLPOutput(
    const hvlm_planner_msgs::msg::HighLevelPlannerOutput::SharedPtr msg) const {
  input::TriggerLLP::Ptr trigger = std::make_shared<input::TriggerLLP>();
  trigger->timestamp_ns = msg->header.stamp.nanosec;
  if (input::string_to_llp_modes.count(msg->mode) == 0) {
    LOG(WARNING) << "Received unknown LLP mode: " << msg->mode
                 << ", defaulting to EXPLORE.";
    trigger->mode = input::LLPMode::EXPLORE;
  } else {
    trigger->mode = input::string_to_llp_modes.at(msg->mode);
  }
  trigger->target_room_id = msg->target_room_id;
  trigger->target_object_ids = msg->target_object_ids;
  trigger->reasoning = msg->reasoning;
  trigger->confidence = msg->confidence;
  trigger->valid = msg->valid;
  auto& queue = PipelineQueues::instance().trigger_llp_queue;
  if (!queue.push(trigger)) {
    LOG(WARNING) << "Failed to push LLP trigger to queue.";
  }
  VLOG(2) << "Pushed LLP trigger to queue with mode: " << msg->mode;
}

void ManagerRosInterface::callbackEQAPlannerOutput(
    const hvlm_planner_msgs::msg::EQAPlannerOutput::SharedPtr msg) const {
  // TODO: Implement message handling

  if (!GlobalInfo::initialized()) {
    LOG(WARNING) << "GlobalInfo not initialized. Cannot add new labels.";
    return;
  }
  if (!GlobalInfo::instance().addNewLabels(msg->new_labels)) {
    LOG(WARNING) << "Could not add new labels to GlobalInfo.";
  }
  LOG(INFO) << "Added " << msg->new_labels.size() << " new labels. Total labels: "
            << GlobalInfo::instance().getNumActiveLabels();

  input::TriggerLLP::Ptr trigger = std::make_shared<input::TriggerLLP>();
  trigger->timestamp_ns = msg->header.stamp.nanosec;
  if (input::string_to_llp_modes.count(msg->mode) == 0) {
    LOG(WARNING) << "Received unknown LLP mode: " << msg->mode
                 << ", defaulting to EXPLORE.";
    trigger->mode = input::LLPMode::EXPLORE;
  } else {
    trigger->mode = input::string_to_llp_modes.at(msg->mode);
  }
  trigger->answered = msg->answered;
  trigger->answer = msg->answer;
  trigger->target_room_id = msg->target_room_id;
  trigger->target_room_label = msg->target_room_label;
  trigger->target_object_ids = msg->target_object_ids;
  conversions::convertFromRosFeatureVectors(msg->transition_prompt_embeddings,
                                            trigger->transition_prompt_embeddings);
  trigger->transition_prompts = msg->transition_prompts;
  trigger->floorplan_room_labels = msg->floorplan_room_labels;
  trigger->floorplan_progress_scores = msg->floorplan_progress_scores;
  conversions::convertFromRosFeatureVectors(msg->floorplan_room_embeddings,
                                            trigger->floorplan_room_embeddings);
  trigger->reasoning = msg->reasoning;
  trigger->confidence = msg->confidence;
  trigger->valid = msg->valid;
  auto& queue = PipelineQueues::instance().trigger_llp_queue;
  if (!queue.push(trigger)) {
    LOG(WARNING) << "Failed to push LLP trigger to queue.";
  }
  VLOG(2) << "Pushed LLP trigger to queue with mode: " << msg->mode;
}

void ManagerRosInterface::callbackFailedSimGoal(
    const std_msgs::msg::Bool::SharedPtr msg) const {
  auto& agent_state = AgentState::instance();
  agent_state.setState(State::READY);
  LOG(WARNING) << "Received failed_sim_goal message, setting agent state to READY.";
}

void ManagerRosInterface::handleStepTrigger(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) const {
  auto& trigger_queue = PipelineQueues::instance().trigger_step_queue;
  if (!trigger_queue.push(std::monostate{})) {
    LOG(WARNING) << "Failed to push step trigger to queue.";
    response->success = false;
    response->message = "Failed to push step trigger to queue.";
  } else {
    response->success = true;
    response->message = "Step trigger pushed to queue.";
  }
}

void ManagerRosInterface::handleStopTrigger(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) const {
  LOG(INFO) << "Received stop trigger service call.";
  auto trigger = std::make_shared<input::TriggerLLP>();
  trigger->mode = input::LLPMode::NONE;
  trigger->valid = true;

  auto& queues = PipelineQueues::instance();
  queues.trigger_step_queue.clear();
  AgentState::instance().setState(State::READY);
  queues.trigger_llp_queue.clear();
  if (!queues.trigger_llp_queue.push(trigger)) {
    LOG(WARNING) << "Failed to push stop trigger to queue.";
    response->success = false;
    response->message = "Failed to push stop trigger to queue.";
  } else {
    response->success = true;
    response->message = "Stop trigger pushed to queue.";
  }
}

void ManagerRosInterface::handleNextObjectTrigger(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) const {
  LOG(INFO) << "Received go to next object trigger service call.";
  auto& queue = PipelineQueues::instance().trigger_next_object_queue;
  if (!queue.push(std::monostate{})) {
    LOG(WARNING) << "Failed to push go to next object trigger to queue.";
    response->success = false;
    response->message = "Failed to push go to next object trigger to queue.";
  } else {
    response->success = true;
    response->message = "Go to next object trigger pushed to queue.";
  }
}

const std::string ManagerRosInterface::printInfo() const {
  return config::toString(config);
}

}  // namespace hvlm_planner
