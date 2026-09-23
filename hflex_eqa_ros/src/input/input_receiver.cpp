/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hflex_eqa_ros/input/input_receiver.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hflex_eqa/common/agent_state.h>
#include <hflex_eqa/common/global_info.h>
#include <hflex_eqa/common/input.h>
#include <hflex_eqa/common/pipeline_queues.h>
#include <hflex_eqa/common/types.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include "hflex_eqa_ros/utils/conversions.h"

namespace hflex_eqa {
namespace input {

void declare_config(InputReceiver::Config& config) {
  using namespace config;
  name("InputReceiverConfig");
  field(config.graph, "graph");
}

InputReceiver::InputReceiver(const Config& config, ianvs::NodeHandle nh)
    : config(config::checkValid(config)),
      nh_(nh),
      graph_wrapper_(config.graph.create(nh_)),
      occupancy_grid_sub_(nh_.create_subscription<nav_msgs::msg::OccupancyGrid>(
          "occupancy_grid",
          rclcpp::QoS(1).transient_local(),
          &InputReceiver::callbackOccupancyGrid,
          this)),
      image_sub_(nh_.create_subscription<sensor_msgs::msg::Image>(
          "image", 1, &InputReceiver::callbackImage, this)),
      ready_sub_(
          nh_.create_subscription<std_msgs::msg::Bool>("ready",
                                                       rclcpp::QoS(1).transient_local(),
                                                       &InputReceiver::callbackReady,
                                                       this)),
      question_sub_(nh_.create_subscription<std_msgs::msg::String>(
          "question", 1, &InputReceiver::callbackQuestion, this)),
      task_parsing_client_(
          nh_.create_client<vlm_msgs::srv::TaskParsing>("task_parsing")),
      task_parsing_output_sub_(
          nh_.create_subscription<vlm_msgs::msg::TaskParsingOutput>(
              "task_parsing_output",
              1,
              &InputReceiver::callbackTaskParsingOutput,
              this)),
      new_labels_sub_(nh_.create_subscription<hydra_msgs::msg::NewLabels>(
          "new_labels", 1, &InputReceiver::callbackNewLabels, this)),
      question_rooms_embeddings_sub_(
          nh_.create_subscription<hflex_eqa_msgs::msg::QuestionRoomsEmbeddings>(
              "question_rooms_embeddings",
              1,
              &InputReceiver::callbackQuestionRoomsEmbeddings,
              this)) {
  LOG(INFO) << "Initialized InputReceiver with configuration\n"
            << config::toString(config);
  // Wait for task parsing service to be available
  while (!task_parsing_client_->wait_for_service(std::chrono::seconds(3))) {
    if (!rclcpp::ok()) {
      LOG(ERROR) << "Interrupted while waiting for task_parsing service. Exiting.";
      throw std::runtime_error("ROS shutdown while waiting for task_parsing service");
    }
    LOG(INFO) << "task_parsing service not available, retrying...";
  }
  LOG(INFO) << "task_parsing service is available.";
  // Call service with question from GlobalInfo
  const auto& question = GlobalInfo::instance().getQuestion();
  auto request = std::make_shared<vlm_msgs::srv::TaskParsing::Request>();
  request->task = question;
  task_parsing_client_->async_send_request(request);
}

void InputReceiver::callbackOccupancyGrid(
    const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  OccupancyGrid::Ptr occupancy_grid;
  conversions::convertFromRosOccupancyGrid(*msg, occupancy_grid);
  if (!graph_wrapper_->hasChange()) {
    VLOG(4) << "No changes in the graph to plan on.";
    return;
  }
  if (occupancy_grid->empty()) {
    VLOG(4) << "Occupancy grid is empty. Cannot plan.";
    return;
  }
  const auto current_graph = graph_wrapper_->get();
  graph_wrapper_->clearChangeFlag();

  VLOG(5) << "Received graph of " << current_graph.graph->numNodes()
          << " nodes and occupancy grid of " << occupancy_grid->grid.size()
          << " cells.";

  auto& agent_state_instance = AgentState::instance();
  const auto& dsg = current_graph.graph;
  if (!agent_state_instance.initializedHome()) {
    // Get pose of agent's home node and set it in AgentState
    const auto agent_layer_id = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
    if (agent_layer_id) {
      const auto& prefix = GlobalInfo::instance().getRobotPrefix();
      const auto agent_layer = dsg->findLayer(agent_layer_id->layer, prefix.key);
      if (agent_layer) {
        spark_dsg::NodeSymbol agent_key(prefix.key, agent_layer->numNodes() - 1);
        const auto& agent_node = agent_layer->getNode(agent_key);
        Pose agent_pose = Pose::Identity();
        agent_pose.linear() = agent_node.attributes<spark_dsg::AgentNodeAttributes>()
                                  .world_R_body.toRotationMatrix();
        agent_pose.translation() =
            agent_node.attributes<spark_dsg::NodeAttributes>().position;
        agent_state_instance.trySetHomePosition(agent_pose, agent_key);
      }
    }
  }

  input::Input::Ptr input_data = std::make_shared<input::Input>(
      current_graph.timestamp.value_or(msg->header.stamp).nanoseconds(),
      dsg,
      occupancy_grid,
      latest_image_);
  auto& queue = PipelineQueues::instance().main_input_queue;
  if (!queue.push(input_data)) {
    LOG(ERROR) << "Failed to push input data to the main input queue.";
  } else {
    VLOG(4) << "Pushed input data to the main input queue.";
  }
}

void InputReceiver::callbackImage(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  // For now, just log that we received an image. We can add processing later if needed.
  VLOG(4) << "Received image with resolution " << msg->width << "x" << msg->height;
  conversions::convertFromRosImage(msg, latest_image_);
}

void InputReceiver::callbackQuestion(const std_msgs::msg::String::SharedPtr msg) {
  GlobalInfo::instance().setQuestion(msg->data);
  LOG(INFO) << "Set question from topic: " << msg->data;
  auto request = std::make_shared<vlm_msgs::srv::TaskParsing::Request>();
  request->task = msg->data;
  task_parsing_client_->async_send_request(request);
}

void InputReceiver::callbackTaskParsingOutput(
    const vlm_msgs::msg::TaskParsingOutput::SharedPtr msg) {
  std::vector<FeatureVector> features;
  conversions::convertFromRosFeatureVectors(msg->objects.features, features);
  GlobalInfo::instance().setSearchFeatures(features);
  std::vector<Label> labels;
  labels.reserve(msg->objects.features.size());
  const auto& label_name_to_id_map = GlobalInfo::instance().getLabelNameToIdMap();
  for (const auto& feature : msg->objects.names) {
    auto it = label_name_to_id_map.find(feature);
    if (it != label_name_to_id_map.end()) {
      labels.push_back(it->second);
    } else {
      LOG(WARNING) << "Received feature with unknown label name: " << feature
                   << " Adding it as a new label with a new ID.";
      if (!GlobalInfo::instance().addNewLabels({feature})) {
        LOG(WARNING) << "Could not add new labels to GlobalInfo.";
        return;
      }
    }
  }
  GlobalInfo::instance().setSearchLabels(labels);
  LOG(INFO) << "Set " << features.size() << " features from task parsing output.";
}

void InputReceiver::callbackNewLabels(const hydra_msgs::msg::NewLabels::SharedPtr msg) {
  if (!GlobalInfo::initialized()) {
    LOG(WARNING) << "GlobalInfo not initialized. Cannot add new labels.";
    return;
  }
  if (!GlobalInfo::instance().addNewLabels(msg->new_labels)) {
    LOG(WARNING) << "Could not add new labels to GlobalInfo.";
    return;
  }
  LOG(INFO) << "Added " << msg->new_labels.size() << " new labels. Total labels: "
            << GlobalInfo::instance().getNumActiveLabels();
}

void InputReceiver::callbackReady(const std_msgs::msg::Bool::SharedPtr msg) {
  if (msg->data) {
    auto& agent_state_instance = AgentState::instance();
    if (agent_state_instance.getState() == State::WAITING) {
      agent_state_instance.trySetReady();
      LOG(INFO) << "Agent state set to READY.";
    }
  }
  auto request = std::make_shared<vlm_msgs::srv::TaskParsing::Request>();
  request->task = GlobalInfo::instance().getQuestion();
  task_parsing_client_->async_send_request(request);
  ready_sub_.reset();
}

void InputReceiver::callbackQuestionRoomsEmbeddings(
    const hflex_eqa_msgs::msg::QuestionRoomsEmbeddings::SharedPtr msg) {
  std::vector<FeatureVector> room_embeddings;
  conversions::convertFromRosFeatureVectors(msg->room_embeddings, room_embeddings);
  std::vector<std::string> room_labels = msg->room_labels;
  FeatureVector question_embedding;
  conversions::convertFromRosFeatureVector(msg->question_embedding, question_embedding);

  GlobalInfo::instance().setRoomEmbeddings(room_embeddings, room_labels);
  GlobalInfo::instance().setQuestionEmbedding(question_embedding);

  LOG(INFO) << "Set question and room embeddings from topic.";
}

}  // namespace input
}  // namespace hflex_eqa
