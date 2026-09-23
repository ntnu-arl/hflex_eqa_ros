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
#pragma once
#include <config_utilities/virtual_config.h>
#include <hvlm_planner/common/input.h>
#include <hvlm_planner/common/types.h>
#include <ianvs/node_handle.h>

#include <vector>

#include <hvlm_planner_msgs/msg/question_rooms_embeddings.hpp>
#include <hydra_msgs/msg/new_labels.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <opencv2/core/mat.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>
#include <vlm_msgs/msg/task_parsing_output.hpp>
#include <vlm_msgs/srv/task_parsing.hpp>

#include "hvlm_planner_ros/io/graph_wrapper.h"

namespace hvlm_planner {
namespace input {

class InputReceiver {
 public:
  struct Config {
    config::VirtualConfig<io::GraphWrapper> graph;
  } const config;

  InputReceiver(const Config& config, ianvs::NodeHandle nh);

 private:
  void callbackOccupancyGrid(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void callbackImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void callbackQuestion(const std_msgs::msg::String::SharedPtr msg);
  void callbackTaskParsingOutput(const vlm_msgs::msg::TaskParsingOutput::SharedPtr msg);
  void callbackNewLabels(const hydra_msgs::msg::NewLabels::SharedPtr msg);
  void callbackReady(const std_msgs::msg::Bool::SharedPtr msg);
  void callbackQuestionRoomsEmbeddings(
      const hvlm_planner_msgs::msg::QuestionRoomsEmbeddings::SharedPtr msg);

  ianvs::NodeHandle nh_;
  io::GraphWrapper::Ptr graph_wrapper_;
  cv::Mat latest_image_;

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr image_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr ready_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr question_sub_;
  rclcpp::Client<vlm_msgs::srv::TaskParsing>::SharedPtr task_parsing_client_;
  rclcpp::Subscription<vlm_msgs::msg::TaskParsingOutput>::SharedPtr
      task_parsing_output_sub_;
  rclcpp::Subscription<hydra_msgs::msg::NewLabels>::SharedPtr new_labels_sub_;
  rclcpp::Subscription<hvlm_planner_msgs::msg::QuestionRoomsEmbeddings>::SharedPtr
      question_rooms_embeddings_sub_;
};

void declare_config(InputReceiver::Config& config);

}  // namespace input
}  // namespace hvlm_planner
