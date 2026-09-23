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
#include <config_utilities/dynamic_config.h>
#include <hvlm_planner/low_level/low_level_module.h>
#include <ianvs/node_handle.h>
#include <spark_dsg/color.h>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/publisher.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace hvlm_planner {

class LowLevelPublisher : public LowLevelModule::Sink {
 public:
  struct Config {
    std::string ns = "~/low_level";

    float point_scale = 0.1f;  // sphere radius
    float line_width = 0.05f;
    spark_dsg::Color poses_color = spark_dsg::Color::blue();
    spark_dsg::Color connecting_line_color = spark_dsg::Color::gray();
    size_t downsample_waypoints_factor = 1;
  } const config;

  explicit LowLevelPublisher(const Config& config);

  virtual ~LowLevelPublisher() = default;

  std::string printInfo() const override;

  void call(LowLevelOutput::Ptr output) const override;

 protected:
  ianvs::NodeHandle nh_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr nav_graph_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr nav_graph_poses_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr waypoints_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr waypoints_poses_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr sim_goal_pub_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<LowLevelModule::Sink, LowLevelPublisher, Config>(
          "LowLevelPublisher");
};

void declare_config(LowLevelPublisher::Config& config);

}  // namespace hvlm_planner
