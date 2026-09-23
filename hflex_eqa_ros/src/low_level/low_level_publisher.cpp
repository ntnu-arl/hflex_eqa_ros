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
#include "hflex_eqa_ros/low_level/low_level_publisher.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <hflex_eqa/common/global_info.h>
#include <hflex_eqa/common/types.h>

#include <rclcpp/time.hpp>

#include "hflex_eqa_ros/utils/color_parsing.h"

namespace hflex_eqa {

void declare_config(LowLevelPublisher::Config& config) {
  using namespace config;
  name("LowLevelPublisher::Config");
  field(config.ns, "ns");
  field(config.point_scale, "point_scale");
  field(config.line_width, "line_width");
  field(config.poses_color, "poses_color");
  field(config.connecting_line_color, "connecting_line_color");
  field(config.downsample_waypoints_factor, "downsample_waypoints_factor");
}

LowLevelPublisher::LowLevelPublisher(const Config& config)
    : config(config),
      nh_(ianvs::NodeHandle::this_node(config.ns)),
      nav_graph_pub_(
          nh_.create_publisher<visualization_msgs::msg::MarkerArray>("nav_graph", 1)),
      nav_graph_poses_pub_(
          nh_.create_publisher<geometry_msgs::msg::PoseArray>("nav_graph_poses", 1)),
      waypoints_pub_(nh_.create_publisher<nav_msgs::msg::Path>("planned_path", 1)),
      waypoints_poses_pub_(
          nh_.create_publisher<geometry_msgs::msg::PoseArray>("planned_poses", 1)),
      sim_goal_pub_(
          nh_.create_publisher<geometry_msgs::msg::PoseStamped>("sim_goal", 1)) {
  RCLCPP_INFO(
      nh_.logger(), "LowLevelPublisher config:\n%s", config::toString(config).c_str());
}

std::string LowLevelPublisher::printInfo() const { return config::toString(config); }

void LowLevelPublisher::call(LowLevelOutput::Ptr output) const {
  if (!output->success) {
    if (output->valid_goal) {
      geometry_msgs::msg::PoseStamped sim_goal_msg;
      sim_goal_msg.header.frame_id = GlobalInfo::instance().getFrames().odom;
      sim_goal_msg.header.stamp = rclcpp::Time(output->timestamp_ns);
      sim_goal_msg.pose.position.x = output->goal_pose.translation().x();
      sim_goal_msg.pose.position.y = output->goal_pose.translation().y();
      sim_goal_msg.pose.position.z = output->goal_pose.translation().z();
      Eigen::Quaterniond q(output->goal_pose.rotation());
      sim_goal_msg.pose.orientation.x = q.x();
      sim_goal_msg.pose.orientation.y = q.y();
      sim_goal_msg.pose.orientation.z = q.z();
      sim_goal_msg.pose.orientation.w = q.w();
      sim_goal_pub_->publish(sim_goal_msg);
    }
    return;
  }
  visualization_msgs::msg::MarkerArray nav_graph_markers;
  // Clean previous markers
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  nav_graph_markers.markers.push_back(clear);

  std::string odom_frame = GlobalInfo::instance().getFrames().odom;

  // Lambda function to convert from Pose to geometry_msgs::msg::Point
  auto pose_to_point =
      [&](const Pose& pose, const uint64_t timestamp_ns, const int id) {
        visualization_msgs::msg::Marker point;
        point.header.frame_id = odom_frame;
        point.header.stamp = rclcpp::Time(timestamp_ns);
        point.ns = "nav_graph_nodes";
        point.id = id;
        point.type = visualization_msgs::msg::Marker::SPHERE;
        point.action = visualization_msgs::msg::Marker::ADD;
        point.pose.position.x = pose.translation().x();
        point.pose.position.y = pose.translation().y();
        point.pose.position.z = pose.translation().z();
        Eigen::Quaterniond q(pose.rotation());
        point.pose.orientation.x = q.x();
        point.pose.orientation.y = q.y();
        point.pose.orientation.z = q.z();
        point.pose.orientation.w = q.w();

        point.scale.x = config.point_scale;
        point.scale.y = config.point_scale;
        point.scale.z = config.point_scale;

        point.color.r = config.poses_color.r / 255.0f;
        point.color.g = config.poses_color.g / 255.0f;
        point.color.b = config.poses_color.b / 255.0f;
        point.color.a = config.poses_color.a / 255.0f;
        return point;
      };

  // Publish nav graph nodes as markers and PoseArray for rotations
  int id = 0;
  geometry_msgs::msg::PoseArray nav_graph_poses_msg;
  nav_graph_poses_msg.header.frame_id = odom_frame;
  nav_graph_poses_msg.header.stamp = rclcpp::Time(output->timestamp_ns);

  auto marker_point = pose_to_point(output->start_pose, output->timestamp_ns, id);
  id++;
  nav_graph_markers.markers.push_back(marker_point);
  nav_graph_poses_msg.poses.push_back(marker_point.pose);

  for (const auto& pose : output->nav_graph_poses) {
    marker_point = pose_to_point(pose, output->timestamp_ns, id);
    id++;
    nav_graph_markers.markers.push_back(marker_point);
    nav_graph_poses_msg.poses.push_back(marker_point.pose);
  }

  marker_point = pose_to_point(output->goal_pose, output->timestamp_ns, id);
  id++;
  nav_graph_markers.markers.push_back(marker_point);
  nav_graph_poses_msg.poses.push_back(marker_point.pose);

  // Connecting lines between nav graph nodes
  visualization_msgs::msg::Marker line_marker;
  line_marker.header.frame_id = odom_frame;
  line_marker.header.stamp = rclcpp::Time(output->timestamp_ns);
  line_marker.ns = "nav_graph_lines";
  line_marker.id = id++;
  line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  line_marker.action = visualization_msgs::msg::Marker::ADD;
  line_marker.scale.x = config.line_width;
  line_marker.color.r = config.connecting_line_color.r / 255.0f;
  line_marker.color.g = config.connecting_line_color.g / 255.0f;
  line_marker.color.b = config.connecting_line_color.b / 255.0f;
  line_marker.color.a = config.connecting_line_color.a / 255.0f;
  // Start pose
  geometry_msgs::msg::Point start_p;
  start_p.x = output->start_pose.translation().x();
  start_p.y = output->start_pose.translation().y();
  start_p.z = output->start_pose.translation().z();
  line_marker.points.push_back(start_p);
  // Nav graph poses
  for (const auto& pose : output->nav_graph_poses) {
    geometry_msgs::msg::Point p;
    p.x = pose.translation().x();
    p.y = pose.translation().y();
    p.z = pose.translation().z();
    line_marker.points.push_back(p);
  }
  // Goal pose
  geometry_msgs::msg::Point goal_p;
  goal_p.x = output->goal_pose.translation().x();
  goal_p.y = output->goal_pose.translation().y();
  goal_p.z = output->goal_pose.translation().z();
  line_marker.points.push_back(goal_p);
  nav_graph_markers.markers.push_back(line_marker);

  // Publish nav graph poses as PoseArray and markers
  nav_graph_pub_->publish(nav_graph_markers);
  nav_graph_poses_pub_->publish(nav_graph_poses_msg);

  // Publish waypoints as Path and PoseArray
  nav_msgs::msg::Path waypoints_msg;
  waypoints_msg.header.frame_id = odom_frame;
  waypoints_msg.header.stamp = rclcpp::Time(output->timestamp_ns);
  geometry_msgs::msg::PoseArray waypoints_poses_msg;
  waypoints_poses_msg.header.frame_id = odom_frame;
  waypoints_poses_msg.header.stamp = rclcpp::Time(output->timestamp_ns);
  for (size_t i = 0; i < output->waypoints.size(); ++i) {
    const auto& waypoint = output->waypoints[i];
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = waypoint.translation().x();
    pose_stamped.pose.position.y = waypoint.translation().y();
    pose_stamped.pose.position.z = waypoint.translation().z();
    Eigen::Quaterniond q(waypoint.rotation());
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();
    waypoints_msg.poses.push_back(pose_stamped);
    if (i % config.downsample_waypoints_factor != 0) {
      continue;
    }
    waypoints_poses_msg.poses.push_back(pose_stamped.pose);
  }
  waypoints_pub_->publish(waypoints_msg);
  waypoints_poses_pub_->publish(waypoints_poses_msg);
}

}  // namespace hflex_eqa
