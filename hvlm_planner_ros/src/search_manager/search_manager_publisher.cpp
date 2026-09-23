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
#include "hvlm_planner_ros/search_manager/search_manager_publisher.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/enum.h>
#include <glog/logging.h>
#include <hvlm_planner/common/exploration_bounding_box.h>
#include <hvlm_planner/common/global_info.h>
#include <hvlm_planner/utils/math.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_types.h>

#include <array>

#include <rclcpp/time.hpp>

#include "hvlm_planner_ros/utils/color_parsing.h"

namespace hvlm_planner {

namespace {
constexpr float kBoundingBoxLineWidth = 0.04f;
constexpr float kBoundingBoxColorR = 0.0f;
constexpr float kBoundingBoxColorG = 170.0f / 255.0f;
constexpr float kBoundingBoxColorB = 1.0f;
constexpr float kBoundingBoxColorA = 1.0f;
}  // namespace

void declare_config(SearchManagerPublisher::Config& config) {
  using namespace config;
  name("SearchManagerPublisher::Config");
  field(config.ns, "ns");
  field(config.blacklist_frontier_color, "blacklist_frontier_color");
  field(config.target_frontier_color, "target_frontier_color");
  field(config.active_frontier_color, "active_frontier_color");
  field(config.frontier_scale, "frontier_scale");
  field(config.z_offset, "z_offset");
  field(config.z_text_offset, "z_text_offset");
  field(config.text_scale, "text_scale");
  field(config.direction_length, "direction_length");
  field(config.direction_shaft_diameter, "direction_shaft_diameter");
  field(config.direction_head_diameter, "direction_head_diameter");
  field(config.direction_head_length, "direction_head_length");
}

SearchManagerPublisher::SearchManagerPublisher(const Config& config)
    : config(config),
      nh_(ianvs::NodeHandle::this_node(config.ns)),
      blacklist_frontiers_pub_(
          nh_.create_publisher<visualization_msgs::msg::MarkerArray>(
              "blacklist_frontiers", rclcpp::QoS(1).transient_local())),
      target_frontier_pub_(nh_.create_publisher<visualization_msgs::msg::MarkerArray>(
          "target_frontier", rclcpp::QoS(1).transient_local())),
      active_frontiers_pub_(nh_.create_publisher<visualization_msgs::msg::MarkerArray>(
          "active_frontiers", rclcpp::QoS(1).transient_local())),
      bounding_box_pub_(nh_.create_publisher<visualization_msgs::msg::MarkerArray>(
          "exploration_bounding_box", rclcpp::QoS(1).transient_local())) {
  RCLCPP_INFO(nh_.logger(), "SearchManagerPublisher config:\n%s", printInfo().c_str());
}

std::string SearchManagerPublisher::printInfo() const {
  return config::toString(config);
}

void SearchManagerPublisher::call(const uint64_t timestamp,
                                  const FrontierBlackList& frontier_blacklist,
                                  const Pose& target_frontier,
                                  const spark_dsg::DynamicSceneGraph::Ptr& dsg) const {
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Time(timestamp);
  header.frame_id = GlobalInfo::instance().getFrames().odom;

  visualizeBlackListedFrontiers(frontier_blacklist, header);
  visualizeTargetFrontier(target_frontier, header);
  visualizeActiveFrontiers(dsg, header);
  visualizeBoundingBox(header);
}

void SearchManagerPublisher::visualizeBlackListedFrontiers(
    const FrontierBlackList& frontier_blacklist,
    const std_msgs::msg::Header& header) const {
  visualization_msgs::msg::MarkerArray marker_array;
  // Clean previous markers
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(clear);
  int id = 0;
  for (const auto& frontier : frontier_blacklist) {
    // Frontier pose
    visualization_msgs::msg::Marker frontier_marker;
    frontier_marker.header = header;
    frontier_marker.ns = "blacklist_frontiers";
    frontier_marker.id = id++;
    frontier_marker.type = visualization_msgs::msg::Marker::CUBE;
    frontier_marker.action = visualization_msgs::msg::Marker::ADD;

    frontier_marker.pose.position.x = frontier.translation().x();
    frontier_marker.pose.position.y = frontier.translation().y();
    frontier_marker.pose.position.z =
        frontier.translation().z() + config.z_offset;  // Add z offset
    frontier_marker.pose.orientation.w = 1.0;

    frontier_marker.scale.x = config.frontier_scale;
    frontier_marker.scale.y = config.frontier_scale;
    frontier_marker.scale.z = config.frontier_scale;

    frontier_marker.color.r = config.blacklist_frontier_color.r / 255.0f;
    frontier_marker.color.g = config.blacklist_frontier_color.g / 255.0f;
    frontier_marker.color.b = config.blacklist_frontier_color.b / 255.0f;
    frontier_marker.color.a = config.blacklist_frontier_color.a / 255.0f;

    marker_array.markers.push_back(frontier_marker);

    // Direction arrow
    visualization_msgs::msg::Marker arrow_marker;
    arrow_marker.header = header;
    arrow_marker.ns = "blacklist_frontier_directions";
    arrow_marker.id = id++;
    arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
    arrow_marker.action = visualization_msgs::msg::Marker::ADD;

    Eigen::Vector3d dir = frontier.rotation() * Eigen::Vector3d::UnitX();

    // Arrow geometry: start at centroid, end in direction of frontier
    geometry_msgs::msg::Point start, end;
    start = frontier_marker.pose.position;

    end.x = start.x + config.direction_length * dir.x();
    end.y = start.y + config.direction_length * dir.y();
    end.z = start.z + config.direction_length * dir.z();

    arrow_marker.points.push_back(start);
    arrow_marker.points.push_back(end);

    // Shaft + head sizes
    arrow_marker.scale.x = config.direction_shaft_diameter;  // shaft diameter
    arrow_marker.scale.y = config.direction_head_diameter;   // head diameter
    arrow_marker.scale.z = config.direction_head_length;     // head length

    // Color (same as frontier color)
    arrow_marker.color = frontier_marker.color;

    marker_array.markers.push_back(arrow_marker);

    // Add text marker with b(id)
    visualization_msgs::msg::Marker text_marker;
    text_marker.header = header;
    text_marker.ns = "blacklist_frontier_ids";
    text_marker.id = id++;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose.position = start;
    text_marker.pose.position.z += config.z_text_offset;  // Slightly above the cube
    text_marker.pose.orientation.w = 1.0;
    text_marker.scale.z = config.text_scale;  // Only z scale is used for text size
    text_marker.color.a = 1.0;                // Black color
    text_marker.color.r = 0.0f;
    text_marker.color.g = 0.0f;
    text_marker.color.b = 0.0f;
    text_marker.text = "b(" + std::to_string(static_cast<size_t>(id / 3)) +
                       ")";  // Assuming each frontier has 3 markers (cube, arrow, text)
    marker_array.markers.push_back(text_marker);
  }
  blacklist_frontiers_pub_->publish(marker_array);
}

void SearchManagerPublisher::visualizeTargetFrontier(
    const Pose& target_frontier, const std_msgs::msg::Header& header) const {
  visualization_msgs::msg::MarkerArray marker_array;
  // Clean previous markers
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(clear);
  int id = 0;

  // Frontier pose
  visualization_msgs::msg::Marker frontier_marker;
  frontier_marker.header = header;
  frontier_marker.ns = "target_frontier";
  frontier_marker.id = id++;
  frontier_marker.type = visualization_msgs::msg::Marker::CUBE;
  frontier_marker.action = visualization_msgs::msg::Marker::ADD;

  frontier_marker.pose.position.x = target_frontier.translation().x();
  frontier_marker.pose.position.y = target_frontier.translation().y();
  frontier_marker.pose.position.z =
      target_frontier.translation().z() + config.z_offset;  // Add z offset
  frontier_marker.pose.orientation.w = 1.0;

  frontier_marker.scale.x = config.frontier_scale;
  frontier_marker.scale.y = config.frontier_scale;
  frontier_marker.scale.z = config.frontier_scale;

  frontier_marker.color.r = config.target_frontier_color.r / 255.0f;
  frontier_marker.color.g = config.target_frontier_color.g / 255.0f;
  frontier_marker.color.b = config.target_frontier_color.b / 255.0f;
  frontier_marker.color.a = config.target_frontier_color.a / 255.0f;

  marker_array.markers.push_back(frontier_marker);

  // Direction arrow
  visualization_msgs::msg::Marker arrow_marker;
  arrow_marker.header = header;
  arrow_marker.ns = "target_frontier_direction";
  arrow_marker.id = id++;
  arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
  arrow_marker.action = visualization_msgs::msg::Marker::ADD;

  Eigen::Vector3d dir = target_frontier.rotation() * Eigen::Vector3d::UnitX();

  // Arrow geometry: start at centroid, end in direction of frontier
  geometry_msgs::msg::Point start, end;
  start = frontier_marker.pose.position;

  end.x = start.x + config.direction_length * dir.x();
  end.y = start.y + config.direction_length * dir.y();
  end.z = start.z + config.direction_length * dir.z();

  arrow_marker.points.push_back(start);
  arrow_marker.points.push_back(end);

  // Shaft + head sizes
  arrow_marker.scale.x = config.direction_shaft_diameter;  // shaft diameter
  arrow_marker.scale.y = config.direction_head_diameter;   // head diameter
  arrow_marker.scale.z = config.direction_head_length;     // head length

  // Color (same as frontier color)
  arrow_marker.color = frontier_marker.color;
  marker_array.markers.push_back(arrow_marker);

  // Text marker with "Target"
  visualization_msgs::msg::Marker text_marker;
  text_marker.header = header;
  text_marker.ns = "target_frontier_text";
  text_marker.id = id++;
  text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker.action = visualization_msgs::msg::Marker::ADD;
  text_marker.pose.position = start;
  text_marker.pose.position.z += config.z_text_offset;  // Slightly above the cube
  text_marker.pose.orientation.w = 1.0;
  text_marker.scale.z = config.text_scale;  // Only z scale is used for text size
  text_marker.color.a = 1.0;                // Black color
  text_marker.color.r = 0.0f;
  text_marker.color.g = 0.0f;
  text_marker.color.b = 0.0f;
  text_marker.text = "Target";
  marker_array.markers.push_back(text_marker);
  target_frontier_pub_->publish(marker_array);
}

void SearchManagerPublisher::visualizeActiveFrontiers(
    const spark_dsg::DynamicSceneGraph::Ptr& dsg,
    const std_msgs::msg::Header& header) const {
  visualization_msgs::msg::MarkerArray marker_array;
  // Clean previous markers
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(clear);
  int id = 0;

  if (!dsg->hasLayer(spark_dsg::DsgLayers::FRONTIERS)) {
    active_frontiers_pub_->publish(
        marker_array);  // Publish empty marker array to clear old markers
    return;
  }
  const auto& frontiers_layer = dsg->getLayer(spark_dsg::DsgLayers::FRONTIERS);
  for (const auto& [node_id, node] : frontiers_layer.nodes()) {
    const auto& pos = node->attributes<spark_dsg::NodeAttributes>().position;
    const auto& dir =
        node->attributes<spark_dsg::GlobalFrontierNodeAttributes>().direction;
    Pose frontier_pose = Pose::Identity();
    frontier_pose.translation() = pos;

    // Frontier pose
    visualization_msgs::msg::Marker frontier_marker;
    frontier_marker.header = header;
    frontier_marker.ns = "active_frontiers";
    frontier_marker.id = id++;
    frontier_marker.type = visualization_msgs::msg::Marker::CUBE;
    frontier_marker.action = visualization_msgs::msg::Marker::ADD;

    frontier_marker.pose.position.x = frontier_pose.translation().x();
    frontier_marker.pose.position.y = frontier_pose.translation().y();
    frontier_marker.pose.position.z =
        frontier_pose.translation().z() + config.z_offset;  // Add z offset
    frontier_marker.pose.orientation.w = 1.0;

    frontier_marker.scale.x = config.frontier_scale;
    frontier_marker.scale.y = config.frontier_scale;
    frontier_marker.scale.z = config.frontier_scale;

    // Color active frontiers in light gray
    frontier_marker.color.r = config.active_frontier_color.r / 255.0f;
    frontier_marker.color.g = config.active_frontier_color.g / 255.0f;
    frontier_marker.color.b = config.active_frontier_color.b / 255.0f;
    frontier_marker.color.a = config.active_frontier_color.a / 255.0f;

    marker_array.markers.push_back(frontier_marker);

    // Direction arrow
    visualization_msgs::msg::Marker arrow_marker;
    arrow_marker.header = header;
    arrow_marker.ns = "active_frontier_directions";
    arrow_marker.id = id++;
    arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
    arrow_marker.action = visualization_msgs::msg::Marker::ADD;

    // Arrow geometry: start at centroid, end in direction of frontier
    geometry_msgs::msg::Point start, end;
    start = frontier_marker.pose.position;

    end.x = start.x + config.direction_length * dir.x();
    end.y = start.y + config.direction_length * dir.y();
    end.z = start.z;

    arrow_marker.points.push_back(start);
    arrow_marker.points.push_back(end);

    arrow_marker.scale.x = config.direction_shaft_diameter;  // shaft diameter
    arrow_marker.scale.y = config.direction_head_diameter;   // head diameter
    arrow_marker.scale.z = config.direction_head_length;     // head length

    // Color (same as frontier color)
    arrow_marker.color = frontier_marker.color;
    marker_array.markers.push_back(arrow_marker);

    // Add text marker with node id
    visualization_msgs::msg::Marker text_marker;
    text_marker.header = header;
    text_marker.ns = "active_frontier_ids";
    text_marker.id = id++;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose.position = start;
    text_marker.pose.position.z += config.z_text_offset * 3;  // Slightly above the cube
    text_marker.pose.orientation.w = 1.0;
    text_marker.scale.z = config.text_scale;  // Only z scale is used for text size
    text_marker.color.a = 1.0;                // Black color
    text_marker.color.r = 0.0f;
    text_marker.color.g = 0.0f;
    text_marker.color.b = 0.0f;
    text_marker.text =
        "active\n" + spark_dsg::NodeSymbol(node_id).str() + "\n" +
        spark_dsg::NodeSymbol(
            node->attributes<spark_dsg::GlobalFrontierNodeAttributes>().connected_nav)
            .str();
    marker_array.markers.push_back(text_marker);
  }
  active_frontiers_pub_->publish(marker_array);
}

void SearchManagerPublisher::visualizeBoundingBox(
    const std_msgs::msg::Header& header) const {
  visualization_msgs::msg::MarkerArray marker_array;
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(clear);

  const auto& bounding_box = ExplorationBoundingBox::instance().config();
  if (!bounding_box.enabled) {
    bounding_box_pub_->publish(marker_array);
    return;
  }

  const auto point = [](double x, double y, double z) {
    geometry_msgs::msg::Point p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
  };

  const auto& min = bounding_box.min;
  const auto& max = bounding_box.max;
  const std::array<geometry_msgs::msg::Point, 8> corners = {
      point(min[0], min[1], min[2]),
      point(max[0], min[1], min[2]),
      point(max[0], max[1], min[2]),
      point(min[0], max[1], min[2]),
      point(min[0], min[1], max[2]),
      point(max[0], min[1], max[2]),
      point(max[0], max[1], max[2]),
      point(min[0], max[1], max[2])};
  constexpr std::array<std::array<size_t, 2>, 12> edges = {std::array<size_t, 2>{0, 1},
                                                           std::array<size_t, 2>{1, 2},
                                                           std::array<size_t, 2>{2, 3},
                                                           std::array<size_t, 2>{3, 0},
                                                           std::array<size_t, 2>{4, 5},
                                                           std::array<size_t, 2>{5, 6},
                                                           std::array<size_t, 2>{6, 7},
                                                           std::array<size_t, 2>{7, 4},
                                                           std::array<size_t, 2>{0, 4},
                                                           std::array<size_t, 2>{1, 5},
                                                           std::array<size_t, 2>{2, 6},
                                                           std::array<size_t, 2>{3, 7}};

  visualization_msgs::msg::Marker box_marker;
  box_marker.header = header;
  box_marker.ns = "exploration_bounding_box";
  box_marker.id = 0;
  box_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  box_marker.action = visualization_msgs::msg::Marker::ADD;
  box_marker.pose.orientation.w = 1.0;
  box_marker.scale.x = kBoundingBoxLineWidth;
  box_marker.color.r = kBoundingBoxColorR;
  box_marker.color.g = kBoundingBoxColorG;
  box_marker.color.b = kBoundingBoxColorB;
  box_marker.color.a = kBoundingBoxColorA;

  for (const auto& edge : edges) {
    box_marker.points.push_back(corners[edge[0]]);
    box_marker.points.push_back(corners[edge[1]]);
  }
  marker_array.markers.push_back(box_marker);
  bounding_box_pub_->publish(marker_array);
}

}  // namespace hvlm_planner
