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
#include "hvlm_planner_ros/utils/conversions.h"

#include <glog/logging.h>

#include <cv_bridge/cv_bridge.hpp>

namespace hvlm_planner {
namespace conversions {

void convertFromRosOccupancyGrid(const nav_msgs::msg::OccupancyGrid& ros_occupancy_grid,
                                 input::OccupancyGrid::Ptr& grid) {
  if (!grid) {
    grid = std::make_shared<input::OccupancyGrid>();
  }
  grid->clear();

  // Set resolution
  grid->resolution = ros_occupancy_grid.info.resolution;
  grid->bounds.width = ros_occupancy_grid.info.width;
  grid->bounds.height = ros_occupancy_grid.info.height;
  grid->bounds.min_col =
      static_cast<int>(ros_occupancy_grid.info.origin.position.x / grid->resolution);
  grid->bounds.min_row =
      static_cast<int>(ros_occupancy_grid.info.origin.position.y / grid->resolution);

  // ROS OccupancyGrid data is row-major, starting at (0,0) in the map frame
  for (size_t i = 0; i < ros_occupancy_grid.data.size(); ++i) {
    const auto& state = ros_occupancy_grid.data[i];
    const float pose_x =
        ros_occupancy_grid.info.origin.position.x +
        (static_cast<float>(i % ros_occupancy_grid.info.width) + 0.5f) *
            grid->resolution;
    const float pose_y =
        ros_occupancy_grid.info.origin.position.y +
        (static_cast<float>(i / ros_occupancy_grid.info.width) + 0.5f) *
            grid->resolution;
    input::GridIndex idx = grid->worldToGrid(input::Point2f(pose_x, pose_y));
    grid->grid[idx] = static_cast<input::CellState>(state);
  }
}

nav_msgs::msg::OccupancyGrid convertToRosOccupancyGrid(
    const input::OccupancyGrid& grid) {
  nav_msgs::msg::OccupancyGrid ros_grid;
  ros_grid.info.resolution = grid.resolution;

  ros_grid.info.width = grid.bounds.width;
  ros_grid.info.height = grid.bounds.height;

  ros_grid.info.origin.position.x = grid.bounds.min_col * grid.resolution;
  ros_grid.info.origin.position.y = grid.bounds.min_row * grid.resolution;
  ros_grid.info.origin.position.z = -2.0;
  ros_grid.info.origin.orientation.w = 1.0;

  ros_grid.data.assign(grid.bounds.width * grid.bounds.height,
                       static_cast<int8_t>(input::CellState::Unknown));

  // Fill data
  for (const auto& [idx, state] : grid.grid) {
    const int row = idx.first - grid.bounds.min_row;
    const int col = idx.second - grid.bounds.min_col;
    const int index = row * grid.bounds.width + col;
    ros_grid.data[index] = static_cast<int8_t>(state);
  }
  return ros_grid;
}

void convertFromRosFeatureVector(
    const semantic_inference_msgs::msg::FeatureVector& ros_feature_vector,
    FeatureVector& feature_vector) {
  feature_vector = Eigen::Map<const FeatureVector>(ros_feature_vector.data.data(),
                                                   ros_feature_vector.data.size());
}

void convertFromRosFeatureVector(
    const hvlm_planner_msgs::msg::FeatureVector& ros_feature_vector,
    FeatureVector& feature_vector) {
  feature_vector = Eigen::Map<const FeatureVector>(ros_feature_vector.feature.data(),
                                                   ros_feature_vector.feature.size());
}

void convertFromRosFeatureVectors(
    const std::vector<semantic_inference_msgs::msg::FeatureVector>& ros_feature_vectors,
    std::vector<FeatureVector>& feature_vectors) {
  feature_vectors.clear();
  feature_vectors.reserve(ros_feature_vectors.size());

  for (const auto& ros_fv : ros_feature_vectors) {
    // Eigen mapping to convert std::vector to Eigen::VectorXf
    FeatureVector fv;
    convertFromRosFeatureVector(ros_fv, fv);
    feature_vectors.push_back(fv);
  }
}

void convertFromRosFeatureVectors(
    const std::vector<hvlm_planner_msgs::msg::FeatureVector>& ros_feature_vectors,
    std::vector<FeatureVector>& feature_vectors) {
  feature_vectors.clear();
  feature_vectors.reserve(ros_feature_vectors.size());

  for (const auto& ros_fv : ros_feature_vectors) {
    // Eigen mapping to convert std::vector to Eigen::VectorXf
    FeatureVector fv;
    convertFromRosFeatureVector(ros_fv, fv);
    feature_vectors.push_back(fv);
  }
}

void convertFromRosImage(const sensor_msgs::msg::Image::ConstSharedPtr ros_image,
                         cv::Mat& image) {
  try {
    image = cv_bridge::toCvCopy(ros_image, sensor_msgs::image_encodings::RGB8)->image;
  } catch (cv_bridge::Exception& e) {
    LOG(ERROR) << "cv_bridge exception: " << e.what();
    image = cv::Mat();  // Return an empty image on failure
  }
}

}  // namespace conversions
}  // namespace hvlm_planner
