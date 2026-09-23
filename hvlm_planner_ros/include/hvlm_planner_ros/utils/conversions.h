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

#include <hvlm_planner/common/input.h>
#include <hvlm_planner/common/types.h>

#include <vector>

#include <hvlm_planner_msgs/msg/feature_vector.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <opencv2/core/mat.hpp>
#include <semantic_inference_msgs/msg/feature_vector.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace hvlm_planner {
namespace conversions {
void convertFromRosOccupancyGrid(const nav_msgs::msg::OccupancyGrid& ros_occupancy_grid,
                                 input::OccupancyGrid::Ptr& grid);

nav_msgs::msg::OccupancyGrid convertToRosOccupancyGrid(
    const input::OccupancyGrid& grid);

void convertFromRosFeatureVector(
    const semantic_inference_msgs::msg::FeatureVector& ros_feature_vector,
    FeatureVector& feature_vector);

void convertFromRosFeatureVector(
    const hvlm_planner_msgs::msg::FeatureVector& ros_feature_vector,
    FeatureVector& feature_vector);

void convertFromRosFeatureVectors(
    const std::vector<semantic_inference_msgs::msg::FeatureVector>& ros_feature_vectors,
    std::vector<FeatureVector>& feature_vectors);

void convertFromRosFeatureVectors(
    const std::vector<hvlm_planner_msgs::msg::FeatureVector>& ros_feature_vectors,
    std::vector<FeatureVector>& feature_vectors);

void convertFromRosImage(const sensor_msgs::msg::Image::ConstSharedPtr ros_image,
                         cv::Mat& image);

}  // namespace conversions
}  // namespace hvlm_planner
