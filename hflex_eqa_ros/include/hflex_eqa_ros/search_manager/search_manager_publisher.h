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
#include <hflex_eqa/common/types.h>
#include <hflex_eqa/search_manager/sinks.h>
#include <ianvs/node_handle.h>
#include <spark_dsg/color.h>

#include <rclcpp/publisher.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace hflex_eqa {

class SearchManagerPublisher : public managers::Sink {
 public:
  struct Config {
    std::string ns = "~/search_manager";
    spark_dsg::Color blacklist_frontier_color = spark_dsg::Color::black();
    spark_dsg::Color target_frontier_color = spark_dsg::Color::green();
    spark_dsg::Color active_frontier_color = spark_dsg::Color::purple();
    float frontier_scale = 0.3f;  // cube_size
    double z_offset = 0.5;        // meters above the frontier to place the marker
    double z_text_offset = 0.2;   // additional offset for text marker
    double text_scale = 0.2;      // scale for text marker
    float direction_length = 0.75f;
    float direction_shaft_diameter = 0.05f;
    float direction_head_diameter = 0.10f;
    float direction_head_length = 0.15f;
  } const config;

  explicit SearchManagerPublisher(const Config& config);

  virtual ~SearchManagerPublisher() = default;

  std::string printInfo() const override;

  void call(const uint64_t timestamp,
            const FrontierBlackList& frontier_blacklist,
            const Pose& target_frontier,
            const spark_dsg::DynamicSceneGraph::Ptr&) const override;

 protected:
  void visualizeBlackListedFrontiers(const FrontierBlackList& frontier_blacklist,
                                     const std_msgs::msg::Header& header) const;
  void visualizeTargetFrontier(const Pose& target_frontier,
                               const std_msgs::msg::Header& header) const;
  void visualizeActiveFrontiers(const spark_dsg::DynamicSceneGraph::Ptr& dsg,
                                const std_msgs::msg::Header& header) const;
  void visualizeBoundingBox(const std_msgs::msg::Header& header) const;

  ianvs::NodeHandle nh_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      blacklist_frontiers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      target_frontier_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      active_frontiers_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bounding_box_pub_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<managers::Sink, SearchManagerPublisher, Config>(
          "SearchManagerPublisher");
};

void declare_config(SearchManagerPublisher::Config& config);

}  // namespace hflex_eqa
