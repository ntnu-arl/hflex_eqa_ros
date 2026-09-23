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
#include "hvlm_planner_ros/search_manager/monitor_publisher.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <hvlm_planner/common/global_info.h>

namespace hvlm_planner {

void declare_config(MonitorPublisher::Config& config) {
  using namespace config;
  name("MonitorPublisher::Config");
  field(config.ns, "ns");
}

MonitorPublisher::MonitorPublisher(const Config& config)
    : config(config),
      nh_(ianvs::NodeHandle::this_node(config.ns)),
      publisher_(nh_.template create_publisher<hvlm_planner_msgs::msg::Monitor>(
          "monitor", rclcpp::QoS(10).transient_local())) {
  RCLCPP_INFO(nh_.logger(), "MonitorPublisher config:\n%s", printInfo().c_str());
}

std::string MonitorPublisher::printInfo() const { return config::toString(config); }

void MonitorPublisher::call(const uint64_t timestamp_ns,
                            const State& state,
                            const size_t iteration) const {
  hvlm_planner_msgs::msg::Monitor msg;
  msg.header.stamp = rclcpp::Time(timestamp_ns);
  msg.header.frame_id = GlobalInfo::instance().getFrames().odom;
  if (state_utils::state_to_string.find(state) != state_utils::state_to_string.end()) {
    msg.state = state_utils::state_to_string.at(state);
  } else {
    msg.state = "UNKNOWN";
  }
  msg.iteration = iteration;
  publisher_->publish(msg);
}
}  // namespace hvlm_planner
