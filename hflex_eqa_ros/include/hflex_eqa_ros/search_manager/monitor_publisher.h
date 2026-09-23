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
#include <hflex_eqa/common/agent_state.h>
#include <hflex_eqa/search_manager/sinks.h>
#include <ianvs/node_handle.h>

#include <hflex_eqa_msgs/msg/monitor.hpp>
#include <rclcpp/publisher.hpp>

namespace hflex_eqa {

class MonitorPublisher : public managers::MonitorSink {
 public:
  struct Config {
    std::string ns = "~/search_manager";
  } const config;

  explicit MonitorPublisher(const Config& config);

  virtual ~MonitorPublisher() = default;

  std::string printInfo() const override;

  void call(const uint64_t timestamp_ns,
            const State& state,
            const size_t iteration) const override;

 protected:
  ianvs::NodeHandle nh_;
  rclcpp::Publisher<hflex_eqa_msgs::msg::Monitor>::SharedPtr publisher_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<managers::MonitorSink, MonitorPublisher, Config>(
          "MonitorPublisher");
};

void declare_config(MonitorPublisher::Config& config);

}  // namespace hflex_eqa
