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
#include <config_utilities/config_utilities.h>
#include <config_utilities/external_registry.h>
#include <config_utilities/formatting/asl.h>
#include <config_utilities/logging/log_to_glog.h>
#include <config_utilities/parsing/context.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/path.h>
#include <config_utilities_ros/ros_dynamic_config_server.h>
#include <glog/logging.h>
#include <ianvs/node_init.h>
#include <ianvs/spin_functions.h>

#include "hflex_eqa_ros/hflex_eqa_ros_pipeline.h"

namespace hflex_eqa {

struct NodeSettings {
  bool show_run_settings = true;
  size_t robot_id = 0;
  bool exit_after_clock = false;
  std::vector<std::string> paths;
  int config_verbosity = 1;
  bool forward_glog_to_ros = true;
  int glog_level = 0;
  int glog_verbosity = 0;
};

void declare_config(NodeSettings& config) {
  using namespace config;
  name("NodeSettings");
  field(config.show_run_settings, "show_run_settings");
  field(config.robot_id, "robot_id");
  field(config.exit_after_clock, "exit_after_clock");
  field(config.paths, "paths");
  field(config.config_verbosity, "config_verbosity");
  field(config.forward_glog_to_ros, "forward_glog_to_ros");
  field(config.glog_level, "glog_level");
  field(config.glog_verbosity, "glog_verbosity");
}

struct RosSink : google::LogSink {
  explicit RosSink(const rclcpp::Logger& logger) : logger_(logger) {}

  void send(google::LogSeverity severity,
            const char* /*full_filename*/,
            const char* base_filename,
            int line,
            const struct ::tm* /*time*/,
            const char* message,
            size_t message_len) override {
    std::stringstream ss;
    ss << "[" << base_filename << ":" << line << "] "
       << std::string(message, message_len);
    switch (severity) {
      case google::GLOG_WARNING:
        RCLCPP_WARN_STREAM(logger_, ss.str());
        break;
      case google::GLOG_ERROR:
        RCLCPP_ERROR_STREAM(logger_, ss.str());
        break;
      case google::GLOG_FATAL:
        RCLCPP_FATAL_STREAM(logger_, ss.str());
        break;
      case google::GLOG_INFO:
      default:
        RCLCPP_INFO_STREAM(logger_, ss.str());
        break;
    }
  }

  rclcpp::Logger logger_;
};

}  // namespace hflex_eqa

int main(int argc, char* argv[]) {
  config::initContext(argc, argv, true);
  config::setConfigSettingsFromContext();
  const auto node_settings = config::fromContext<hflex_eqa::NodeSettings>();

  FLAGS_minloglevel = node_settings.glog_level;
  FLAGS_v = node_settings.glog_verbosity;
  FLAGS_logtostderr = node_settings.forward_glog_to_ros ? 0 : 1;
  FLAGS_colorlogtostderr = 1;

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  [[maybe_unused]] const auto node = ianvs::init_node(argc, argv, "hflex_eqa_ros_node");
  auto nh = ianvs::NodeHandle::this_node();
  const config::RosDynamicConfigServer config_server(nh.node());

  std::shared_ptr<hflex_eqa::RosSink> ros_sink;
  if (node_settings.forward_glog_to_ros) {
    ros_sink = std::make_shared<hflex_eqa::RosSink>(nh.logger());
    google::AddLogSink(ros_sink.get());
  }

  config::Settings().setLogger("glog");
  if (node_settings.show_run_settings) {
    LOG(INFO) << "Using node settings\n" << config::toString(node_settings);
  }

  [[maybe_unused]] const auto plugins =
      config::loadExternalFactories(node_settings.paths);
  {  // start hflex_eqa planner scope
    hflex_eqa::HflexEqaRosPipeline pipeline(node_settings.robot_id,
                                            node_settings.config_verbosity);
    pipeline.init();
    pipeline.start();
    ianvs::spinAndWait(nh, node_settings.exit_after_clock);
    pipeline.stop();
  }  // end hflex_eqa planner scope

  if (ros_sink) {
    google::RemoveLogSink(ros_sink.get());
  }

  rclcpp::shutdown();
  return 0;
}
