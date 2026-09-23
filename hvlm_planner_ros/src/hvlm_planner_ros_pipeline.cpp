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
#include "hvlm_planner_ros/hvlm_planner_ros_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/context.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <ianvs/node_handle.h>

namespace hvlm_planner {

void declare_config(HVlmPlannerRosPipeline::ManagerCallableNames& config) {
  using namespace config;
  name("ManagerCallableNames");
  field(config.eqa_callable, "eqa_callable");
  field(config.hlp_callable, "hlp_callable");
  field(config.finish_callable, "finish_callable");
  field(config.eqa_planner_callable, "eqa_planner_callable");
}

void declare_config(HVlmPlannerRosPipeline::Config& config) {
  using namespace config;
  name("HVlmPlannerRosConfig");
  field(config.low_level, "low_level");
  field(config.search_manager, "search_manager");
  field(config.manager_callable_names, "manager_callable_names");
  field(config.input_receiver, "input_receiver");
  field(config.manager_ros_interface, "manager_ros_interface");
  field(config.verbosity, "verbosity");
  field(config.preprint_config, "preprint_config");
}

HVlmPlannerRosPipeline::HVlmPlannerRosPipeline(int robot_id, int config_verbosity)
    : HVlmPlannerPipeline(
          config::fromContext<PipelineConfig>(), robot_id, config_verbosity),
      config(config::checkValid(config::fromContext<Config>())) {
  if (config.preprint_config) {
    LOG(INFO) << "Using configuration to start HVlmPlanner\n"
              << config::toString(config);
  } else {
    LOG_IF(INFO, config.verbosity >= 1)
        << "Starting HVlmPlanner-ROS with input configuration\n"
        << config::toString(config.input_receiver);
  }
}

HVlmPlannerRosPipeline::~HVlmPlannerRosPipeline() {}

void HVlmPlannerRosPipeline::init() {
  auto nh = ianvs::NodeHandle::this_node("~");

  low_level_module_ = config.low_level.create();
  modules_["low_level"] = CHECK_NOTNULL(low_level_module_);

  search_manager_module_ = config.search_manager.create();
  modules_["search_manager"] = CHECK_NOTNULL(search_manager_module_);
  // ROS interface for search manager
  manager_ros_interface_ =
      std::make_shared<ManagerRosInterface>(config.manager_ros_interface);
  // search_manager_module_->addFunction(
  //     config.manager_callable_names.hlp_callable,
  //     std::make_unique<FunctionWrapper<void(const input::HLPInput::Ptr&, bool)>>(
  //         [this](const input::HLPInput::Ptr& input, bool flag) {
  //           manager_ros_interface_->publishHLPInfo(input, flag);
  //         }));
  search_manager_module_->addFunction(
      config.manager_callable_names.eqa_planner_callable,
      std::make_unique<FunctionWrapper<void(const input::EQAInput::Ptr&)>>(
          [this](const input::EQAInput::Ptr& input) {
            manager_ros_interface_->publishEQAInfo(input, true);
          }));
  search_manager_module_->addFunction(
      config.manager_callable_names.finish_callable,
      std::make_unique<FunctionWrapper<void(const bool)>>([this](const bool complete) {
        manager_ros_interface_->publishMissionComplete(complete);
      }));

  auto inh = nh / "input";
  input_receiver_ = std::make_shared<input::InputReceiver>(config.input_receiver, inh);
}

void HVlmPlannerRosPipeline::start() { HVlmPlannerPipeline::start(); }

void HVlmPlannerRosPipeline::stop() {
  low_level_module_->stop();
  HVlmPlannerPipeline::stop();
}
}  // namespace hvlm_planner
