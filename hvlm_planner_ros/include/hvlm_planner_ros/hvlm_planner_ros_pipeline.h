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
#include <config_utilities/virtual_config.h>
#include <hvlm_planner/common/hvlm_pipeline.h>
#include <hvlm_planner/common/module.h>

#include <memory>

#include "hvlm_planner_ros/input/input_receiver.h"
#include "hvlm_planner_ros/search_manager/manager_ros_interface.h"

namespace hvlm_planner {

class HVlmPlannerRosPipeline : public HVlmPlannerPipeline {
 public:
  struct ManagerCallableNames {
    std::string eqa_callable = "call_eqa";
    std::string hlp_callable = "call_hlp";
    std::string finish_callable = "call_finish";
    std::string eqa_planner_callable = "call_eqa_planner";
  };
  struct Config {
    //! @brief Configuration for high-level module
    //! @brief Configuration for search manager module
    config::VirtualConfig<Module> search_manager;
    //! @brief Manager callable names
    ManagerCallableNames manager_callable_names;
    //! @brief Configuration for low-level module
    config::VirtualConfig<Module> low_level;
    //! @brief Receiver config
    input::InputReceiver::Config input_receiver;
    //! @brief Search manager ROS interface config
    ManagerRosInterface::Config manager_ros_interface;
    //! @brief Verbosity setting for main pipeline class
    int verbosity = 1;
    //! @brief Show the config passed to Hydra (before resolving sensor configurations)
    bool preprint_config = false;
  } const config;

  explicit HVlmPlannerRosPipeline(int robot_id = 1, int config_verbosity = 1);

  virtual ~HVlmPlannerRosPipeline();

  void init() override;

  void start() override;

  void stop() override;

 protected:
  std::shared_ptr<Module> low_level_module_;
  std::shared_ptr<Module> search_manager_module_;
  std::shared_ptr<input::InputReceiver> input_receiver_;
  std::shared_ptr<ManagerRosInterface> manager_ros_interface_;
};

void declare_config(HVlmPlannerRosPipeline::ManagerCallableNames& config);
void declare_config(HVlmPlannerRosPipeline::Config& config);

}  // namespace hvlm_planner
