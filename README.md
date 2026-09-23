# HFLEX-EQA ROS 2

ROS 2 Jazzy interfaces for HFLEX-EQA, including planner nodes, RViz integration, Habitat launch files, benchmark orchestration, and the reusable real-robot scene-graph/planning launches. See the [`hflex_eqa` README](https://github.com/ntnu-arl/hflex_eqa/tree/main) for the complete Docker installation and experiment workflow.

## Packages

- `hflex_eqa_msgs`: planner messages and services.
- `hflex_eqa_ros`: C++/Python ROS wrappers and launch/config files.
- `hflex_eqa_ui`: RViz controls for questions, choices, and planner actions.
- `simulation_manager_ros`: OpenEQA/ExploreEQA episode runner and HM3D utilities.

## Build

Import this repository through `hflex_eqa/install/default.repos` or `thor.repos`, then build from the workspace root:

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --continue-on-error --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Habitat simulation

Run a single episode with:

```bash
ros2 launch hflex_eqa_ros habitat_eqa.launch.yaml \
  scene_file:=/developer/hm3d/val/<scene>/<scene>.basis.glb \
  question:="What color is the microwave?"
```

The launch composes Habitat, semantic inference, Hydra, task parsing, and the hierarchical planner. Relevant configuration is in:

- `hflex_eqa_ros/config/habitat/` for ROS topics and node behavior;
- `hflex_eqa/config/habitat/` for planner behavior, VLMs, and prompts;
- `simulation_manager_ros/config/habitat_open_eqa.yaml` for dataset paths and episode selection.

For a benchmark loop, edit a copy of the simulation-manager config and run:

```bash
ros2 launch simulation_manager_ros simulate.launch.yaml \
  simulation_manager_config:=/path/to/experiment.yaml
```

Set `data.dataset` to `openeqa` or `exploreeqa`. Results are written beneath the configured `log_folder`.

## Real robot

The two top-level launches were extracted from the deployment-specific ANYmal repository so that HFLEX-EQA can be deployed independently:

```bash
ros2 launch hflex_eqa_ros scene_graph.launch.yaml
ros2 launch hflex_eqa_ros eqa.launch.yaml
```

`scene_graph.launch.yaml` expects RealSense color, aligned depth, and camera-info topics under `/camera/camera/`, plus TF frames `world`, `body`, and `camera_link`. Override remappings for other sensors. `eqa.launch.yaml` starts the high- and low-level planners and task parser. It loads `config/anymal/eqa_floorplan.json` by default; pass `floorplan_json_path` for a new building, or set `floorplan_source:=list` with `floorplan_nodes` and `floorplan_edges`.

The example floorplan is the weak topological prior used for the paper deployment: it contains room labels and connectivity, not metric geometry or robot localization. Keep hardware calibration and odometry configuration in the relevant sensor/odometry packages; these launches consume their ROS topics and transforms.

## Tests and formatting

Run package tests with `colcon test --packages-select hflex_eqa_ros hflex_eqa_ui simulation_manager_ros`, inspect them with `colcon test-result --verbose`, and run `pre-commit run --all-files` before submitting changes.

## License

BSD 3-Clause. See package manifests for maintainers and dependency licenses.
