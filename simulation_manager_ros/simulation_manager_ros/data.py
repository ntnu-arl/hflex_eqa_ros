# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Dataset loaders for Habitat EQA experiments."""

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import spark_config as sc
from scipy.spatial.transform import Rotation as R

from hflex_eqa_ros import LoggerInfo

choice_to_number = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass
class DataConfig(sc.Config):
    dataset: str = "openeqa"
    scene_data_path: str = ""
    semantic_annot_data_path: str = ""
    openeqa_question_data_path: str = ""
    openeqa_init_pose_data_path: str = ""
    openeqa_choices_data_path: str = ""
    exploreeqa_question_data_path: str = ""
    exploreeqa_init_pose_data_path: str = ""
    use_semantic_data: bool = False
    use_multifloor_questions: bool = False
    load_test_split_only: bool = False
    test_split_manifest_path: str = ""


def check_if_multifloor(traj: np.ndarray) -> bool:
    """
    Check if the trajectory is multi-floor based on the variance of the y-coordinates.
    :param traj: A numpy array of shape (N, 3) representing the trajectory positions.
    :return: True if the trajectory is multi-floor, False otherwise.
    """
    mean = sum(traj) / traj.shape[0]
    variance = sum((traj - mean) ** 2) / traj.shape[0]
    return np.sqrt(variance) > 0.5


def _read_json(path: str) -> Any:
    with open(path) as file:
        return json.load(file)


def _load_semantic_scenes(config: DataConfig) -> set[str]:
    if not config.semantic_annot_data_path:
        return set()

    semantic_path = Path(config.semantic_annot_data_path)
    if semantic_path.is_file():
        semantic_annots = _read_json(str(semantic_path))
        return set(
            semantic_annots["stages"]["paths"][".glb"][i].split("/*.basis")[0]
            for i in range(len(semantic_annots["stages"]["paths"][".glb"]))
        )

    semantic_scenes = set()
    for split in ["train", "val"]:
        split_dir = semantic_path / split
        if not split_dir.exists():
            continue
        for entry in split_dir.iterdir():
            if entry.is_dir():
                semantic_scenes.add(f"{split}/{entry.name}")
    return semantic_scenes


def _get_first_available(
    row: dict[str, Any],
    keys: list[str],
    default: Any = None,
    required: bool = False,
) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in [None, ""]:
            return value
    if required:
        raise KeyError(f"None of the keys {keys} were found in row: {row}")
    return default


def _split_scene_floor(scene_floor: str) -> tuple[str, str]:
    scene_id, floor = scene_floor.rsplit("_", 1)
    return scene_id, floor


def _infer_scene_split(
    scene_data_path: str, scene_id: str, scene_ref: str = ""
) -> str | None:
    scene_root = Path(scene_data_path)
    candidate_paths = []
    part = -2
    if scene_ref:
        candidate_paths.extend(
            [scene_root / "train" / scene_ref, scene_root / "val" / scene_ref]
        )
        part = -3
    else:
        candidate_paths.extend(
            [scene_root / "train" / scene_id, scene_root / "val" / scene_id]
        )

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate.parts[part]
    return None


def _resolve_scene_file(scene_dir: Path, scene_id: str, scene_ref: str = "") -> str:
    if scene_ref:
        direct_path = (
            scene_dir.parent / scene_ref
            if not scene_ref.startswith(scene_dir.name)
            else scene_dir.parent / scene_ref
        )
        if direct_path.exists():
            return str(direct_path)
        direct_path = scene_dir / Path(scene_ref).name
        if direct_path.exists():
            return str(direct_path)

    preferred_candidates = [
        scene_dir / f"{scene_id}.basis.glb",
        scene_dir / f"{scene_id}.glb",
    ]
    preferred_candidates.extend(
        sorted(
            path
            for path in scene_dir.glob("*.basis.glb")
            if "semantic" not in path.name
        )
    )
    preferred_candidates.extend(
        sorted(path for path in scene_dir.glob("*.glb") if "semantic" not in path.name)
    )

    for candidate in preferred_candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not resolve scene file for scene '{scene_id}' in {scene_dir}"
    )


def _parse_extra_answers(raw_value: Any) -> list[str]:
    if raw_value in [None, ""]:
        return []
    if isinstance(raw_value, list):
        return [str(value) for value in raw_value if str(value)]
    if isinstance(raw_value, str):
        for delimiter in ["|", ";"]:
            if delimiter in raw_value:
                return [
                    value.strip()
                    for value in raw_value.split(delimiter)
                    if value.strip()
                ]
        return [raw_value.strip()] if raw_value.strip() else []
    return [str(raw_value)]


def _quat_xyzw_from_yaw(yaw_rad: float) -> list[float]:
    # Convert ange to quaternion (x, y, z, w) with only yaw rotation.
    # Assume angle axis is (0, 1, 0).
    return R.from_euler("y", yaw_rad).as_quat().tolist()


def _normalize_dataset_name(dataset: str) -> str:
    return dataset.lower()


def _load_test_split_manifest(path: str) -> dict[str, Any]:
    if not path:
        raise ValueError(
            "DataConfig.test_split_manifest_path must be set when "
            "load_test_split_only is enabled."
        )
    return _read_json(path)


def _filter_test_split_episodes(
    config: DataConfig, episodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    manifest = _load_test_split_manifest(config.test_split_manifest_path)
    dataset = _normalize_dataset_name(config.dataset)
    dataset_info = manifest.get("datasets", {}).get(dataset)
    if dataset_info is None:
        raise KeyError(
            f"Dataset '{config.dataset}' was not found in test split manifest "
            f"{config.test_split_manifest_path}."
        )

    selected_ids = {str(question_id) for question_id in dataset_info["question_ids"]}
    filtered_episodes = [
        episode for episode in episodes if str(episode["question_id"]) in selected_ids
    ]
    LoggerInfo.info(
        f"Loaded {len(filtered_episodes)} test-split episodes for {dataset} from "
        f"{config.test_split_manifest_path}."
    )
    return filtered_episodes


def _get_effective_load_settings(
    config: DataConfig, filter_floorplan: bool
) -> tuple[DataConfig, bool]:
    if not config.load_test_split_only:
        return config, filter_floorplan

    manifest = _load_test_split_manifest(config.test_split_manifest_path)
    effective_config = replace(
        config,
        use_multifloor_questions=manifest["use_multifloor_questions"],
        use_semantic_data=manifest["use_semantic_data"],
    )
    effective_filter_floorplan = manifest["filter_floorplan"]
    LoggerInfo.info(
        "Using test split manifest settings: "
        f"filter_floorplan={effective_filter_floorplan}, "
        f"use_multifloor_questions={effective_config.use_multifloor_questions}, "
        f"use_semantic_data={effective_config.use_semantic_data}."
    )
    return effective_config, effective_filter_floorplan


def _load_openeqa_data(
    config: DataConfig, filter_floorplan: bool
) -> list[dict[str, Any]]:
    questions_data = _read_json(config.openeqa_question_data_path)
    init_poses = _read_json(config.openeqa_init_pose_data_path)
    choices = _read_json(config.openeqa_choices_data_path)
    semantic_scenes = _load_semantic_scenes(config)

    episodes = []
    for data in questions_data:
        episode_history = data.get("episode_history", "")
        if "hm3d-v0" not in episode_history:
            continue
        if episode_history not in init_poses:
            continue

        init_pose_data = init_poses[episode_history]
        full_scene_id = init_pose_data["scene_id"]
        scene_id = full_scene_id.split("/")[0]
        split = _infer_scene_split(config.scene_data_path, scene_id, full_scene_id)
        if split is None:
            continue

        has_semantics = f"{split}/{scene_id}" in semantic_scenes
        if config.use_semantic_data and not has_semantics:
            continue
        if not config.use_semantic_data and has_semantics:
            continue

        if not config.use_multifloor_questions:
            full_traj = np.array(init_pose_data.get("full_traj_pos", []), dtype=float)
            if full_traj.size and check_if_multifloor(full_traj[:, 1]):
                continue

        scene_dir = Path(config.scene_data_path) / split / scene_id
        floorplan_path = scene_dir / "regions" / "topological_graph.json"
        if filter_floorplan and not floorplan_path.exists():
            continue

        question_id = str(data["question_id"])
        choice_data = choices.get(question_id, {})
        gt_answers = []
        if choice_data.get("answer"):
            gt_answers.append(choice_data["answer"])
        gt_answers.extend(_parse_extra_answers(data.get("extra_answers", [])))
        gt_answers.extend(_parse_extra_answers(choice_data.get("extra_answers", [])))

        pose = list(init_pose_data["init_pos"])
        init_quat = list(init_pose_data["quat_wxyz"])
        initial_pose = pose + [init_quat[1], init_quat[2], init_quat[3], init_quat[0]]

        episodes.append(
            {
                "question_id": question_id,
                "question": data["question"],
                "category": data.get("category", "unknown"),
                "scene_id": scene_id,
                "scene_file": str(Path(config.scene_data_path) / split / full_scene_id),
                "floorplan_path": str(floorplan_path),
                "initial_pose": initial_pose,
                "gt_answers": gt_answers,
                "choices": choice_data.get("choices", []),
            }
        )

    return episodes


def _infer_scene_floor(
    row: dict[str, Any], init_pose_data: dict[str, dict[str, Any]]
) -> str | None:
    scene_floor = _get_first_available(row, ["scene_floor", "episode_history"], None)
    if scene_floor:
        return str(scene_floor)

    scene_id = _get_first_available(row, ["scene", "scene_id"], None)
    floor = _get_first_available(row, ["floor", "floor_id", "level"], None)
    if scene_id is not None and floor is not None:
        return f"{scene_id}_{floor}"

    if scene_id is None:
        return None

    matching_scene_floors = [
        key for key in init_pose_data if key.startswith(f"{scene_id}_")
    ]
    if len(matching_scene_floors) == 1:
        return matching_scene_floors[0]
    return None


def _load_exploreeqa_data(
    config: DataConfig, filter_floorplan: bool
) -> list[dict[str, Any]]:
    with open(config.exploreeqa_question_data_path) as file:
        questions_data = [
            {key: value for key, value in row.items()}
            for row in csv.DictReader(file, skipinitialspace=True)
        ]

    init_pose_data = {}
    with open(config.exploreeqa_init_pose_data_path) as file:
        for row in csv.DictReader(file, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }

    semantic_scenes = _load_semantic_scenes(config)
    episodes = []
    for idx, row in enumerate(questions_data):
        scene_id = str(_get_first_available(row, ["scene", "scene_id"], required=True))
        scene_floor = _infer_scene_floor(row, init_pose_data)
        if scene_floor is None or scene_floor not in init_pose_data:
            continue

        split = _infer_scene_split(config.scene_data_path, scene_id)
        if split is None:
            continue

        has_semantics = f"{split}/{scene_id}" in semantic_scenes
        if config.use_semantic_data and not has_semantics:
            continue
        if not config.use_semantic_data and has_semantics:
            continue

        _, floor = _split_scene_floor(scene_floor)
        scene_dir = Path(config.scene_data_path) / split / scene_id
        floorplan_path = (
            scene_dir / f"explore_eqa_regions_{floor}" / "topological_graph.json"
        )
        if filter_floorplan and not floorplan_path.exists():
            continue

        init_pose = init_pose_data[scene_floor]
        answers = []
        primary_answer = _get_first_available(
            row,
            ["answer", "gt_answer", "expected_answer", "label"],
            None,
        )
        if primary_answer not in [None, ""]:
            answers.append(eval(row["choices"])[choice_to_number[primary_answer]])
        answers.extend(
            _parse_extra_answers(
                _get_first_available(row, ["extra_answers", "alternate_answers"], [])
            )
        )

        question_id = str(
            _get_first_available(
                row, ["question_id", "id", "uid"], f"exploreeqa_{idx:06d}"
            )
        )
        question = str(
            _get_first_available(
                row, ["question", "question_text", "prompt"], required=True
            )
        )
        episodes.append(
            {
                "question_id": question_id,
                "question": question,
                "category": str(
                    _get_first_available(
                        row,
                        ["label"],
                        "unknown",
                    )
                ),
                "scene_id": scene_id,
                "scene_file": _resolve_scene_file(scene_dir, scene_id),
                "floorplan_path": str(floorplan_path),
                "initial_pose": init_pose["init_pts"]
                + _quat_xyzw_from_yaw(init_pose["init_angle"]),
                "gt_answers": answers,
                "choices": eval(row["choices"]) if "choices" in row else [],
            }
        )

    return episodes


def load_eqa_data(config: DataConfig, filter_floorplan: bool) -> list[dict[str, Any]]:
    """
    Load the configured EQA dataset and normalize it into a common episode format.
    :param config: DataConfig object containing dataset paths and filtering options.
    :param filter_floorplan: Whether to keep only episodes with prior floorplans.
    :return: A list of normalized episode dictionaries.
    """
    config, filter_floorplan = _get_effective_load_settings(config, filter_floorplan)
    dataset = _normalize_dataset_name(config.dataset)
    if dataset == "openeqa":
        episodes = _load_openeqa_data(config, filter_floorplan)
    elif dataset == "exploreeqa":
        episodes = _load_exploreeqa_data(config, filter_floorplan)
    else:
        raise ValueError(
            f"Unsupported dataset '{config.dataset}'. "
            f"Expected 'openeqa' or 'exploreeqa'."
        )

    if config.load_test_split_only:
        episodes = _filter_test_split_episodes(config, episodes)

    return episodes
