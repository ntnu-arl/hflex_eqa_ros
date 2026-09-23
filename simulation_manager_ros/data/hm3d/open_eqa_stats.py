import json
from pathlib import Path

import numpy as np


def check_if_multifloor(traj: np.ndarray) -> bool:
    """
    Check if the trajectory is multi-floor based on the variance of the y-coordinates.
    :param traj: A numpy array of shape (N, 3) representing the trajectory positions.
    :return: True if the trajectory is multi-floor, False otherwise.
    """
    mean = sum(traj) / traj.shape[0]
    variance = sum((traj - mean) ** 2) / traj.shape[0]
    return np.sqrt(variance) > 0.5


BASE_PATH = Path(
    "/media/albert/ExtremeAlbert3/habitat-sim-data/versioned_data/hm3d-0.2/hm3d"
)

with (BASE_PATH / "open-eqa-v0.json").open() as f:
    questions_data = json.load(f)

with (BASE_PATH / "openeqa_init_trajs.json").open() as f:
    trajs_data = json.load(f)

total_questions = 0
all_scenes = set()
non_multifloor_questions = 0
non_multifloor_scenes = set()
scene_num_questions = {}
semantic_scenes_num_questions = {}
num_semantic_questions = 0

for question_data in questions_data:
    if "scannet" in question_data["episode_history"]:
        continue
    total_questions += 1
    init_traj = trajs_data[question_data["episode_history"]]
    all_scenes.add(init_traj["scene_id"])
    if init_traj["scene_id"] not in scene_num_questions:
        scene_num_questions[init_traj["scene_id"]] = 0
    scene_num_questions[init_traj["scene_id"]] += 1
    if check_if_multifloor(np.array(init_traj["full_traj_pos"])[:, 1]):
        continue
    non_multifloor_questions += 1
    non_multifloor_scenes.add(init_traj["scene_id"])

    scene_name = init_traj["scene_id"].split("/")[0]
    txt_files = list((BASE_PATH / "val" / scene_name).glob("*.txt"))
    if not txt_files:
        continue

    if init_traj["scene_id"] not in semantic_scenes_num_questions:
        semantic_scenes_num_questions[init_traj["scene_id"]] = 0
    semantic_scenes_num_questions[init_traj["scene_id"]] += 1
    num_semantic_questions += 1

# Output statistics
print(f"Total questions: {total_questions}")
print(f"Total unique scenes: {len(all_scenes)}")
print(f"Non-multifloor questions: {non_multifloor_questions}")
print(f"Non-multifloor unique scenes: {len(non_multifloor_scenes)}")
# print(f"Scene-wise question counts: {scene_num_questions}")
# print(f"Semantic scene-wise question counts: {semantic_scenes_num_questions}")
print(f"Total semantic questions: {num_semantic_questions}")
print(f"Semantic unique scenes: {len(semantic_scenes_num_questions)}")
