import csv
import os
from pathlib import Path


def load_eqa_data(
    question_data_path: Path,
    init_pose_data_path: Path,
    semantic_annot_data_path: Path,
    use_semantic_data: bool,
):
    # Load dataset
    with open(question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    # Filter to include only scenes with semantic annotations
    semantic_scenes = [
        f
        for f in os.listdir(semantic_annot_data_path / "val")
        if os.path.isdir(os.path.join(semantic_annot_data_path / "val", f))
    ]
    semantic_scenes.extend(
        [
            f
            for f in os.listdir(semantic_annot_data_path / "train")
            if os.path.isdir(os.path.join(semantic_annot_data_path / "train", f))
        ]
    )

    filtered_question_data = []
    if use_semantic_data:
        for data in questions_data:
            if data["scene"] in semantic_scenes:
                filtered_question_data.append(data)
    else:
        for data in questions_data:
            if data["scene"] not in semantic_scenes:
                filtered_question_data.append(data)

    with open(init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    print(f"Loaded {len(filtered_question_data)} questions.")
    return filtered_question_data, init_pose_data


def main():
    BASE_PATH = Path(
        "/media/albert/ExtremeAlbert3/habitat-sim-data/versioned_data/hm3d-0.2/hm3d"
    )
    question_data_path = BASE_PATH / "explore_eqa_questions.csv"
    init_pose_data_path = BASE_PATH / "explore_eqa_scene_init_poses.csv"
    semantic_annot_data_path = BASE_PATH

    questions_data, init_pose_data = load_eqa_data(
        question_data_path,
        init_pose_data_path,
        semantic_annot_data_path,
        use_semantic_data=True,
    )
    total_questions = 0
    all_scenes = set()
    scene_num_questions = {}
    semantic_scenes_num_questions = {}
    num_semantic_questions = 0

    for question_data in questions_data:
        total_questions += 1
        all_scenes.add(question_data["scene"])
        if question_data["scene"] not in scene_num_questions:
            scene_num_questions[question_data["scene"]] = 0
        scene_num_questions[question_data["scene"]] += 1
        scene_path = (
            BASE_PATH / "val" / question_data["scene"]
            if (BASE_PATH / "val" / question_data["scene"]).exists()
            else BASE_PATH / "train" / question_data["scene"]
        )
        txt_files = list(scene_path.glob("*.txt"))
        if not txt_files:
            continue
        if question_data["scene"] not in semantic_scenes_num_questions:
            semantic_scenes_num_questions[question_data["scene"]] = 0
        semantic_scenes_num_questions[question_data["scene"]] += 1
        num_semantic_questions += 1

    # Output statistics
    print(f"Total questions: {total_questions}")
    print(f"Total unique scenes: {len(all_scenes)}")
    print(f"Total semantic questions: {num_semantic_questions}")
    print(f"Total unique semantic scenes: {len(semantic_scenes_num_questions)}")


if __name__ == "__main__":
    main()
