#!/usr/bin/env python3
# BSD 3-Clause License
#
# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
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
"""Create deterministic test splits for OpenEQA and ExploreEQA."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from simulation_manager_ros.data import DataConfig, load_eqa_data

DEFAULT_TEST_PERCENTAGE = 20.0
DEFAULT_SEED = 7
DATASETS = ("openeqa", "exploreeqa")


def _build_dataset_config(base_config: DataConfig, dataset: str) -> DataConfig:
    config_dict = asdict(base_config)
    config_dict["dataset"] = dataset
    return DataConfig(**config_dict)


def _sample_episode_ids(
    episodes: list[dict[str, Any]], test_percentage: float, seed: int
) -> list[str]:
    if not episodes:
        return []

    num_test = max(1, int(round(len(episodes) * test_percentage / 100.0)))
    num_test = min(num_test, len(episodes))

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(episodes), size=num_test, replace=False))
    return [str(episodes[idx]["question_id"]) for idx in indices]


def _load_base_data_config(config_path: Path) -> DataConfig:
    with config_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    config_data = raw_config.get("data", raw_config)
    return DataConfig(**config_data)


def build_split_manifest(
    base_config: DataConfig,
    filter_floorplan: bool,
    use_multifloor_questions: bool,
    use_semantic_data: bool,
    test_percentage: float,
    seed: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "filter_floorplan": filter_floorplan,
        "use_multifloor_questions": use_multifloor_questions,
        "use_semantic_data": use_semantic_data,
        "test_percentage": test_percentage,
        "seed": seed,
        "datasets": {},
    }

    base_config.use_multifloor_questions = use_multifloor_questions
    base_config.use_semantic_data = use_semantic_data

    for dataset in DATASETS:
        dataset_config = _build_dataset_config(base_config, dataset)
        episodes = load_eqa_data(dataset_config, filter_floorplan)
        question_ids = _sample_episode_ids(episodes, test_percentage, seed)
        manifest["datasets"][dataset] = {
            "num_total_episodes": len(episodes),
            "num_test_episodes": len(question_ids),
            "question_ids": question_ids,
        }

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic test splits for OpenEQA and ExploreEQA."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config that can be parsed as DataConfig.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eqa_test_splits.json"),
        help="Path to the output JSON split manifest.",
    )
    parser.add_argument(
        "--test-percentage",
        type=float,
        default=DEFAULT_TEST_PERCENTAGE,
        help=(
            "Percentage of each filtered dataset to reserve for test. "
            f"Default: {DEFAULT_TEST_PERCENTAGE}."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for deterministic sampling. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--filter-floorplan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only episodes with an existing floorplan prior.",
    )
    parser.add_argument(
        "--use-multifloor-questions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match the DataConfig multifloor filtering behavior.",
    )
    parser.add_argument(
        "--use-semantic-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the DataConfig semantic-scene filtering behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_percentage <= 100.0:
        raise ValueError("--test-percentage must be in the range (0, 100].")

    base_config = _load_base_data_config(args.config)
    manifest = build_split_manifest(
        base_config=base_config,
        filter_floorplan=args.filter_floorplan,
        use_multifloor_questions=args.use_multifloor_questions,
        use_semantic_data=args.use_semantic_data,
        test_percentage=args.test_percentage,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")

    for dataset in DATASETS:
        dataset_info = manifest["datasets"][dataset]
        print(
            f"{dataset}: selected {dataset_info['num_test_episodes']} / "
            f"{dataset_info['num_total_episodes']} episodes"
        )
    print(f"Saved split manifest to {args.output}")


if __name__ == "__main__":
    main()
