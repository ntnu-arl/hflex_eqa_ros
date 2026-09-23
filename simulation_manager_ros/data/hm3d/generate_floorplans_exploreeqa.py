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

"""Script to generate floorplans from HM3D dataset."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import trimesh
from extract_topological_graph import GlobalAdjacencyFloorplanGraphBuilder
from openai_client import OpenAIClient, OpenAIClientConfig
from PIL import ImageColor

SPLITS = ["val", "train"]
SYSTEM_PROMPT = (
    "You are a helpful assistant for labeling rooms in a 3D scene. "
    "You will be given a list of object categories present in a room, and you need to "
    "propose a label for the room based on the objects and using common sense. "
    "For example, if the room contains a bed, a nightstand, and a dresser, it's likely "
    "a bedroom. If it contains a stove, a refrigerator, and a sink, it's likely "
    "a kitchen. Output a single JSON file with the key 'label' "
    "and the proposed room label as the value. "
    "Do NOT include anything else in the output except the JSON."
)

T_HI = np.eye(4)
T_HI[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
T_CB = np.eye(4)
T_CB[:3, :3] = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

WALL_CATEGORIES = [
    "wall",
    "blinds",
    "window",
    "garage door",
    "window frame",
    "painting",
    "whine shelf",
    "door",
    "closet mirror wall",
    "curtain",
]


class Region:
    def __init__(self, region_id: int, categories: list[str], scene_name: str) -> None:
        self.region_id = region_id
        self.categories = categories
        self.categories_to_semantic_ids = {}
        self.points = None
        self.colors = None
        self.point_semantics = None
        self.scene_name: str = scene_name
        self.label = "unknown room"

    def filter(self) -> None:
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(self.points)
        pcl.colors = o3d.utility.Vector3dVector(self.colors / 255.0)
        pcl, ind = pcl.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.points = np.asarray(pcl.points)
        self.colors = np.asarray(pcl.colors) * 255.0
        self.point_semantics = self.point_semantics[ind]

    def generate_floorplan(
        self,
        output_dir: Path,
        traj_wf: np.ndarray,
        mode: str = "semantic",
        use_mean_height: bool = True,
    ) -> None:
        """
        Generate a wall occupancy floorplan.

        Modes
        -----
        geometry : detect walls using surface normals
        semantic : use semantic labels to keep only wall points
        """

        if self.points is None or len(self.points) == 0:
            return

        # --------------------------------------------------
        # 1. Check trajectory intersects room height
        # --------------------------------------------------
        min_h = np.min(self.points[:, 2])
        max_h = np.max(self.points[:, 2])

        traj_z = traj_wf[:, 2]

        if not np.any((traj_z >= min_h) & (traj_z <= max_h)):
            return

        # --------------------------------------------------
        # 2. Get wall points depending on mode
        # --------------------------------------------------
        if mode == "semantic":
            if not any(
                cat in WALL_CATEGORIES for cat in self.categories_to_semantic_ids
            ):
                return

            mask = np.zeros_like(self.point_semantics, dtype=bool)
            for wall_cat in WALL_CATEGORIES:
                if wall_cat in self.categories_to_semantic_ids:
                    for semantic_id in self.categories_to_semantic_ids[wall_cat]:
                        mask |= self.point_semantics == semantic_id
            wall_points = self.points[mask]

            if len(wall_points) < 50:
                return

        elif mode == "geometry":
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(self.points)

            pcl.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.2,
                    max_nn=30,
                )
            )

            normals = np.asarray(pcl.normals)
            points = np.asarray(pcl.points)

            # vertical surfaces (walls)
            vertical_mask = np.abs(normals[:, 2]) < 0.3
            wall_points = points[vertical_mask]

            if len(wall_points) < 100:
                return

            # remove furniture clusters
            wall_pcl = o3d.geometry.PointCloud()
            wall_pcl.points = o3d.utility.Vector3dVector(wall_points)

            labels = np.array(
                wall_pcl.cluster_dbscan(
                    eps=0.25,
                    min_points=50,
                    print_progress=False,
                )
            )

            if labels.max() < 0:
                return

            counts = np.bincount(labels[labels >= 0])
            keep_clusters = np.where(counts > 200)[0]

            cluster_mask = np.isin(labels, keep_clusters)
            wall_points = wall_points[cluster_mask]

            if len(wall_points) == 0:
                return

        else:
            raise ValueError(f"Unknown floorplan mode: {mode}")

        pcl_wall = o3d.geometry.PointCloud()
        pcl_wall.points = o3d.utility.Vector3dVector(wall_points)
        o3d.io.write_point_cloud(
            str(output_dir / f"{self.region_id}" / "wall_points.ply"), pcl_wall
        )

        # --------------------------------------------------
        # 3. BEV projection
        # --------------------------------------------------
        if use_mean_height:
            mean_h = np.mean(wall_points[:, 2])
            mask = np.logical_and(
                wall_points[:, 2] >= mean_h - 0.25, wall_points[:, 2] <= mean_h + 0.25
            )
            xy = wall_points[mask][:, :2]
        else:
            xy = wall_points[:, :2]

        resolution = 0.05  # meters / pixel

        min_xy = xy.min(axis=0)
        max_xy = xy.max(axis=0)

        width = int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1
        height = int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1

        grid = np.zeros((height, width), dtype=np.uint8)

        px = ((xy[:, 0] - min_xy[0]) / resolution).astype(int)
        py = ((xy[:, 1] - min_xy[1]) / resolution).astype(int)

        grid[py, px] = 1

        # --------------------------------------------------
        # 4. Morphological cleanup
        # --------------------------------------------------
        kernel = np.ones((3, 3), np.uint8)

        grid = cv2.dilate(grid, kernel, iterations=1)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --------------------------------------------------
        # 5. Save floorplan
        # --------------------------------------------------
        output_path = output_dir / f"{self.region_id}"
        output_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path / "floorplan.png"), grid * 255)

        metadata = {
            "resolution": resolution,
            "origin_xy": min_xy.tolist(),
            "width": width,
            "height": height,
            "mode": mode,
        }

        with (output_path / "floorplan_meta.json").open("w") as f:
            json.dump(metadata, f, indent=4)

    def label_region(
        self,
        region_votes: pd.DataFrame,
        llm_for_unlabeled_rooms: bool = False,
        llm: OpenAIClient = None,
    ) -> None:
        """Label the region with the most voted category."""
        for ind in region_votes.index:
            if (
                region_votes["Scene Name"][ind] == self.scene_name
                and region_votes["Region #"][ind] == self.region_id
            ):
                self.label = region_votes["Weighted Room Proposal"][ind].strip().lower()
                if "unknown" in self.label:
                    break
                else:
                    return
        print(
            f"No votes found for region {self.region_id} in scene {self.scene_name}, "
            f"keeping label as 'unknown room'."
        )
        if llm_for_unlabeled_rooms and llm is not None:
            categories = self.categories
            # Remove duplicates
            categories = list(set(categories))
            prompt = (
                f"The following object categories are present in a room: {categories}."
            )
            response, success = llm.inference(prompt)
            if success and "label" in response:
                self.label = response["label"].strip().lower()
                print(
                    f"LLM proposed label '{self.label}' for region "
                    f"{self.region_id} in scene {self.scene_name}."
                )
            else:
                print(
                    f"LLM failed to propose a label for region {self.region_id} "
                    f"in scene {self.scene_name}, keeping label as 'unknown room'."
                )

    def save(self, output_dir: Path) -> None:
        """Save the region as a point cloud file."""
        if self.points is None or self.colors is None:
            print(
                f"No points or colors to save for region {self.region_id} "
                f"in scene {self.scene_name}, skipping."
            )
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors / 255.0)
        output_path = output_dir / f"{self.region_id}"
        output_path.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path / "room.ply"), pcd)
        # Save BEV projection as a pointcloud
        bev_pcl = o3d.geometry.PointCloud()
        min_height = np.min(self.points[:, 2])
        bev_pcl.points = o3d.utility.Vector3dVector(
            np.hstack(
                (self.points[:, :2], np.full((self.points.shape[0], 1), min_height))
            )
        )
        o3d.io.write_point_cloud(str(output_path / "room_bev.ply"), bev_pcl)
        # Save other info as json
        info = {
            "region_id": self.region_id,
            "categories": self.categories,
            "label": self.label,
            "categories_to_semantic_ids": self.categories_to_semantic_ids,
        }
        with (output_path / "info.json").open("w") as f:
            json.dump(info, f, indent=4)


def rows_in_array(A, B):
    A_view = A.view([("", A.dtype)] * A.shape[1])
    B_view = B.view([("", B.dtype)] * B.shape[1])
    return np.isin(A_view, B_view)


def check_if_multifloor(traj: np.ndarray) -> bool:
    """
    Check if the trajectory is multi-floor based on the variance of the y-coordinates.
    :param traj: A numpy array of shape (N, 3) representing the trajectory positions.
    :return: True if the trajectory is multi-floor, False otherwise.
    """
    mean = sum(traj) / traj.shape[0]
    variance = sum((traj - mean) ** 2) / traj.shape[0]
    return np.sqrt(variance) > 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate floorplans from HM3D dataset."
    )
    parser.add_argument(
        "--hm3d_dir",
        type=str,
        required=True,
        help="Path to the HM3D dataset directory.",
    )
    parser.add_argument(
        "--region_votes_path",
        type=str,
        required=True,
        help="Path to the region votes file.",
    )
    parser.add_argument(
        "--init_poses_path",
        type=str,
        required=True,
        help="Path to the OpenEQA initial poses file.",
    )
    parser.add_argument(
        "--llm_for_unlabeled_rooms",
        action="store_true",
        help="Whether to use an LLM to label rooms that have no votes.",
    )
    parser.add_argument(
        "--extract_topological_graph",
        action="store_true",
        help="Whether to extract the topological graph.",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=2000000,
        help="Number of points to sample from the semantic mesh for each scene.",
    )
    parser.add_argument(
        "--visualize_rooms",
        action="store_true",
        help="Whether to visualize the generated floorplans.",
    )
    parser.add_argument(
        "--show_mesh",
        action="store_true",
        help="Whether to visualize the original semantic mesh.",
    )
    return parser.parse_args()


def process_split(args: argparse.Namespace, split: str) -> None:
    split_dir = args.hm3d_dir / split
    topological_graph_builder = GlobalAdjacencyFloorplanGraphBuilder(
        connect_distance_m=1.0, line_block_fraction=0.25, visualize=False, save=True
    )

    openai_config = OpenAIClientConfig(system_prompt=SYSTEM_PROMPT)
    openai_client = OpenAIClient(openai_config)

    with args.init_poses_path.open() as f:
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

    for scene_floor, data in init_pose_data.items():
        scene_dir = split_dir / scene_floor.split("_")[0]
        floor = scene_floor.split("_")[1]
        if not scene_dir.exists():
            continue

        print(f"Processing {scene_dir}...")
        # Get all files ending with .txt
        txt_files = list(scene_dir.glob("*.txt"))
        if not txt_files:
            print(f"No semantic annotation files found in {scene_dir}, skipping.")
            continue

        traj_wf = T_HI[:3, :3] @ np.array(data["init_pts"])
        traj_wf = traj_wf[None, :]
        with open(txt_files[0]) as f:
            lines = f.readlines()[1:]
        colors = []
        for line in lines:
            color = line.strip().split(",")[1]
            try:
                colors.append(np.array(ImageColor.getcolor("#" + color, "RGB")))
            except Exception as e:
                print(f"Error parsing color {color} in {scene_dir}: {e}")
                continue
        semantic_colors = np.stack(colors, axis=0)
        # Get semantic mesh ending with semantic.glb
        semantic_mesh_files = list(scene_dir.glob("*semantic.glb"))
        if not semantic_mesh_files:
            print(f"No semantic mesh files found in {scene_dir}, skipping.")
            continue
        semantic_mesh_file = semantic_mesh_files[0]
        # Load the semantic mesh
        scene = trimesh.load(semantic_mesh_file)
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
        for chunk in scene.geometry.values():
            chunk_colors = chunk.visual.to_color().vertex_colors[:, :3]
            points_chunk, face_idx = trimesh.sample.sample_surface(
                chunk, args.n_points // len(scene.geometry)
            )
            us_colors = chunk_colors[chunk.faces[face_idx]][:, 0]
            mask = np.squeeze(rows_in_array(us_colors, semantic_colors))
            points = np.vstack((points, points_chunk[mask]))
            colors = np.vstack((colors, us_colors[mask]))

        # tm = trimesh.util.concatenate(tuple(scene.geometry.values()))
        # tm.visual = tm.visual.to_color()
        # points, face_idx = trimesh.sample.sample_surface(tm, args.n_points)

        # # get triangle vertex colors
        # face_vertices = tm.faces[face_idx]
        # vertex_colors = tm.visual.vertex_colors[:, :3]

        # # choose dominant vertex color per triangle
        # colors = vertex_colors[face_vertices][:,0]

        # build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        if args.show_mesh:
            o3d.visualization.draw_geometries(
                [pcd], window_name=f"Semantic PCL of {scene_dir}"
            )

        # Generate regions from semantic annotations
        regions = {}
        scene_total_points = 0
        for line in lines:
            line = line.strip().split(",")
            room_id = int(line[3])
            object_semantic_label = line[2].replace('"', "")
            object_semantic_id = int(line[0])
            try:
                object_color = np.asarray(ImageColor.getcolor("#" + line[1], "RGB"))
            except Exception as e:
                print(f"Error parsing color {line[1]} in {scene_dir}: {e}")
                continue
            if room_id not in regions:
                regions[room_id] = Region(room_id, [], scene_dir.name)
            regions[room_id].categories.append(object_semantic_label)
            if object_semantic_label not in regions[room_id].categories_to_semantic_ids:
                regions[room_id].categories_to_semantic_ids[object_semantic_label] = [
                    object_semantic_id
                ]
            else:
                regions[room_id].categories_to_semantic_ids[
                    object_semantic_label
                ].append(object_semantic_id)

            # Get all points with the same color as the object color
            mask = np.all(colors == object_color, axis=1)
            if not np.any(mask):
                print(
                    f"No points found for object with semantic ID {object_semantic_id} "
                    f"in room {room_id} of scene {scene_dir}, skipping."
                )
                continue
            scene_total_points += np.sum(mask)
            if regions[room_id].points is None:
                regions[room_id].points = points[mask]
                regions[room_id].colors = colors[mask]
                regions[room_id].point_semantics = np.full(
                    points[mask].shape[0], object_semantic_id, dtype=np.int32
                )
            else:
                regions[room_id].points = np.vstack(
                    (regions[room_id].points, points[mask])
                )
                regions[room_id].colors = np.vstack(
                    (regions[room_id].colors, colors[mask])
                )
                regions[room_id].point_semantics = np.hstack(
                    (
                        regions[room_id].point_semantics,
                        np.full(
                            points[mask].shape[0], object_semantic_id, dtype=np.int32
                        ),
                    )
                )

        total_pcl = o3d.geometry.PointCloud()
        region_votes = pd.read_csv(
            args.region_votes_path,
            header=0,
            usecols=["Scene Name", "Region #", "Weighted Room Proposal"],
            sep=",",
        )
        for region in regions.values():
            region.label_region(
                region_votes, args.llm_for_unlabeled_rooms, openai_client
            )
            print(
                f"Region {region.region_id} labeled as {region.label} "
                f"with categories {region.categories}."
            )
            region.filter()
            region.save(scene_dir / f"explore_eqa_regions_{floor}")
            region.generate_floorplan(
                scene_dir / f"explore_eqa_regions_{floor}", traj_wf
            )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(region.points)
            pcd.colors = o3d.utility.Vector3dVector(region.colors / 255.0)
            if args.visualize_rooms:
                o3d.visualization.draw_geometries(
                    [pcd],
                    window_name=f"Region {region.region_id} "
                    f"with categories {region.categories}",
                )
            total_pcl += pcd
        o3d.io.write_point_cloud(
            str(scene_dir / f"explore_eqa_regions_{floor}" / "all_regions.ply"),
            total_pcl,
        )
        traj_pcl = o3d.geometry.PointCloud()
        traj_pcl.points = o3d.utility.Vector3dVector(traj_wf)
        o3d.io.write_point_cloud(
            str(scene_dir / f"explore_eqa_regions_{floor}" / "trajectory.ply"), traj_pcl
        )
        if args.visualize_rooms:
            o3d.visualization.draw_geometries(
                [total_pcl], window_name=f"All regions combined for {scene_dir}"
            )

        if args.extract_topological_graph:
            topological_graph_builder.process_rooms(
                scene_dir / f"explore_eqa_regions_{floor}"
            )


def main(args: argparse.Namespace) -> None:
    for split in SPLITS:
        process_split(args, split)


if __name__ == "__main__":
    args = parse_args()
    args.hm3d_dir = Path(args.hm3d_dir)
    args.region_votes_path = Path(args.region_votes_path)
    args.init_poses_path = Path(args.init_poses_path)
    if not args.region_votes_path.exists():
        raise FileNotFoundError(
            f"Region votes file not found at {args.region_votes_path}"
        )
    if not args.hm3d_dir.exists():
        raise FileNotFoundError(f"HM3D dataset directory not found at {args.hm3d_dir}")
    if not args.init_poses_path.exists():
        raise FileNotFoundError(
            f"Initial poses file not found at {args.init_poses_path}"
        )
    main(args)
