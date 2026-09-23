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
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree


class GlobalAdjacencyFloorplanGraphBuilder:
    """
    Better suited for per-room occupancy maps with metadata.

    Strategy:
    1. Load each room occupancy map
    2. Extract free-space + wall masks
    3. Stitch all rooms into one global canvas using origin_xy and resolution
    4. For each pair of rooms, detect whether their boundary free-space is close
       and whether the straight segment between them is not blocked by walls
    5. Build a room connectivity graph

    Input / output kept the same:
        graph, rooms = builder.process_rooms(room_inputs)
    """

    def __init__(
        self,
        free_thresh=50,
        wall_thresh=200,
        connect_distance_m=0.8,
        boundary_sample_step=2,
        max_candidate_pairs=20,
        line_block_fraction=0.2,
        visualize=False,
        save=False,
    ):
        self.free_thresh = free_thresh
        self.wall_thresh = wall_thresh
        self.connect_distance_m = connect_distance_m
        self.boundary_sample_step = boundary_sample_step
        self.max_candidate_pairs = max_candidate_pairs
        self.line_block_fraction = line_block_fraction
        self.visualize_flag = visualize
        self.save_flag = save

    # ----------------------------------------------------
    # Load data
    # ----------------------------------------------------

    def load_map(self, image_path, metadata_path, info_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        with open(metadata_path) as f:
            meta = json.load(f)

        with open(info_path) as f:
            info = json.load(f)

        return img, meta, info

    # ----------------------------------------------------
    # Grid ↔ world
    # ----------------------------------------------------

    def grid_to_world(self, px, py, meta):
        x = meta["origin_xy"][0] + px * meta["resolution"]
        y = meta["origin_xy"][1] + py * meta["resolution"]
        return np.array([x, y], dtype=np.float32)

    def world_to_global_grid(self, x, y, global_origin_xy, resolution):
        gx = int(round((x - global_origin_xy[0]) / resolution))
        gy = int(round((y - global_origin_xy[1]) / resolution))
        return gx, gy

    # ----------------------------------------------------
    # Masks
    # ----------------------------------------------------

    def get_free_space(self, img):
        return (img < self.free_thresh).astype(np.uint8)

    def get_wall_mask(self, img):
        return (img > self.wall_thresh).astype(np.uint8)

    def clean_free_space(self, free):
        # keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            free, connectivity=8
        )
        if num_labels <= 1:
            return free

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = (labels == largest).astype(np.uint8)

        # mild closing to reduce tiny gaps/noise
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

        return clean

    def compute_room_center(self, free_mask, meta):
        cy, cx = center_of_mass(free_mask)
        return self.grid_to_world(cx, cy, meta)

    def extract_boundary(self, free_mask):
        # 1-pixel free-space boundary
        eroded = cv2.erode(free_mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary = (free_mask > 0) & (eroded == 0)
        return boundary.astype(np.uint8)

    # ----------------------------------------------------
    # Build global stitched canvas
    # ----------------------------------------------------

    def prepare_rooms(self, room_data: Path):
        rooms = []
        resolutions = []

        min_x = np.inf
        min_y = np.inf
        max_x = -np.inf
        max_y = -np.inf

        room_counters = {}

        for room_dir in room_data.iterdir():
            if not room_dir.is_dir():
                continue
            if not (room_dir / "floorplan.png").exists():
                continue
            img, meta, info = self.load_map(
                room_dir / "floorplan.png",
                room_dir / "floorplan_meta.json",
                room_dir / "info.json",
            )
            free = self.clean_free_space(self.get_free_space(img))
            wall = self.get_wall_mask(img)
            boundary = self.extract_boundary(free)

            resolutions.append(meta["resolution"])

            h, w = free.shape
            origin_x, origin_y = meta["origin_xy"]
            res = meta["resolution"]

            room_max_x = origin_x + w * res
            room_max_y = origin_y + h * res

            min_x = min(min_x, origin_x)
            min_y = min(min_y, origin_y)
            max_x = max(max_x, room_max_x)
            max_y = max(max_y, room_max_y)

            center = self.compute_room_center(free, meta)

            rooms.append(
                {
                    "id": info["label"]
                    + " "
                    + str(room_counters.get(info["label"], 0)),
                    "image_path": room_dir / "floorplan.png",
                    "meta_path": room_dir / "floorplan_meta.json",
                    "img": img,
                    "meta": meta,
                    "free_local": free,
                    "wall_local": wall,
                    "boundary_local": boundary,
                    "center": center.tolist(),
                    "doors": [],
                }
            )
            room_counters[info["label"]] = room_counters.get(info["label"], 0) + 1

        # assume same resolution; if not, use smallest
        global_res = min(resolutions)
        global_origin_xy = np.array([min_x, min_y], dtype=np.float32)

        global_w = int(np.ceil((max_x - min_x) / global_res)) + 5
        global_h = int(np.ceil((max_y - min_y) / global_res)) + 5

        global_wall = np.zeros((global_h, global_w), dtype=np.uint8)

        for room in rooms:
            room["free_global"] = np.zeros((global_h, global_w), dtype=np.uint8)
            room["boundary_global"] = np.zeros((global_h, global_w), dtype=np.uint8)

            free = room["free_local"]
            wall = room["wall_local"]
            boundary = room["boundary_local"]
            meta = room["meta"]

            h, w = free.shape
            ox, oy = meta["origin_xy"]

            gx0, gy0 = self.world_to_global_grid(ox, oy, global_origin_xy, global_res)

            # stamp masks into global canvas
            room["free_global"][gy0 : gy0 + h, gx0 : gx0 + w] = np.maximum(
                room["free_global"][gy0 : gy0 + h, gx0 : gx0 + w], free
            )
            room["boundary_global"][gy0 : gy0 + h, gx0 : gx0 + w] = np.maximum(
                room["boundary_global"][gy0 : gy0 + h, gx0 : gx0 + w], boundary
            )
            global_wall[gy0 : gy0 + h, gx0 : gx0 + w] = np.maximum(
                global_wall[gy0 : gy0 + h, gx0 : gx0 + w], wall
            )

        return rooms, global_wall, global_origin_xy, global_res

    # ----------------------------------------------------
    # Geometry helpers
    # ----------------------------------------------------

    def sample_line_pixels(self, p0, p1):
        """
        p0, p1 are (x, y) in global grid coordinates
        """
        x0, y0 = p0
        x1, y1 = p1
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n).round().astype(int)
        ys = np.linspace(y0, y1, n).round().astype(int)
        return xs, ys

    def boundary_points_from_mask(self, boundary_mask, step=1):
        ys, xs = np.where(boundary_mask > 0)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.int32)
        pts = np.stack([xs, ys], axis=1)
        if step > 1 and len(pts) > step:
            pts = pts[::step]
        return pts

    def local_open_space_score(self, free_mask, x, y, radius=3):
        h, w = free_mask.shape
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        patch = free_mask[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return patch.mean()

    def is_connection_open(self, p0, p1, global_wall):
        xs, ys = self.sample_line_pixels(p0, p1)
        h, w = global_wall.shape
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[valid]
        ys = ys[valid]
        if len(xs) == 0:
            return False

        wall_values = global_wall[ys, xs]
        blocked_fraction = wall_values.mean()

        return blocked_fraction <= self.line_block_fraction

    # ----------------------------------------------------
    # Pairwise room connection detection
    # ----------------------------------------------------

    def detect_room_connections(self, rooms, global_wall, global_origin_xy, global_res):
        connect_px = self.connect_distance_m / global_res

        room_boundaries = {}
        room_trees = {}

        for room in rooms:
            pts = self.boundary_points_from_mask(
                room["boundary_global"], step=self.boundary_sample_step
            )
            room_boundaries[room["id"]] = pts
            room_trees[room["id"]] = cKDTree(pts) if len(pts) > 0 else None

        edges = []

        n = len(rooms)
        for i in range(n):
            for j in range(i + 1, n):
                ri = rooms[i]
                rj = rooms[j]

                pts_i = room_boundaries[ri["id"]]
                pts_j = room_boundaries[rj["id"]]

                if len(pts_i) == 0 or len(pts_j) == 0:
                    continue

                tree_j = room_trees[rj["id"]]
                dists, nn_idx = tree_j.query(
                    pts_i, k=1, distance_upper_bound=connect_px
                )

                valid = np.isfinite(dists) & (dists < connect_px)
                if not np.any(valid):
                    continue

                candidate_indices = np.where(valid)[0]
                order = np.argsort(dists[candidate_indices])
                candidate_indices = candidate_indices[order[: self.max_candidate_pairs]]

                best = None
                best_dist = np.inf

                for idx_i in candidate_indices:
                    p_i = pts_i[idx_i]
                    p_j = pts_j[nn_idx[idx_i]]

                    # The straight segment between closest
                    # boundary points should not cross walls
                    if not self.is_connection_open(tuple(p_i), tuple(p_j), global_wall):
                        continue

                    dist = np.linalg.norm(p_i - p_j)
                    if dist < best_dist:
                        best_dist = dist
                        best = (p_i, p_j)

                if best is None:
                    continue

                p_i, p_j = best
                mid = ((p_i + p_j) / 2.0).astype(np.float32)

                door_world = np.array(
                    [
                        global_origin_xy[0] + mid[0] * global_res,
                        global_origin_xy[1] + mid[1] * global_res,
                    ],
                    dtype=np.float32,
                )

                edges.append(
                    {
                        "room_a": ri["id"],
                        "room_b": rj["id"],
                        "door_world": door_world.tolist(),
                        "distance_m": float(best_dist * global_res),
                    }
                )

        return edges

    # ----------------------------------------------------
    # Visualization
    # ----------------------------------------------------

    def _world_to_visual_grid(self, point):
        x = (point[0] - self._vis_global_origin[0]) / self._vis_global_res
        y = (point[1] - self._vis_global_origin[1]) / self._vis_global_res
        return x, y

    def _room_colors(self, n_rooms):
        if n_rooms == 0:
            return []

        hue_order = (np.arange(n_rooms) * 0.618033988749895) % 1.0
        return plt.cm.hsv(hue_order)

    def _draw_topological_graph(self, ax, rooms, graph=None, draw_labels=True):
        room_by_id = {r["id"]: r for r in rooms}

        for room in rooms:
            gx, gy = self._world_to_visual_grid(room["center"])
            ax.plot(gx, gy, "rx", markersize=8)
            if draw_labels:
                ax.text(gx + 2, gy + 2, room["id"], color="red", fontsize=9)

            for d in room["doors"]:
                dx, dy = self._world_to_visual_grid(d)
                ax.plot(dx, dy, "bo", markersize=5)

        if graph is None:
            return

        for u, v, data in graph.edges(data=True):
            cu_x, cu_y = self._world_to_visual_grid(room_by_id[u]["center"])
            cv_x, cv_y = self._world_to_visual_grid(room_by_id[v]["center"])

            ax.plot([cu_x, cv_x], [cu_y, cv_y], color="lime", linewidth=2)

            if "door" in data and data["door"] is not None:
                dx, dy = self._world_to_visual_grid(data["door"])
                ax.plot(dx, dy, "yo", markersize=7)

    def visualize_stitched(self, rooms, global_wall, graph=None):
        fig, ax = plt.subplots(figsize=(10, 10))

        # background walls
        ys, xs = np.where(global_wall > 0)
        if len(xs) > 0:
            ax.scatter(xs, ys, s=2, c="black", alpha=0.8)

        colors = self._room_colors(len(rooms))

        # draw filled room regions
        for idx, room in enumerate(rooms):
            free = room["free_global"]
            ys, xs = np.where(free > 0)
            # Uncomment if you want filled room regions too
            if len(xs) > 0:
                ax.scatter(xs, ys, s=2, color=colors[idx], alpha=0.2)

        self._draw_topological_graph(ax, rooms, graph)

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(
            "Stitched room maps with centers, detected doors, and topological graph"
        )
        ax.set_xlabel("global x (pixels)")
        ax.set_ylabel("global y (pixels)")
        # Tight layout to minimize whitespace
        plt.tight_layout()
        if self.visualize_flag:
            plt.show()

        return fig

    def visualize_topological_floorplan(self, rooms, global_wall, graph=None):
        fig, ax = plt.subplots(figsize=(10, 10))

        occupancy = np.full(global_wall.shape, 255, dtype=np.uint8)
        occupancy[global_wall > 0] = 0
        ax.imshow(occupancy, cmap="gray", vmin=0, vmax=255, origin="upper")

        colors = self._room_colors(len(rooms))
        for idx, room in enumerate(rooms):
            free = room["free_global"]
            ys, xs = np.where(free > 0)
            if len(xs) > 0:
                ax.scatter(xs, ys, s=2.5, color=colors[idx], alpha=0.2)

        ax.set_aspect("equal")
        # ax.set_title("Topological floorplan map")
        # ax.set_xlabel("global x (pixels)")
        # ax.set_ylabel("global y (pixels)")
        # Remove axis ticks and labels for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        if self.visualize_flag:
            plt.show()

        return fig

    def save_topological_floorplan_map(
        self,
        rooms,
        global_wall,
        graph,
        output_dir,
        stem="topological_floorplan_map",
        formats=("png", "eps", "svg"),
    ):
        fig = self.visualize_topological_floorplan(rooms, global_wall, graph)

        for fmt in formats:
            fig.savefig(output_dir / f"{stem}.{fmt}", dpi=300, bbox_inches="tight")

        plt.close(fig)

    # ----------------------------------------------------
    # Build graph
    # ----------------------------------------------------

    def build_graph(self, rooms, edges):
        G = nx.Graph()

        for room in rooms:
            G.add_node(room["id"], center=room["center"])

        for e in edges:
            G.add_edge(
                e["room_a"],
                e["room_b"],
                door=e["door_world"],
                distance_m=e["distance_m"],
            )

        return G

    # ----------------------------------------------------
    # Full pipeline
    # ----------------------------------------------------

    def process_rooms(self, room_data: Path):
        rooms, global_wall, global_origin_xy, global_res = self.prepare_rooms(room_data)

        self._vis_global_wall = global_wall
        self._vis_global_origin = global_origin_xy
        self._vis_global_res = global_res

        edges = self.detect_room_connections(
            rooms, global_wall, global_origin_xy, global_res
        )

        # fill per-room door lists from edges
        room_by_id = {r["id"]: r for r in rooms}
        for e in edges:
            room_by_id[e["room_a"]]["doors"].append(e["door_world"])
            room_by_id[e["room_b"]]["doors"].append(e["door_world"])

        graph = self.build_graph(rooms, edges)

        fig = self.visualize_stitched(rooms, global_wall, graph)

        save_dir = room_data / "topological_graph_output"
        save_dir.mkdir(exist_ok=True)

        if self.save_flag:
            # Save graph and room details to JSON
            output = {
                "nodes": [n for n in graph.nodes()],
                "edges": [[u, v] for u, v, data in graph.edges(data=True)],
            }
            # The episode loader reads the graph directly from the regions directory.
            with (room_data / "topological_graph.json").open("w") as f:
                json.dump(output, f, indent=2)

            fig.savefig(save_dir / "stitched_map.png")
            self.save_topological_floorplan_map(rooms, global_wall, graph, save_dir)

        return graph, rooms


def main():
    room_path = Path(
        "/media/albert/ExtremeAlbert/habitat-sim-data/versioned_data/hm3d-0.2/hm3d/val/00808-y9hTuugGdiq/regions"
    )

    builder = GlobalAdjacencyFloorplanGraphBuilder(
        connect_distance_m=1.0,
        line_block_fraction=0.25,
        visualize=True,
        save=True,
    )

    graph, rooms = builder.process_rooms(room_path)

    print("Nodes:", list(graph.nodes()))
    print("Edges:", list(graph.edges(data=True)))
    print("Room details:")
    for r in rooms:
        print(f"  {r['id']}: center={r['center']}, doors={r['doors']}")


if __name__ == "__main__":
    main()
