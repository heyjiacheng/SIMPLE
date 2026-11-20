# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Copy a recorded teleop dataset with each episode's table replaced by another one.

The copy keeps the recorded states and actions and only edits the scene config
in ``meta/episodes.jsonl``, so ``replay_decoupled_wbc.py`` re-simulates every
episode in MuJoCo and re-renders it in Isaac Sim with the new table.

Variants:

* ``big``   -- round table of radius 0.65-0.85 m. The clutter objects (target and
  distractors) move with the table and stay on it.
* ``small`` -- round table just big enough for the faucet base (radius
  0.07-0.09 m). The clutter objects are removed.
* ``rect``  -- rectangular table 0.5-1.8 m wide and 0.5-1.5 m deep, the faucet
  base anywhere along its near edge but 10 cm from its ends. The clutter objects
  move with the table if they fit on it, else they are removed.
* ``triangle`` -- isosceles triangle with its base on the near edge and its apex
  away from the robot, base 0.9-1.8 m, height 0.8-1.5 m, the faucet base 8 cm
  from the slanted edges. Clutter as for ``rect``.
* ``small_triangle`` -- triangle just big enough for the faucet base. The
  clutter objects are removed.
* ``mixed`` -- per episode, at random: the recorded box table, a big or a small
  round one.

With ``--rooms`` each episode also gets a random background: an HSSD room from
the given JSON, a random room offset and rotation within that room's configured
limits, re-sampled lighting and a random table material. These only change the
Isaac Sim rendering, not the MuJoCo physics. The JSON maps each room to the
table variants it fits without the table running into its furniture, either as
a list (``{"scene1": ["box", "big", "small"]}``) or with a bound on how far
sideways the table may reach (``{"scene6": {"tables": ["rect", "small"],
"y_limit": 0.55}}``; it applies to ``rect`` and the triangles).

``--plan`` replaces ``--src``/``--variant``: a JSON list of
``{"src": <dataset dir>, "episode_index": i, "variant": v, "room": <uid>}``
(``room`` optional, from ``--rooms``). The output then holds exactly these
episodes, renumbered 0..N-1 in plan order, which can come from several datasets.

Per episode the new table is placed so that

* its top height and thickness are the recorded ones;
* the whole faucet base sits on it, every base corner ``--margin`` inside the edge;
* its edge in front of the faucet stays where the recorded table edge was (same
  distance to the robot). It moves toward the robot only as far as the faucet
  base needs (the recorded faucet often overhangs the edge slightly), plus, for
  round tables, a small per-variant tolerance that buys sideways freedom;
* it is shifted sideways (left/right only) by a random offset;
* (unless ``--no-clearance-check``) kinematically replaying the recorded robot
  motion never brings a robot link closer to the new table than it came to the
  recorded one, or closer than ``--clearance``. The physics replay then does not
  touch the table where the recording did not.

Where the margin and the clearance conflict (faucet over the edge and the robot
walking up to it), the margin is thinned down to a base flush with the edge,
then the clearance down to ``--min-clearance``. Each episode's config records
the result under ``table_edit``.

Example::

    python scripts/make_round_table_dataset.py \\
        --src data/psi-data/simple-teleop/G1WholebodyOpenFaucetTeleop-v0/G1WholebodyOpenFaucetTeleop-v0/level-0 \\
        --out data/round_table/big/G1WholebodyOpenFaucetTeleop-v0/level-0 \\
        --variant big --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
import yaml

from simple.dr.lighting import LightingDR, LightingDRCfg
from simple.dr.types import Box
from simple.utils import NumpyArrayEncoder, resolve_data_path, resolve_res_path

TABLE_TYPES = ("box", "big", "small")  # "mixed" picks among these; "box" keeps the recorded table
VARIANTS = {
    # Round tables. radius: range (m). lateral: max sideways offset of the table center from
    # the faucet (m). shift_tolerance: how much more than the least needed to keep the faucet
    # base on, a sideways offset may move the edge in front of the faucet toward the robot (m).
    "big": dict(shape="round", radius=(0.65, 0.85), lateral=0.25, shift_tolerance=0.01, objects="keep"),
    "small": dict(shape="round", radius=(0.07, 0.09), lateral=0.05, shift_tolerance=0.005, objects="remove"),
    # Rectangles. width (sideways) and depth: ranges (m). side_margin: min distance from the
    # faucet base to either end of the near edge (m).
    "rect": dict(shape="rect", width=(0.5, 1.8), depth=(0.5, 1.5), side_margin=0.1, objects="fit"),
    # Isosceles triangles, base on the near edge. base and height: ranges (m). side_margin: min
    # distance from the faucet base to the slanted edges (m).
    "triangle": dict(shape="triangle", base=(0.9, 1.8), height=(0.8, 1.5), side_margin=0.08, objects="fit"),
    "small_triangle": dict(shape="triangle", base=(0.16, 0.24), height=(0.14, 0.22), side_margin=0.01, objects="remove"),
}
# objects: "keep" -- the clutter must fit on the table; "fit" -- kept if it fits, else removed;
# "remove" -- always removed
ROBOT_MJCF = "robots/g1_sonic/g1_29dof_with_hand.xml"


def quat_to_mat(quat_wxyz) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


def base_footprint(mjcf_path: str, band: float = 0.012) -> np.ndarray:
    """Corners (4x2, object frame) of an articulated object's footprint on its support.

    Bounding box of every vertex within ``band`` of the lowest point of the
    object's static (joint-free) bodies -- for the faucet, its base plate.
    """
    m = mujoco.MjModel.from_xml_path(mjcf_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def is_static(body: int) -> bool:
        while body != 0:
            if m.body_jntnum[body] > 0:
                return False
            body = m.body_parentid[body]
        return True

    verts, static = [], []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mesh]: m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
        verts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        static.append(np.full(len(v), is_static(m.geom_bodyid[g])))
    verts, static = np.concatenate(verts), np.concatenate(static)
    low = verts[verts[:, 2] < verts[static, 2].min() + band]
    (x0, y0), (x1, y1) = low[:, :2].min(0), low[:, :2].max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def object_radius(uid: str, quat_wxyz) -> float:
    """Horizontal reach of a graspnet object around its origin, in its recorded orientation."""
    path = resolve_data_path(f"assets/graspnet/collision_models_mujoco/{int(uid):03d}/{int(uid):03d}_32x8.obj")
    verts = trimesh.load(path, force="mesh").vertices @ quat_to_mat(quat_wxyz).T
    return float(np.linalg.norm(verts[:, :2], axis=1).max())


def edge_distances(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """(points, edges) distances from 2D points to the edge lines of a convex counterclockwise
    polygon, positive on the inner side; edge i runs from vertex i to vertex i+1."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    inward = np.stack([-edges[:, 1], edges[:, 0]], 1) / np.linalg.norm(edges, axis=1, keepdims=True)
    return ((points[:, None, :] - vertices[None]) * inward[None]).sum(-1)


def footprint_distance(local_xy: np.ndarray, table: dict) -> np.ndarray:
    """Signed distance (negative inside) from table-frame xy points to the table's footprint."""
    if table["uid"] == "cylinder":
        return np.hypot(local_xy[:, 0], local_xy[:, 1]) - 0.5 * table["size"][0]
    if table["uid"] == "prism":
        start = np.asarray(table["vertices"], dtype=np.float64)
        edge = np.roll(start, -1, axis=0) - start
        rel = local_xy[:, None, :] - start[None]
        t = np.clip((rel * edge).sum(-1) / (edge * edge).sum(-1), 0.0, 1.0)
        dist = np.linalg.norm(rel - t[..., None] * edge, axis=-1).min(1)
        inside = (edge_distances(local_xy, start) >= 0).all(1)
        return np.where(inside, -dist, dist)
    q = np.abs(local_xy) - 0.5 * np.asarray(table["size"][:2], dtype=np.float64)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)


def table_signed_distance(points: np.ndarray, table: dict) -> np.ndarray:
    """Signed distance (negative inside) from world points to a box, cylinder or prism table."""
    local = (points - np.asarray(table["pose"]["position"])) @ quat_to_mat(table["pose"]["quaternion"])
    q = np.stack([footprint_distance(local[:, :2], table), np.abs(local[:, 2]) - 0.5 * table["size"][2]], 1)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)


def top_signed_distance(xy: np.ndarray, table: dict) -> np.ndarray:
    """Signed distance (negative inside) from world xy points to the outline of the table top
    (tables are only rotated about z)."""
    local = (xy - np.asarray(table["pose"]["position"][:2])) @ quat_to_mat(table["pose"]["quaternion"])[:2, :2]
    return footprint_distance(local, table)


def geom_points(m: mujoco.MjModel, g: int) -> tuple[np.ndarray, float]:
    """Points (geom frame) and inflation radius that represent a collision geom's surface:
    hull vertices for meshes, a dense surface sampling for boxes and cylinders, the axis for
    spheres and capsules (inflated by their radius)."""
    size = m.geom_size[g]
    gtype = m.geom_type[g]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[g]
        verts = m.mesh_vert[m.mesh_vertadr[mesh]: m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
        return trimesh.convex.convex_hull(verts).vertices, 0.0
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.zeros((1, 3)), size[0]
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.stack([np.zeros(9), np.zeros(9), np.linspace(-size[1], size[1], 9)], 1), size[0]
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, 7)] * 3, indexing="ij"), -1).reshape(-1, 3)
        return grid[np.abs(grid).max(1) == 1] * size, 0.0  # surface points only
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = np.stack([np.cos(angle), np.sin(angle)], 1) * size[0]
        rings = [np.column_stack([ring * s, np.full(len(ring), z)])
                 for z in np.linspace(-size[1], size[1], 5) for s in (1.0, 0.5, 0.0)]
        return np.concatenate(rings), 0.0
    raise TypeError(f"unsupported robot geom type {gtype}")


class RobotTableClearance:
    """Robot-to-table distances along a recorded trajectory, by kinematic replay.

    Links are measured through surface points against the table's exact signed
    distance: ``mj_geomDistance`` returns spurious zero/negative distances for
    some G1 hand geoms far from the table.
    """

    def __init__(self, joint_names: list[str], max_dist: float = 0.1, stride: int = 2) -> None:
        self.m = mujoco.MjModel.from_xml_path(resolve_data_path(ROBOT_MJCF))
        self.d = mujoco.MjData(self.m)
        self.robot_geoms = np.array([g for g in range(self.m.ngeom)
                                     if (self.m.geom_contype[g] or self.m.geom_conaffinity[g])
                                     and self.m.geom_bodyid[g] != 0])
        self.points = {g: geom_points(self.m, g) for g in self.robot_geoms}
        self.qpos_adr = np.array([self.m.joint(name).qposadr[0] for name in joint_names])
        self.max_dist = max_dist
        self.stride = stride

    def distances(self, base_pose: np.ndarray, joint_pos: np.ndarray, table: dict) -> np.ndarray:
        """Per-frame distance (clipped at max_dist) from the nearest robot link to ``table``."""
        frames = range(0, len(base_pose), self.stride)
        out = np.full(len(frames), self.max_dist)
        for i, t in enumerate(frames):
            self.d.qpos[:7] = base_pose[t]
            self.d.qpos[self.qpos_adr] = joint_pos[t]
            mujoco.mj_kinematics(self.m, self.d)
            # bounding-sphere prefilter: point distances only for links near the table
            center_dist = table_signed_distance(self.d.geom_xpos[self.robot_geoms], table)
            near = center_dist - self.m.geom_rbound[self.robot_geoms] < self.max_dist
            for g in self.robot_geoms[near]:
                points, inflate = self.points[g]
                world = points @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
                out[i] = min(out[i], table_signed_distance(world, table).min() - inflate)
        return out


def place_table(radius: float, offset: float, corners: np.ndarray, x_edge: float, margin: float):
    """Round-table center (cx, cy) and edge shift for one sideways offset, or None if infeasible.

    ``corners`` are the faucet-base corners in world xy, the robot faces +x and
    ``x_edge`` is the recorded table's near edge.
    """
    if abs(offset) >= radius:
        return None
    cy = corners[:, 1].mean() + offset
    room = (radius - margin) ** 2 - (corners[:, 1] - cy) ** 2
    if np.any(room < 0):
        return None
    cx_max = np.min(corners[:, 0] + np.sqrt(room))  # every corner `margin` inside the near edge
    cx_min = np.max(corners[:, 0] - np.sqrt(room))  # ... and inside the far edge
    cx_keep = x_edge + np.sqrt(radius ** 2 - offset ** 2)  # edge in front of the faucet on the recorded edge
    cx = min(cx_keep, cx_max)
    if cx < cx_min:
        return None
    return cx, cy, cx_keep - cx


def sample_round(cfg: dict, rng: np.random.Generator, corners: np.ndarray, x_edge: float, margin: float,
                 y_limit: float | None, args) -> dict | None:
    """A random round table keeping the faucet base ``margin`` inside, or None."""
    radius = rng.uniform(*cfg["radius"])
    candidates = []
    for offset in np.linspace(-cfg["lateral"], cfg["lateral"], 201):
        placed = place_table(radius, offset, corners, x_edge, margin)
        # how far the round top reaches toward the robot past the recorded edge line
        if placed is not None and x_edge - (placed[0] - radius) <= args.max_protrusion:
            candidates.append((offset, placed))
    if not candidates:
        return None
    # keep the edge in front of the faucet as close to the recorded one as this radius allows
    min_shift = min(p[2] for _, p in candidates)
    candidates = [c for c in candidates if c[1][2] <= min_shift + cfg["shift_tolerance"]]
    offset, (cx, cy, edge_shift) = candidates[rng.integers(len(candidates))]
    return dict(uid="cylinder", center=[cx, cy], size=[2 * radius, 2 * radius], offset=offset,
                edge_shift=edge_shift, protrusion=x_edge - (cx - radius), params=dict(radius=radius))


def sample_rect(cfg: dict, rng: np.random.Generator, corners: np.ndarray, x_edge: float, margin: float,
                y_limit: float | None, args) -> dict | None:
    """A random rectangular table with the faucet base on its near edge, or None.

    Where the faucet sits along the edge is drawn uniformly from one end to the other,
    then a width that keeps the table within ``y_limit``.
    """
    depth = rng.uniform(*cfg["depth"])
    x_near = min(x_edge, corners[:, 0].min() - margin)
    y0, y1 = corners[:, 1].min() - cfg["side_margin"], corners[:, 1].max() + cfg["side_margin"]  # edge must cover
    # the table reaches `t * slack` past y0 and `(1 - t) * slack` past y1
    t = rng.uniform()
    slack_max = cfg["width"][1] - (y1 - y0)
    if y_limit is not None:
        slack_max = min(slack_max, (y0 + y_limit) / max(t, 1e-9), (y_limit - y1) / max(1 - t, 1e-9))
    slack_min = max(cfg["width"][0] - (y1 - y0), 0.0)
    if slack_max < slack_min or x_near + depth < corners[:, 0].max() + margin:
        return None
    slack = rng.uniform(slack_min, slack_max)
    width = slack + (y1 - y0)
    cy = y0 - t * slack + 0.5 * width
    return dict(uid="box", center=[x_near + 0.5 * depth, cy], size=[depth, width], offset=cy - corners[:, 1].mean(),
                edge_shift=x_edge - x_near, protrusion=x_edge - x_near, params=dict(width=width, depth=depth))


def sample_triangle(cfg: dict, rng: np.random.Generator, corners: np.ndarray, x_edge: float, margin: float,
                    y_limit: float | None, args) -> dict | None:
    """A random isosceles triangle table, base on the near edge, faucet base on it, or None."""
    base, height = rng.uniform(*cfg["base"]), rng.uniform(*cfg["height"])
    x_near = min(x_edge, corners[:, 0].min() - margin)
    # counterclockwise around the bounding-box center: the base runs from vertex 2 to vertex 0
    vertices = np.array([[-0.5 * height, -0.5 * base], [0.5 * height, 0.0], [-0.5 * height, 0.5 * base]])
    feasible = []
    for cy in np.linspace(corners[:, 1].min() - 0.5 * base, corners[:, 1].max() + 0.5 * base, 201):
        if y_limit is not None and abs(cy) + 0.5 * base > y_limit:
            continue
        dist = edge_distances(corners - [x_near + 0.5 * height, cy], vertices)
        if dist[:, 2].min() >= margin - 1e-9 and dist[:, :2].min() >= cfg["side_margin"]:
            feasible.append(cy)
    if not feasible:
        return None
    cy = feasible[rng.integers(len(feasible))]
    return dict(uid="prism", center=[x_near + 0.5 * height, cy], size=[height, base], vertices=vertices.tolist(),
                offset=cy - corners[:, 1].mean(), edge_shift=x_edge - x_near, protrusion=x_edge - x_near,
                params=dict(base=base, height=height))


SAMPLERS = {"round": sample_round, "rect": sample_rect, "triangle": sample_triangle}


def place_objects(objects: list[dict], radius: float, cx: float, cy: float, old_center: np.ndarray,
                  rng: np.random.Generator, margin: float, keep_away: float):
    """New xy for the clutter objects: their layout relative to the recorded table center is
    kept and the group slides along x to a random spot that fits on the round table, at least
    ``keep_away`` behind its near edge (the faucet side). None if they cannot fit."""
    lo, hi = -np.inf, np.inf
    for obj in objects:
        rel = np.asarray(obj["position"][:2]) - old_center
        reach = radius - margin - obj["radius"]
        if reach <= abs(rel[1]):
            return None
        half_chord = np.sqrt(reach ** 2 - rel[1] ** 2)
        lo = max(lo, -rel[0] - half_chord, obj["radius"] - rel[0] - radius + keep_away)
        hi = min(hi, -rel[0] + half_chord)
    if lo > hi:
        return None
    shift = np.array([cx + rng.uniform(lo, hi), cy]) - old_center
    return [(np.asarray(obj["position"][:2]) + shift).tolist() for obj in objects]


def fit_objects(objects: list[dict], table: dict, old_center: np.ndarray, rng: np.random.Generator,
                margin: float, keep_away: float):
    """Like :func:`place_objects`, for any table shape: the group keeps its layout, is centered
    on the table's center line and slides along x to a random spot where every object is
    ``margin`` inside the edge and ``keep_away`` behind the near edge. None if it cannot fit."""
    center = np.asarray(table["pose"]["position"][:2], dtype=np.float64)
    x_near = center[0] - 0.5 * table["size"][0]
    rel = np.array([np.asarray(obj["position"][:2]) - old_center for obj in objects])
    radii = np.array([obj["radius"] for obj in objects])
    valid = []
    for shift in np.arange(-0.5 * table["size"][0], 0.5 * table["size"][0], 0.01):
        xy = center + [shift, 0.0] + rel
        if (np.all(top_signed_distance(xy, table) <= -(radii + margin))
                and np.all(xy[:, 0] - radii >= x_near + keep_away)):
            valid.append(shift)
    if not valid:
        return None
    return (center + [valid[rng.integers(len(valid))], 0.0] + rel).tolist()


def edit_table(env_conf: dict, variant: str, rng: np.random.Generator, footprints: dict,
               clearance: RobotTableClearance | None, recorded: tuple | None, y_limit: float | None, args) -> dict:
    """Return a copy of ``env_conf`` with a new table (and moved/removed clutter objects)."""
    cfg = VARIANTS[variant]
    dr = env_conf["dr_state_dict"]
    spatial = dr["spatial"]
    old_table = dr["scene"]["table"]
    if old_table["uid"] != "box" or not np.allclose(old_table["pose"]["quaternion"], [1, 0, 0, 0]):
        raise ValueError(f"expected an axis-aligned box table, got {old_table}")
    old_center = np.asarray(old_table["pose"]["position"][:2], dtype=np.float64)
    x_edge = old_center[0] - 0.5 * old_table["size"][0]

    # faucet-base corners in world xy
    art = dr["articulated"]
    faucet = spatial[f"articulated_{art['uid']}"]
    if art["uid"] not in footprints:
        mjcf = resolve_data_path(f"assets/articulated/{int(art['uid']):03d}/output_mjcf/{int(art['uid']):03d}.xml")
        footprints[art["uid"]] = base_footprint(mjcf)
    rot = quat_to_mat(faucet["quaternion"])[:2, :2]
    corners = footprints[art["uid"]] @ rot.T + np.asarray(faucet["position"][:2])

    object_uids = ([dr["target"]["uid"]] if dr.get("target") else []) + list(dr["distractors"].keys())
    objects = [dict(uid=uid, position=spatial[uid]["position"],
                    radius=object_radius(uid, spatial[uid]["quaternion"])) for uid in object_uids]

    d_recorded = clearance.distances(*recorded, old_table) if clearance is not None else None

    def sample(margin: float, min_clearance: float):
        """One random table keeping the faucet base ``margin`` inside and the recorded robot
        motion ``min_clearance`` away (or as far as from the recorded table), or None."""
        placed = SAMPLERS[cfg["shape"]](cfg, rng, corners, x_edge, margin, y_limit, args)
        if placed is None:
            return None
        table = copy.deepcopy(old_table)
        table["uid"] = placed["uid"]
        table["size"] = [*placed["size"], old_table["size"][2]]
        table["pose"]["position"] = [*placed["center"], old_table["pose"]["position"][2]]
        if "vertices" in placed:
            table["vertices"] = placed["vertices"]

        new_xy = None
        if cfg["objects"] == "keep" and objects:
            radius = placed["params"]["radius"]
            new_xy = place_objects(objects, radius, *placed["center"], old_center, rng, args.object_margin, args.keep_away)
            if new_xy is None:
                return None
        elif cfg["objects"] == "fit" and objects:
            new_xy = fit_objects(objects, table, old_center, rng, args.object_margin, args.keep_away)

        d_new = None
        if clearance is not None:
            d_new = clearance.distances(*recorded, table)
            if np.any(d_new < np.minimum(min_clearance, d_recorded) - 1e-3):
                return None
        return dict(placed=placed, table=table, new_xy=new_xy, d_new=d_new, margin=margin, min_clearance=min_clearance)

    # When the faucet base overhangs the recorded edge and the robot walks close to it, the
    # table cannot keep both the full `--margin` and `--clearance`. The base stays on the table:
    # first thin the margin down to a base flush with the edge, then the clearance.
    ladder = [(margin, args.clearance) for margin in np.linspace(args.margin, 0.0, 5)]
    ladder += [(0.0, c) for c in np.linspace(args.clearance, args.min_clearance, 3)[1:]]
    tries = 0
    for margin, min_clearance in ladder:
        for _ in range(args.max_tries):
            tries += 1
            result = sample(margin, min_clearance)
            if result is not None:
                break
        else:
            continue
        break
    else:
        raise RuntimeError(f"no valid {variant} table after {tries} tries")
    placed, table, new_xy, d_new = result["placed"], result["table"], result["new_xy"], result["d_new"]

    out = copy.deepcopy(env_conf)
    dr, layout = out["dr_state_dict"], out["layout"]
    dr["scene"]["table"] = table
    layout["actors"]["table"] = copy.deepcopy(table)
    layout["scene"]["table"] = copy.deepcopy(table)
    clutter = {name: actor for name, actor in layout["actors"].items()
               if name == "target" or name.startswith("distractor")}
    if new_xy is not None:
        for obj, xy in zip(objects, new_xy):
            position = [*xy, obj["position"][2]]
            dr["spatial"][obj["uid"]]["position"] = position
            for actor in clutter.values():
                if str(actor["uid"]) == obj["uid"]:
                    actor["pose"]["position"] = list(position)
    else:
        dr["target"] = None
        dr["distractors"] = {}
        for uid in object_uids:
            dr["spatial"].pop(uid)
        for name in clutter:
            layout["actors"].pop(name)
        dr["material"]["object_shader_params"] = []

    out["table_edit"] = {
        "variant": variant,
        **placed["params"],
        "lateral_offset": placed["offset"],  # table center y minus faucet-base center y
        "edge_shift": placed["edge_shift"],  # how far the edge in front of the faucet moved toward the robot
        "protrusion": placed["protrusion"],  # closest table point past the recorded edge line
        "faucet_base_inset": float(-top_signed_distance(corners, table).max()),  # min faucet-base corner to edge distance
        "objects": ("moved" if new_xy is not None else "removed") if objects else "none",
        "required_margin": result["margin"],
        "required_clearance": None if clearance is None else result["min_clearance"],
        "min_clearance": None if d_new is None else float(d_new.min()),
        "min_clearance_recorded_table": None if d_recorded is None else float(d_recorded.min()),
        "tries": tries,
    }
    return out


def randomize_background(env_conf: dict, room: dict, rng: np.random.Generator, table_materials: list) -> tuple[dict, dict]:
    """Return a copy of ``env_conf`` set in HSSD ``room`` (its config.yaml entry), with a random
    room offset and rotation within the room's limits, re-sampled lighting and a random table
    material, plus a summary of the choices. MuJoCo has no room, lights or materials, so this
    only changes the rendering."""
    out = copy.deepcopy(env_conf)
    dr, layout = out["dr_state_dict"], out["layout"]

    def between(up, down):
        up, down = np.asarray(up, dtype=np.float64), np.asarray(down, dtype=np.float64)
        return (up + rng.uniform(size=up.shape) * (down - up)).tolist()

    scene = dict(uid=f"hssd:{room['uid']}", conf=room, name=room["name"], data_dir=f"{room['data_dir']}/{room['name']}",
                 center_offset=between(room["center_offset_limit_up"], room["center_offset_limit_down"]),
                 center_orientation=between(room["center_orientation_limit_up"], room["center_orientation_limit_down"]))
    dr["scene"].update(scene)
    layout["scene"].update(scene)

    # same lighting randomizer and ranges as the recording (it samples from np.random)
    cfg = {k: Box(v["low"], v["high"]) if isinstance(v, dict) else v
           for k, v in out["dr_cfgs"]["lighting"].items() if k != "randmizer_class"}
    cfg["light_num"] = tuple(cfg["light_num"])
    lighting = LightingDR(LightingDRCfg(**cfg))
    lighting("train")
    dr["lighting"] = json.loads(json.dumps(lighting.state_dict(), cls=NumpyArrayEncoder))
    layout["lights"] = list(dr["lighting"].values())

    material = table_materials[rng.integers(len(table_materials))]
    dr["material"]["table_material"] = material
    for table in (dr["scene"]["table"], layout["actors"]["table"], layout["scene"]["table"]):
        table["material"] = material
    return out, {"room": room["uid"], "room_offset": scene["center_offset"],
                 "room_orientation": scene["center_orientation"], "table_material": material["name"]}


def load_rooms(path: str) -> dict[str, dict]:
    """Rooms JSON -> {uid: {"conf": config.yaml entry, "tables": [variants it fits], "y_limit": m or None}}."""
    hssd = {room["uid"]: room for room in yaml.safe_load(open(resolve_res_path("hssd-scenes/config.yaml")))}
    rooms = {}
    for uid, spec in json.loads(Path(path).read_text()).items():
        spec = {"tables": spec} if isinstance(spec, list) else spec
        rooms[uid] = {"conf": hssd[uid], "tables": spec["tables"], "y_limit": spec.get("y_limit")}
    return rooms


class Source:
    """Episodes, recorded trajectories and meta files of one LeRobot teleop dataset."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.info = json.loads((root / "meta" / "info.json").read_text())
        self.entries = {e["episode_index"]: e for e in map(json.loads, open(root / "meta" / "episodes.jsonl"))}
        self.parquet = {int(p.stem.split("_")[-1]): p for p in (root / "data").rglob("episode_*.parquet")}
        stats = root / "meta" / "episodes_stats.jsonl"
        self.stats = {s["episode_index"]: s for s in map(json.loads, open(stats))} if stats.exists() else {}

    def recorded_motion(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        table = pq.read_table(self.parquet[idx], columns=["observation.base_pose", "observation.state"])
        return (np.asarray(table["observation.base_pose"].to_pylist(), dtype=np.float64),
                np.asarray(table["observation.state"].to_pylist(), dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=None, help="LeRobot dataset dir with meta/episodes.jsonl and data/")
    parser.add_argument("--out", required=True, help="output dataset dir (must not exist)")
    parser.add_argument("--variant", choices=sorted(VARIANTS) + ["mixed"], default=None)
    parser.add_argument("--plan", default=None,
                        help='JSON list of {"src", "episode_index", "variant", "room"}: exactly these episodes, renumbered')
    parser.add_argument("--rooms", default=None,
                        help='JSON {"<hssd scene uid>": [variants it fits] or {"tables": [...], "y_limit": m}}: '
                             "random room, lighting and table material per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.01,
                        help="distance from every faucet-base corner to the table edge (m); lowered step-wise "
                             "to 0 for episodes where it would push the table into the robot")
    parser.add_argument("--max-protrusion", type=float, default=0.05,
                        help="max distance a round top may reach toward the robot past the recorded edge line (m)")
    parser.add_argument("--object-margin", type=float, default=0.02, help="min object-to-edge distance (m)")
    parser.add_argument("--keep-away", type=float, default=0.3,
                        help="min distance of the clutter objects behind the table's near edge (m)")
    parser.add_argument("--clearance", type=float, default=0.02,
                        help="min robot-to-new-table distance where the robot kept at least this far from the recorded table (m)")
    parser.add_argument("--min-clearance", type=float, default=0.01,
                        help="clearance floor for episodes where --clearance cannot be kept with the faucet base on the table (m)")
    parser.add_argument("--no-clearance-check", action="store_true")
    parser.add_argument("--max-tries", type=int, default=20, help="random tables tried per fallback step")
    parser.add_argument("--limit", type=int, default=None, help="only the first N episodes (for testing)")
    args = parser.parse_args()
    if args.plan is None and (args.src is None or args.variant is None):
        parser.error("give --src and --variant, or --plan")
    if args.plan is not None and (args.src is not None or args.variant is not None):
        parser.error("--plan replaces --src and --variant")

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it first")
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # the lighting randomizer draws from np.random

    rooms, table_materials = None, None
    if args.rooms is not None:
        rooms = load_rooms(args.rooms)
        # opaque materials only: a see-through top would hide the table edge in front of the faucet
        table_materials = [m for m in np.load(resolve_res_path("vMaterials_2/material_split.npy"), allow_pickle=True).item()["train"]["table"]
                           if m["path"].split("/")[-2] not in ("Glass", "Gems", "Liquids")
                           and not re.search(r"clear|translucent|transparent", m["name"], re.IGNORECASE)]

    # jobs: (source, source episode, variant, room uid); output episode k is jobs[k]
    sources: dict[str, Source] = {}
    if args.plan is not None:
        plan = json.loads(Path(args.plan).read_text())
        for item in plan:
            sources.setdefault(item["src"], Source(Path(item["src"])))
            if item.get("room") and (rooms is None or item["variant"] not in rooms[item["room"]]["tables"]):
                raise ValueError(f"plan item {item}: room not in --rooms or does not fit {item['variant']}")
        jobs = [(sources[item["src"]], item["episode_index"], item["variant"], item.get("room")) for item in plan]
    else:
        source = sources.setdefault(args.src, Source(Path(args.src)))
        indices = sorted(source.entries)
        if args.limit is not None:
            indices = indices[: args.limit]
        # the replay walks episode indices 0..N-1, so the kept episodes must stay contiguous
        if indices != list(range(len(indices))):
            raise ValueError(f"episode indices in {args.src} are not 0..{len(indices) - 1}")
        jobs = [(source, idx, None, None) for idx in indices]  # variant and room are drawn per episode below

    def pick_room(variant: str) -> str:
        fitting = [uid for uid, room in rooms.items() if variant in room["tables"]]
        return fitting[rng.integers(len(fitting))]

    first = next(iter(sources.values()))
    clearance = None
    if not args.no_clearance_check:
        clearance = RobotTableClearance(first.info["features"]["observation.state"]["names"])

    footprints: dict = {}
    edited = []
    for k, (source, idx, variant, room) in enumerate(jobs):
        entry = source.entries[idx]
        env_conf = json.loads(entry["environment_config"])
        # without --plan, each episode draws its table type, its table, then its room, in this order
        # (the random-room configs were made with it); rect and triangle tables draw the room first,
        # since its y_limit bounds them
        room_after = False
        if args.plan is None:
            variant = args.variant if args.variant != "mixed" else TABLE_TYPES[rng.integers(len(TABLE_TYPES))]
            if rooms is not None:
                room_after = variant == "box" or VARIANTS[variant]["shape"] == "round"
                room = None if room_after else pick_room(variant)
        if variant == "box":
            env_conf["table_edit"] = {"variant": "box"}
        else:
            recorded = source.recorded_motion(idx) if clearance is not None else None
            y_limit = rooms[room]["y_limit"] if room is not None else None
            env_conf = edit_table(env_conf, variant, rng, footprints, clearance, recorded, y_limit, args)
        if room_after:
            room = pick_room(variant)
        if room is not None:
            env_conf, background = randomize_background(env_conf, rooms[room]["conf"], rng, table_materials)
            env_conf["table_edit"].update(background)
        env_conf["table_edit"].update(seed=args.seed, source_dataset=str(source.root), source_episode_index=idx)
        edited.append((source, idx, {**entry, "episode_index": k, "environment_config": json.dumps(env_conf, cls=NumpyArrayEncoder)}))
        e = env_conf["table_edit"]
        line = f"ep {k:3d} (source {idx:3d}): {variant:14s}"
        if variant != "box":
            shape = " ".join(f"{name}={e[name]:.3f}" for name in ("radius", "width", "depth", "base", "height") if name in e)
            line += (f" {shape} offset={e['lateral_offset']:+.3f} "
                     f"edge_shift={e['edge_shift']:.3f} protrusion={e['protrusion']:+.3f} "
                     f"inset={e['faucet_base_inset']:.3f} (required {e['required_margin']:.4f}) "
                     f"objects={e['objects']} tries={e['tries']}")
            if e["min_clearance"] is not None:
                line += (f" clearance={e['min_clearance']:.3f} (required {e['required_clearance']:.3f}, "
                         f"recorded table {e['min_clearance_recorded_table']:.3f})")
        if room is not None:
            line += f" room={e['room']} yaw={e['room_orientation'][2]:.0f} material={e['table_material']}"
        print(line)

    (out / "meta").mkdir(parents=True)
    info = copy.deepcopy(first.info)
    data_path = info["data_path"]
    total_frames = 0
    with open(out / "meta" / "episodes.jsonl", "w") as f_ep, open(out / "meta" / "episodes_stats.jsonl", "w") as f_stats:
        for source, idx, entry in edited:
            k = entry["episode_index"]
            f_ep.write(json.dumps(entry) + "\n")
            if idx in source.stats:
                f_stats.write(json.dumps({**source.stats[idx], "episode_index": k}) + "\n")
            # renumber the episode; its frames get global indices in output order
            table = pq.read_table(source.parquet[idx])
            n = table.num_rows
            for name, values in (("episode_index", np.full(n, k, dtype=np.int64)),
                                 ("index", np.arange(total_frames, total_frames + n, dtype=np.int64))):
                if name in table.column_names:
                    table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
            dst = out / data_path.format(episode_chunk=k // info["chunks_size"], episode_index=k)
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst)
            total_frames += n
    for name in ("tasks.jsonl", "modality.json"):
        shutil.copy2(first.root / "meta" / name, out / "meta" / name)
    info["total_episodes"] = len(edited)
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{len(edited)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    edits = [json.loads(entry["environment_config"])["table_edit"] for _, _, entry in edited]
    thinned = [i for i, e in enumerate(edits) if e.get("required_margin", args.margin) < args.margin
               or (e.get("required_clearance") or args.clearance) < args.clearance]
    print(f"wrote {len(edited)} episodes to {out}; fallback margin/clearance used in episodes {thinned}")


if __name__ == "__main__":
    main()
