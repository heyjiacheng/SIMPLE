# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Bimanual teleop task: one hand picks up a bottle, the other pulls its cap straight off (no unscrewing).

The bottle is `bottle_cap:0` (see `simple.assets.bottle_cap`): the PartNet bottle `articulated:1`, its cap on a
slide joint along the bottle axis. Friction holds the cap on harder than the bottle weighs: pulling the cap alone
lifts the whole bottle, so one hand must hold the bottle while the other pulls. After 1 cm of pull the cap comes
off and is a free body: it stays in the hand or falls. The bottle stands on the table in front of the robot, with
graspnet clutter further back.

`G1WholebodyUncapSmallBottleTaskTeleop` is the same task with a smaller bottle, `bottle_cap:1`: 5.2 cm wide and
12.2 cm tall instead of 7.4 cm and 17.4 cm. The hands that work on it hide more of it from the head camera.

Success: the cap is off, at least 4 cm from where it sat, and the bottle is held at least `lift_height` (1 cm) up off
the table. A hand (palm or fingers) touches the bottle, and nothing else but the robot does. The order is free: uncap
on the table and then pick the bottle up, or the other way round. The cap may still be in the other hand or may have
been put down. This has to last `success_criteria` / 0.03 control steps (50 Hz) in a row.

The robot starts close to the table, with its arms already in the whole-body controller's start pose (forearms
forward, hands above the table top), so the bottle, which stands between its hands, is within reach without a step
(see `_start_arms_forward`). `G1WholebodyUncapSmallBottleTaskTeleop` keeps the earlier setup: the robot further back,
arms spawned down, the bottle up to 6 cm to the side, and a 2.5 cm lift.

`side_view_cfgs` are two fixed stereo cameras, one on each side of the bottle. They are for the teleoperator only:
`teleop_decoupled_wbc --vr-view side` adds them in its own process to show the headset a side view. They are not
sensors of the task, so recorded datasets, replay and eval keep only the head camera.
"""

from __future__ import annotations

import re
from copy import copy, deepcopy
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Type

import numpy as np
import transforms3d as t3d
from gymnasium import spaces

from simple.assets import AssetManager
from simple.assets.bottle_cap import CapMechanism
from simple.core.actor import Actor, ArticulatedObjectActor, ObjectActor
from simple.core.layout import Layout
from simple.core.randomizer import Randomizer, RandomizerCfg
from simple.core.task import Task
from simple.dr import (
    ArticulatedObjectDrCfg,
    CameraDRCfg,
    DistractorDRCfg,
    LanguageDRCfg,
    LightingDRCfg,
    MaterialDRCfg,
    SpatialDR,
    SpatialDRCfg,
    TabletopSceneDRCfg,
)
from simple.dr.manager import TabletopGraspDRManager
from simple.dr.types import Box
from simple.robots.protocols import Controllable
from simple.robots.registry import RobotRegistry
from simple.sensors import SensorCfg, StereoCameraCfg
from simple.tasks.registry import TaskRegistry

_CAP_OFF = 0.04  # m the cap's centre must be from where it sat on the bottle (a cap put back on top doesn't count)
_REWARD_STEP = 0.03  # reward per control step while the goal state holds; it drops to 0 when the state is lost
_HAND_BODY = re.compile(r"(left|right)_(wrist_yaw_link|hand_\w+)")  # palm and fingers
_SETTLE_DROP = 0.002  # m: the bottle is placed this far above the table top and settles onto it

# graspnet1b objects that are bottles, caps or bottle-like, kept out of the clutter so "the bottle" is unambiguous
_BOTTLE_LIKE = ["3", "34", "35", "36", "37", "38", "40", "42", "45", "61", "62", "63", "64", "65", "66"]

# The teleop side views (world frame, m). Each camera is 0.62 m to one side of the bottle region and 0.3 m in front of
# it, looking down at the bottle's top half, where the hands work. Picked on a replay of a teleop uncap: from there the
# near hand and the cap stay in view, and the robot's body fills the far half of the image.
_SIDE_EYE = (-0.40, 0.62, 1.15)  # the left camera; the right one mirrors y
_SIDE_TARGET = (-0.70, 0.0, 0.86)


def _look_at(eye, target) -> list[float]:
    """The wxyz quaternion of a level camera at `eye` looking at `target`, in SIMPLE's camera frame (x: view
    direction, y: left, z: up), as `eye_on_base` cameras take it."""
    x = np.subtract(target, eye, dtype=float)
    x /= np.linalg.norm(x)
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    return t3d.quaternions.mat2quat(np.stack([x, y, np.cross(x, y)], axis=1)).tolist()


def _side_view_cfg(sign: float) -> StereoCameraCfg:
    eye = (_SIDE_EYE[0], sign * _SIDE_EYE[1], _SIDE_EYE[2])
    return StereoCameraCfg(
        uid="side_view",
        mount="eye_on_base",  # a fixed camera, posed in the world frame
        width=640,  # the head camera's resolution
        height=360,
        focal_length=1.93,
        fov=np.deg2rad(70),
        near=0.05,
        far=5,
        baseline=0.064,  # a human's eye distance, so the headset shows true depth
        pose=dict(position=list(eye), quaternion=_look_at(eye, _SIDE_TARGET)),
    )


def _start_arms_forward(robot) -> None:
    """Spawn the robot's arms in the whole-body controller's start pose: upper arms down, forearms forward, the hands
    about 0.8 m up and 0.36 m ahead of the pelvis.

    The robot's own spawn pose has the arms down, the hands 0.6-0.7 m up, and the controller swings them up to its
    start pose in the first 0.2 s. Close to the table (top at 0.68 m), that swing sweeps the hands through the table
    edge, and the pushed robot walks back, up to a metre (MuJoCo, 2026-11-12). Starting in the controller's pose
    leaves nothing to swing. Only this robot instance changes: its arm controllers get their own copy of the robot
    class's controller config, which every other G1 task keeps.
    """
    model = robot.sonic_robot_model
    upper_body = model.get_joint_group_indices("upper_body")
    start = dict(zip((model.joint_names[i] for i in upper_body), model.get_initial_upper_body_pose()))
    controller = robot.controller
    for arm in (controller.left_arm, controller.right_arm):
        arm.cfg = copy(arm.cfg)
        arm.cfg.init_qpos = [float(start[name]) for name in arm.cfg.joint_names]


class _BottleSpatialDR(SpatialDR):
    """SpatialDR that stands the bottle on the table top.

    SpatialDR's own articulated-object branch takes the height from the region. In random mode it also reads the
    surface of the object placed just before, and fails if there is none (no clutter at dr_level >= 2). This
    class places the bottle itself: an x, y and yaw from the region, and a height that puts its bottom on the
    table top, whatever the table height. The placement goes in the inner state like SpatialDR's, so recorded
    episodes replay with the same layout.
    """

    cfg: _BottleSpatialDRCfg  # type: ignore[assignment]

    def __call__(self, split: str, layout: Layout, table_height: float = 0.0) -> None:
        bottle = layout.actors.pop("articulated", None)  # hidden from SpatialDR
        try:
            super().__call__(split, layout, table_height=table_height)
        finally:
            if bottle is not None:
                layout.actors["articulated"] = bottle  # back in last place, where Task.reset put it
        if bottle is not None:
            assert isinstance(bottle, ArticulatedObjectActor)
            self._place_bottle(bottle, layout)

    def _place_bottle(self, bottle: ArticulatedObjectActor, layout: Layout) -> None:
        key = f"articulated_{bottle.uid}"
        if key not in self._inner_state:
            pick = (lambda box: box.middle()) if self.spatial_mode == "fixed" else (lambda box: box.sample())
            assert self.cfg.articulated_region is not None, "articulated_region: where the bottle stands (x, y)"
            x, y = pick(self.cfg.articulated_region)[:2]
            side_limit = getattr(self.cfg, "bottle_side_limit", None)
            if side_limit is not None and self.spatial_mode != "fixed":
                # how far to the side depends on how far ahead of the robot's spawn point the bottle stands
                ahead = float(x) - float(layout.robot.pose.position[0])
                limit = float(np.interp(ahead, *zip(*side_limit)))
                y = float(np.random.uniform(-limit, limit))
            yaw = pick(self.cfg.articulated_rotate_z) if self.cfg.articulated_rotate_z is not None else 0.0
            table = layout.scene.table
            table_top = table.pose.position[2] + 0.5 * table.size[2]
            z = table_top - bottle.asset.bottom_z + _SETTLE_DROP  # type: ignore[attr-defined]
            self._inner_state[key] = {
                "position": [float(x), float(y), float(z)],
                "quaternion": t3d.euler.euler2quat(0.0, 0.0, yaw).tolist(),
            }
        state = self._inner_state[key]
        bottle.pose.position = list(state["position"])
        bottle.pose.quaternion = list(state["quaternion"])


@dataclass
class _BottleSpatialDRCfg(SpatialDRCfg):
    randmizer_class: Type[Randomizer] = _BottleSpatialDR


@dataclass
class _BottleSideLimitDRCfg(_BottleSpatialDRCfg):
    # How far the bottle may stand to either side (m), by how far ahead of the robot's spawn point it stands (m):
    # ((ahead, side), ...), linearly interpolated, flat beyond the ends. The side offset is drawn within it.
    bottle_side_limit: tuple[tuple[float, float], ...] = ()


def _spatial_cfg(robot_x: tuple[float, float], bottle_y: float) -> _BottleSpatialDRCfg:
    """The layout. The table (hssd:scene2's, 2.06 m deep) spans x = -0.73..1.33; its near edge faces the robot.
    - Robot: its pelvis starts at an x in `robot_x`, on y = 0, facing the table.
    - Bottle: 0.07-0.13 m in from the edge, at most `bottle_y` to either side of y = 0.
    - Clutter: 0.4-0.8 m in, out of the way.
    """
    return _BottleSpatialDRCfg(
        spatial_mode="random",
        robot_region=Box(low=[robot_x[0], 0.0, 0.0], high=[robot_x[1], 0.0, 0.0]),
        articulated_region=Box(low=[-0.66, -bottle_y], high=[-0.60, bottle_y]),
        articulated_rotate_z=Box(low=-np.pi, high=np.pi),
        distractors_region=Box(low=[-0.30, -0.35], high=[0.10, 0.35]),
    )


class _UncapBottleDRManager(TabletopGraspDRManager):
    """TabletopGraspDRManager that keeps the task's own rooms at every DR level.

    From dr_level 1 up, the stock manager fixes the room to hssd:scene0. Its 1.25 x 0.79 m table doesn't fit this
    task's layout: the bottle would stand past the table edge and fall to the floor. This manager keeps the
    configured room and lets the stock levels fix it (one room, middle pose and table height).
    """

    def __init__(self, level: int, **kwargs) -> None:
        self._rooms = list(kwargs["scene"].room_choices)
        super().__init__(level, **kwargs)

    def set_level(self, dr_level: int) -> None:
        super().set_level(dr_level)
        scene = self.randomizers.get("scene")
        if scene is not None and list(scene.cfg.room_choices) != self._rooms:
            scene = deepcopy(scene)  # the cfg may be the task's class-level one
            scene.cfg.room_choices = list(self._rooms)
            self.randomizers["scene"] = scene


@TaskRegistry.register("g1_wholebody_uncap_bottle_teleop")
class G1WholebodyUncapBottleTaskTeleop(Task):
    uid: str = "g1_wholebody_uncap_bottle_teleop"
    label: str = "G1 TELEOP Uncap Bottle"
    description: str = (
        "A bimanual task where the G1 robot picks up a bottle with one hand and pulls its cap off with the other."
    )

    metadata: dict[str, Any] = {
        "physics_dt": 0.005,
        "control_hz": 200,
        "render_hz": 50,
        "dr_level": 0,
        "version": 1.0,
        "reward_dt": 0.02,
        "image_dt": 0.033333,
        "need_gravity": True,
        "max_episode_steps": 1500,
        "success_criteria": 0.6,  # 20 control steps (0.4 s) with the bottle held up and uncapped
    }

    robot_cfg: dict[str, Any] = dict(
        uid="g1_sonic",
    )

    lift_height: float = 0.01  # m the bottle's centre of mass must rise above where it stood (2.5 cm until 2026-11-12)
    # teleop_decoupled_wbc --record stops once the dataset has this many successful episodes, counting those already
    # there (the default of --num-episodes; other tasks: 100 new ones per session)
    teleop_target_episodes: int | None = 80
    arms_start_forward: bool = True  # spawn the arms in the controller's start pose (`_start_arms_forward`)

    sensor_cfgs: dict[str, SensorCfg] = dict(
        head_stereo=StereoCameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=640,
            height=360,
            focal_length=1.93,
            fov=np.deg2rad(110),
            near=0.2,
            far=5,
            baseline=0.05,
            pose=dict(
                position=[0.0, 0.0, 0.0],
            ),
        ),
    )

    # Teleop-only cameras (see the module docstring); `<id>_left` and `<id>_right` are each camera's two eyes.
    side_view_cfgs: dict[str, SensorCfg] = dict(
        side_left=_side_view_cfg(+1.0),
        side_right=_side_view_cfg(-1.0),
    )

    dr_cfgs: dict[str, RandomizerCfg] = dict(
        language=LanguageDRCfg(
            instructions=[
                "pick up the bottle with one hand and take off its cap with the other hand",
            ]
        ),
        articulated=ArticulatedObjectDrCfg(
            asset_id="bottle_cap:0",
        ),
        distractors=DistractorDRCfg(
            res_id="graspnet1b",
            number_of_distractors=2,
            allow_duplicates=False,
            exclude=list(_BOTTLE_LIKE),
        ),
        # Robot: its pelvis starts 0.12-0.17 m from the table edge (x = -0.73) and settles about 7 cm further back,
        # 0.19-0.24 m from it. The bottle is then 0.26-0.37 m ahead of the pelvis, between the hands of the start pose
        # (0.25-0.36 m ahead, 0.74-0.81 m up): within reach without a step. Standing this close needs
        # `arms_start_forward`. Until 2026-11-12 the robot settled 0.37-0.42 m from the edge, the bottle 0.44-0.55 m
        # ahead (the small-bottle task still does).
        # Bottle: at most 2.5 cm to the side (6 cm until 2026-11-12), so it stands between the hands. Their thumb
        # tips reach to about 9 cm from the centre line; a bottle 4 cm or more to the side gets clipped by a thumb
        # while the robot settles and can fall off the table (MuJoCo, 2026-11-12: 3 cm clear in 8 of 8 worst cases).
        # Spatial level 1 spreads it further: `wide_bottle_side_limit`.
        spatial=_spatial_cfg(robot_x=(-0.90, -0.85), bottle_y=0.025),
        camera=CameraDRCfg(
            cam_id="franka_camera",
        ),
        scene=TabletopSceneDRCfg(
            table_position=Box(low=[0.3, 0], high=[0.3, 0]),
            table_height=Box(low=0.67, high=0.68),
            room_choices=["hssd:scene2"],
            scene_manager="hssd",
        ),
        lighting=LightingDRCfg(
            light_mode="random",  # fixed, random
            light_num=(2, 3),
            light_color_temperature=Box(low=2001, high=8001),
            light_intensity=Box(low=1e4 * 0.8, high=1e4 * 1.2),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 1.1),
            light_spacing=Box((1.0, 1.0), (2.0, 2.0)),
            light_position=Box((-0.5, -0.5, 1.3), (0.5, 0.5, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        material=MaterialDRCfg(
            material_mode="rand_all",  # fixed, rand_all, rand_tableground, rand_objects
        ),
    )

    # Spatial level 1 (`set_spatial_level`, --spatial-level 1): the bottle up to 3 cm to either side, and up to 6 cm
    # when it stands the furthest ahead of the robot, as ((ahead, side), ...) in m (`_BottleSideLimitDRCfg`). Not the
    # generic 5 cm more: the bottle stands in the gap between the thumbs of the start pose, which reach to within
    # ~9 cm of the centre line at its height, and that gap is wider further ahead of the robot. A bottle beyond it
    # gets hit by a thumb while the robot settles and falls over or off the table. Measured on the settling robot in
    # MuJoCo (2026-11-12): clear sideways to 3 cm when the bottle stands 0.19-0.27 m ahead of the spawn point, 4 cm at
    # 0.28 m, 5 cm at 0.29 m, 6 cm at 0.30 m (the farthest these ranges allow). None: the generic widening.
    wide_bottle_side_limit: tuple[tuple[float, float], ...] | None = ((0.275, 0.03), (0.30, 0.06))

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        split: str = "train",  # train, val, test
        render_hz: int | None = None,
        dr_level: int = 0,
        target: str | None = None,
        success_criteria: float | None = None,
        *args,
        **kwargs,
    ):
        # lazy init instance variables
        self._instruction = None
        self._layout = None
        self.reward = 0.0
        self._progress = self._new_progress()
        self._cap: CapMechanism | None = None  # the cap of the current MuJoCo model, made at the first check

        if target is not None:
            print(f"[{self.uid}] ignoring target={target!r}: the task object is always the capped bottle")

        # teleop/eval CLIs pass --success-criteria here; replay and eval may also set the attribute later
        self.success_criteria = self.metadata["success_criteria"] if success_criteria is None else success_criteria

        self.robot_cfg.update(dict(uid=robot_uid))
        self._robot = RobotRegistry.make(**self.robot_cfg, **kwargs)
        if self.arms_start_forward:
            _start_arms_forward(self._robot)

        drmgr = _UncapBottleDRManager(level=dr_level, **self.dr_cfgs)
        super().__init__(
            dr=drmgr,
            split=split,
            render_hz=render_hz,
            dr_level=dr_level,
            *args,
            **kwargs,
        )

    @property
    def layout(self) -> Layout:
        """Returns the layout of the task."""
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction

    @property
    def bottle(self) -> ArticulatedObjectActor:
        actor = self.layout.actors["articulated"]
        assert isinstance(actor, ArticulatedObjectActor)
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            "joint_qpos": spaces.Box(
                -np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32
            ),  # type:ignore
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)

    def reset(self, seed: int | None = None, options: Optional[dict[str, Any]] = None) -> None:
        super().reset(seed, options)
        split = self.metadata.get("split", "train")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        self._instruction = lang_dr(split)
        self.reward = 0.0
        self._progress = self._new_progress()
        self._cap = None  # the engine compiles a new model, with the cap back on
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def state_dict(self) -> Dict[str, Any]:
        return super().state_dict()

    # ------------------------------------------------------------------
    # success
    # ------------------------------------------------------------------

    @staticmethod
    def _new_progress() -> dict[str, Any]:
        return dict(time=None, rest_z=None, lift=0.0, cap_off=0.0, cap_free=False, supported=True,
                    bottle_hands=[], cap_hands=[], two_hands=False, done=False)

    @property
    def progress(self) -> dict[str, Any]:
        """The goal state at the last check: how far the bottle is lifted and the cap is from its seat (m), whether
        the cap has come off, whether the bottle still touches something other than the robot, which hands touch
        the bottle and the cap, and whether the goal holds."""
        return {k: v for k, v in self._progress.items() if k not in ("time", "rest_z")}

    def _update(self, mujoco_env) -> None:
        """Once per control step: take the cap off if it has been pulled far enough, check the goal state, and
        grow or reset the reward."""
        m, d = mujoco_env.mjModel, mujoco_env.mjData
        progress = self._progress
        if progress["time"] == d.time:  # compute_reward and check_success both run each step: count it once
            return
        progress["time"] = d.time

        if self._cap is None or self._cap.model is not m or self._cap.data is not d:
            self._cap = CapMechanism(m, d)
        cap_free = self._cap.update()
        bottle, cap = self._cap.bottle, self._cap.cap
        robot = m.body("pelvis").id  # root of the robot's body tree
        z = float(d.xipos[bottle][2])  # the bottle's centre of mass
        if progress["rest_z"] is None:  # standing on the table at the episode's first step
            progress["rest_z"] = z
        cap_off = self._cap.off

        hands: dict[int, set[str]] = {bottle: set(), cap: set()}
        supported = False
        for contact in d.contact[: d.ncon]:
            body1, body2 = int(m.geom_bodyid[contact.geom1]), int(m.geom_bodyid[contact.geom2])
            for part, other in ((body1, body2), (body2, body1)):
                if part not in hands or other in hands:
                    continue
                if m.body_rootid[other] == robot:
                    hand = _HAND_BODY.fullmatch(m.body(other).name)
                    if hand:
                        hands[part].add(hand.group(1))
                elif part == bottle:
                    supported = True  # the table, the floor, clutter: anything but the robot (or the cap)

        two_hands = any(side in hands[bottle] and other in hands[cap]  # reported, not required
                        for side, other in (("left", "right"), ("right", "left")))
        lift = z - progress["rest_z"]
        done = cap_free and cap_off >= _CAP_OFF and lift >= self.lift_height and not supported and bool(hands[bottle])
        self.reward = self.reward + _REWARD_STEP if done else 0.0
        progress.update(lift=lift, cap_off=cap_off, cap_free=cap_free, supported=supported,
                        bottle_hands=sorted(hands[bottle]), cap_hands=sorted(hands[cap]), two_hands=two_hands,
                        done=done)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is not None:
            self._update(mujoco_env)
        return self.reward

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self.compute_reward(info, *args, **kwargs) >= self.success_criteria

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task."""
        asset_manager = AssetManager.get("graspnet1b")
        return [ObjectActor(asset=asset) for asset in asset_manager]

    def widened_spatial_cfg(self, cfg: RandomizerCfg) -> RandomizerCfg:
        """Spatial level 1: the bottle within `wide_bottle_side_limit`; the robot and the clutter as at level 0."""
        if self.wide_bottle_side_limit is None:
            return super().widened_spatial_cfg(cfg)
        assert isinstance(cfg, _BottleSpatialDRCfg) and cfg.articulated_region is not None
        side = max(limit for _, limit in self.wide_bottle_side_limit)
        region = cfg.articulated_region
        values = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "bottle_side_limit"}
        values["articulated_region"] = Box(low=[region.low[0], -side], high=[region.high[0], side])
        return _BottleSideLimitDRCfg(**values, bottle_side_limit=self.wide_bottle_side_limit)


@TaskRegistry.register("g1_wholebody_uncap_small_bottle_teleop")
class G1WholebodyUncapSmallBottleTaskTeleop(G1WholebodyUncapBottleTaskTeleop):
    """The uncap-bottle task with the smaller bottle, `bottle_cap:1`, in the uncap-bottle task's setup from before
    2026-11-12: the robot further from the table, its arms spawned down, the bottle up to 6 cm to the side, and a
    2.5 cm lift."""

    uid: str = "g1_wholebody_uncap_small_bottle_teleop"
    label: str = "G1 TELEOP Uncap Small Bottle"
    description: str = (
        "A bimanual task where the G1 robot picks up a small bottle with one hand and pulls its cap off with the other."
    )

    # Own copies: Task.__init__ and this task's __init__ update `metadata` and `robot_cfg` in place.
    metadata: dict[str, Any] = dict(G1WholebodyUncapBottleTaskTeleop.metadata)
    robot_cfg: dict[str, Any] = dict(G1WholebodyUncapBottleTaskTeleop.robot_cfg)
    lift_height: float = 0.025
    arms_start_forward: bool = False
    wide_bottle_side_limit = None  # spatial level 1: 5 cm more to either side, the bottle stands past the fingertips
    teleop_target_episodes = None  # teleop: --num-episodes, 100 new successful episodes per session, as before
    dr_cfgs: dict[str, RandomizerCfg] = dict(
        G1WholebodyUncapBottleTaskTeleop.dr_cfgs,
        articulated=ArticulatedObjectDrCfg(asset_id="bottle_cap:1"),
        # Robot: its pelvis starts 0.32-0.37 m from the table edge, the bottle 0.39-0.50 m ahead of it, just past the
        # fingertips of the start pose, so the operator steps in to reach it. Any closer, and the arms' swing up at
        # spawn hits the table and the robot walks back.
        spatial=_spatial_cfg(robot_x=(-1.10, -1.05), bottle_y=0.06),
    )
