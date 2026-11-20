# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""
Teleoperation using the decoupled whole-body control policy from decoupled_wbc.
"""

from __future__ import annotations

import enum
import typer
from typing_extensions import Annotated, TYPE_CHECKING
import gymnasium as gym
import numpy as np
import simple.envs as _  # import all envs

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

from simple.agents.pico_decoupled_agent import PSI0_CROP, PSI0_RESIZE, VR_VIEWS, PicoDecoupledAgent, policy_view
from simple.cli._decoupled_wbc_recording import (
    FAILED_EPISODES_FILE,
    load_failed_episodes,
    mark_failed_episode,
    prune_failed_episodes,
    set_ego_view_feature_shape,
    validate_existing_ego_view_feature_shape,
)
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.core.task import Task
from simple.robots.g1_sonic import G1Sonic

import json
import os
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import faulthandler
import signal
import time
import tyro
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Episode recording state machine
# ---------------------------------------------------------------------------

class RecordingState(enum.Enum):
    WAITING_FOR_LANDING = "waiting_for_landing"
    RECORDING = "recording"
    EPISODE_DONE = "episode_done"


LIFT_THRESHOLD = 0.10  # metres above initial target z to end episode


def _save_episode_env_config(exporter, task, episode_index: int, teleop_info: dict, teleop_cameras=()):
    """Write the task state_dict as environment_config, and the teleop settings (hand mode, VR view, ...; see
    TELEOP_EPISODE_KEYS), into episodes.jsonl. `teleop_cameras`: sensor_cfgs this session added for the headset only
    (--vr-view side), left out of environment_config like any other teleop setting."""
    from simple.utils import NumpyArrayEncoder
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    env_conf = task.state_dict()
    layout_cameras = env_conf.get("layout", {}).get("cameras", {})
    for name in teleop_cameras:
        env_conf["sensor_cfgs"].pop(name, None)
        for camera in (name, f"{name}_left", f"{name}_right"):
            layout_cameras.pop(camera, None)
    lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
    lines[episode_index].update(teleop_info)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def _init_exporter(
    save_dir: str,
    task_prompt: str,
    robot_model,
    obj_names: list[str],
    joint_names: list[str],
    ego_view_shape=None,
):
    """Create a Gr00tDataExporter for LeRobot-format recording."""
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    # A session that exits before saving its first episode leaves meta/info.json
    # but no meta/episodes.jsonl. LeRobot can't resume from that stub and falls
    # back to querying the HF Hub (crashes offline), so drop it and start fresh.
    if os.path.isdir(save_dir) and not os.path.exists(os.path.join(save_dir, "meta", "episodes.jsonl")):
        import shutil

        print(f"[Record] Removing empty dataset stub (0 episodes) at {save_dir}")
        shutil.rmtree(save_dir)

    features = get_dataset_features(robot_model)
    set_ego_view_feature_shape(features, ego_view_shape)
    validate_existing_ego_view_feature_shape(save_dir, ego_view_shape)
    features["observation.state"]["names"] = joint_names # state joint names
    modality_config = get_modality_config(robot_model)

    # # Add torso RPY command feature (3D: roll, pitch, yaw)
    # features["observation.torso_rpy_command"] = {
    #     "dtype": "float64",
    #     "shape": (3,),
    #     "names": ["roll", "pitch", "yaw"],
    # }

    # Add object poses feature: each object has 7D (pos xyz + quat wxyz)
    num_objects = len(obj_names)
    if num_objects > 0:
        obj_names_flat = []
        for name in obj_names:
            for suffix in ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]:
                obj_names_flat.append(f"{name}.{suffix}")
        features["observation.object_poses"] = {
            "dtype": "float64",
            "shape": (num_objects * 7,),
            "names": obj_names_flat,
        }

    exporter = Gr00tDataExporter.create(
        save_root=save_dir,
        fps=50,
        features=features,
        modality_config=modality_config,
        task=task_prompt,
    )
    return exporter


def _build_frame(agent, obj_names: list[str], observation, privileged_info, action, target_waist):
    """Assemble one recording frame from current sim state and agent caches."""
    proprio = privileged_info["proprio"]
    rm = agent._dwbc_robot_model

    # action: 43-DOF target q in robot_model joint order
    # NOTE: joint matches with rm.joint_names
    action_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=action["target_q"],
        left_hand_actuated_joint_values=action["left_hand_q"], # natural order: thumb/index/middle
        right_hand_actuated_joint_values=action["right_hand_q"],
    )

    # # Extract torso RPY command directly from waist joints (waist_yaw, waist_roll, waist_pitch)
    # # These three joints fully determine the torso orientation relative to pelvis
    # waist_indices = rm.get_joint_group_indices("waist")
    # torso_rpy_command = action_q[waist_indices]

    mjcf_to_natural_order = lambda q: np.concatenate([q[:3], q[5:7], q[3:5]])
    proprio_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=proprio["body_q"],
        left_hand_actuated_joint_values=mjcf_to_natural_order(proprio["left_hand_q"]),
        right_hand_actuated_joint_values=mjcf_to_natural_order(proprio["right_hand_q"]),
    )
    proprio_joints = dict(zip(rm.joint_names, proprio_q))
    from simple.robots.g1_sonic import WHOLE_BODY_JOINTS
    assert np.allclose(observation["joint_qpos"], np.array([proprio_joints[joint] for joint in WHOLE_BODY_JOINTS], dtype=np.float32))

    # override waist joints in action_q with target_waist from IK, to reflect the actual input to lower-body policy
    action_q[12:15] = target_waist  # waist joints are indices 12,13,14 in robot_model order
    frame = {
        "observation.images.ego_view": observation["head_stereo_left"],
        "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64), # obs_state
        "observation.eef_state": np.asarray(action["action_eef"], dtype=np.float64), # FIXME
        "action": np.asarray(action_q, dtype=np.float64),
        "action.eef": np.asarray(action["action_eef"], dtype=np.float64), # 1-cycle delayed teleop eef
        "observation.img_state_delta": np.array([0.0], dtype=np.float32), # FIXME
        "teleop.navigate_command": np.asarray(
            action["navigate_cmd"],
            dtype=np.float64
        ),
        "teleop.base_height_command": np.asarray(
            action["base_height_command"],
            dtype=np.float64
        ),
        "observation.base_pose": np.asarray(
            proprio["floating_base_pose"], 
            dtype=np.float64
        ),
        "observation.base_vel": np.asarray(
            proprio["floating_base_vel"],
            dtype=np.float64
        ),
    }

    # Object poses: concatenate all object (pos + quat) in order
    if obj_names:
        obj_poses = []
        for name in obj_names:
            obj_poses.append(privileged_info[name])
        frame["observation.object_poses"] = np.concatenate(obj_poses).astype(np.float64)

    return frame


def _load_sonic_config() -> dict:
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    # sonic_config["high_elbow_pose"] = True
    return sonic_config

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyBendPick-v1",
    target: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    render_hz: Annotated[int, typer.Option()] = 50,
    data_format: Annotated[str, typer.Option()] = "lerobot",
    save_dir: Annotated[str, typer.Option()] = "data/teleop_decoupled_wbc",
    num_episodes: Annotated[
        int | None,
        typer.Option(
            help="Stop after this many successful episodes. By default 100 new ones in this session. A task that sets "
            "teleop_target_episodes (the uncap bottle: 80) counts the whole dataset instead, the episodes already "
            "there too, and that number is the default. With several --vr-view: per view."
        ),
    ] = None,
    shard_size: Annotated[int, typer.Option()] = 100,
    dr_level: Annotated[int, typer.Option()] = 0,
    record: Annotated[bool, typer.Option()] = False,
    success_criteria: Annotated[float, typer.Option()] = 3,
    hand_mode: Annotated[
        str,
        typer.Option(
            help="How the triggers close the hands. split: index trigger = thumb + index finger, middle trigger = "
            "thumb + middle finger, both = full grasp; reset is left menu + both middle triggers. binary: a trigger "
            "closes the whole hand (the original mode); reset is both middle triggers. Saved per episode in "
            "meta/episodes.jsonl."
        ),
    ] = "split",
    vr_view: Annotated[
        list[str] | None,
        typer.Option(
            help="What the headset shows. head (default): the head camera's stereo pair. vla: the head camera's left "
            "eye, the image the dataset records, exactly as the VLA policy sees it (its resize and center crop, "
            "--vla-resize / --vla-crop), in both eyes and with no text while teleop runs. side: one of the task's "
            "side cameras, in stereo; the A button swaps the left and the right one. Repeat the option "
            "(--vr-view vla --vr-view side) to record one dataset per view in one session: the views take turns, "
            "one saved episode each. A view other than head records to <save-dir>/vr-<view>/<env-id>/level-<n>, "
            "and --num-episodes counts per view. Saved per episode in meta/episodes.jsonl."
        ),
    ] = None,
    torso_stick: Annotated[
        bool,
        typer.Option(
            help="The right joystick leans the torso (left/right and forward/back) so the head camera can see past "
            "the hands; it comes back upright when released. The robot can't turn on the spot then."
        ),
    ] = False,
    vla_resize: Annotated[
        tuple[int, int], typer.Option(help="--vr-view vla: the policy's image resize, height width (Psi-0's).")
    ] = PSI0_RESIZE,
    vla_crop: Annotated[
        tuple[int, int], typer.Option(help="--vr-view vla: the policy's center crop after it, height width (Psi-0's).")
    ] = PSI0_CROP,
    keep_failures: Annotated[
        bool,
        typer.Option(
            help="Keep the tries that fail (needs --record). Left Menu + both index triggers ends the episode being "
            "recorded and saves it as a failed episode; so does the reset (Left Menu + both middle triggers), which "
            f"otherwise throws the episode away. They are listed in <dataset>/{FAILED_EPISODES_FILE}, with an episode "
            "cut by the time limit or by Ctrl+C. --num-episodes counts the other episodes, those the task's success "
            "check ended. Replay keeps the failed episodes and the list, and scripts/merge_psi0_datasets.py "
            "--drop-failed makes a training set without them."
        ),
    ] = False,
    spatial_level: Annotated[
        int,
        typer.Option(
            help="How far to either side the task's objects are placed. 0: the task's own placement. 1: wider, 5 cm "
            "more to each side, or the task's own wider rule (the uncap bottle: 3 cm, up to 6 cm). The robot, the "
            "table and the clutter stay as they are. Give the test set the same level (dr_decoupled_wbc "
            "--spatial-level). Saved per episode in meta/episodes.jsonl."
        ),
    ] = 0,
):
    assert sim_mode in ["mujoco"], f"Invalid sim_mode {sim_mode} for teleop."
    if spatial_level not in Task.SPATIAL_LEVELS:
        raise typer.BadParameter(f"--spatial-level {spatial_level}: expected one of {list(Task.SPATIAL_LEVELS)}")
    if keep_failures and not record:
        raise typer.BadParameter("--keep-failures saves failed episodes: it needs --record")
    vr_views = list(dict.fromkeys(vr_view or ["head"]))  # the given order, without repeats
    if unknown := [v for v in vr_views if v not in VR_VIEWS]:
        raise typer.BadParameter(f"unknown --vr-view {unknown}, expected some of {list(VR_VIEWS)}")
    # Only a session given a view other than head does anything new for the views. Without --vr-view, teleop builds
    # the env, renders, records, saves and prints exactly as it always has.
    view_session = vr_views != ["head"]
    if "vla" in vr_views:
        try:
            policy_view(np.zeros((8, 8, 3), np.uint8), vla_resize, vla_crop)
        except ValueError as e:
            raise typer.BadParameter(f"--vla-crop: {e}") from e
    if len(vr_views) > 1 and not record:
        print(f"[TeleopDecoupled] Without --record the views don't take turns: showing {vr_views[0]} only")
    sim_cnt = 0

    # If the session seems stuck: `kill -USR1 <pid>` prints every thread's stack to the terminal.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    print(f"[TeleopDecoupled] pid {os.getpid()} (if it seems stuck: kill -USR1 {os.getpid()} prints the stacks)")

    sonic_config = _load_sonic_config()

    print(f"Creating environment: {env_id}")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=render_hz,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=headless,
        max_episode_steps=max_episode_steps,
        sonic_config=sonic_config,
        target=target,
        dr_level=dr_level,
        success_criteria=success_criteria,
        # --vr-view: the cameras rendered change from step to step (skip_cameras below), which gym's checker rejects.
        # Otherwise None: gym's default, the checker, as before.
        disable_env_checker=True if view_session else None,
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task = sonic_env.task
    robot = task.robot
    assert sonic_env.spec is not None
    assert isinstance(robot, G1Sonic)
    if spatial_level:
        task.set_spatial_level(spatial_level)  # before the first reset places the objects
        print(f"[TeleopDecoupled] Spatial level {spatial_level}: the task's objects are spread further to either side")

    side_cfgs = {}
    if "side" in vr_views:
        side_cfgs = getattr(task, "side_view_cfgs", None) or {}
        if set(side_cfgs) != {"side_left", "side_right"}:
            raise typer.BadParameter(f"--vr-view side: {env_id} defines no side_left/side_right side_view_cfgs")
        # Only this process's task gets them, before its first reset builds the cameras. A replay builds the task's
        # own sensor_cfgs, the head camera, whatever the episode's saved config lists.
        task.sensor_cfgs = {**task.sensor_cfgs, **side_cfgs}
        sonic_env.observation_space = task.observation_space

    # A task may set how many successful episodes its dataset needs (teleop_target_episodes): teleop then counts those
    # already in the dataset too. Otherwise --num-episodes (100) counts the ones of this session, as before.
    dataset_target = getattr(task, "teleop_target_episodes", None)
    if num_episodes is None:
        num_episodes = dataset_target or 100
    count_dataset = record and dataset_target is not None
    if count_dataset:
        print(f"[TeleopDecoupled] {task.uid} needs {num_episodes} successful episodes per dataset: those already "
              "there count")

    agent = PicoDecoupledAgent(
        robot, hand_mode=hand_mode, vr_view=vr_views[0], torso_stick=torso_stick,
        vla_resize=vla_resize, vla_crop=vla_crop, keep_failures=keep_failures,
    )
    agent.num_episodes = num_episodes
    if torso_stick:
        print("[TeleopDecoupled] Right joystick: lean the torso (left/right, forward/back); the robot doesn't turn")

    # --- Recording setup ---
    exporter = None  # the dataset of the current view (exporters[view])
    exporters = {}
    run_save_dirs = {}
    view = vr_views[0]
    saved = dict.fromkeys(vr_views, 0)  # episodes saved per view in this session (--keep-failures: the successful ones)
    failed = dict.fromkeys(vr_views, 0)  # --keep-failures: failed episodes saved per view in this session
    saved_before = dict.fromkeys(vr_views, 0)  # teleop_target_episodes: the successful episodes already in the dataset
    failed_before = dict.fromkeys(vr_views, 0)  # and the failed ones
    failure_reason = None  # --keep-failures: why the episode being recorded failed, once it has (FAILED_EPISODES_FILE)
    side_log = []  # "side" view: [frame index, side shown from that frame on], for the episode being recorded
    rec_state = RecordingState.WAITING_FOR_LANDING
    initial_target_z = None
    episodes_saved = 0
    control_decimal =  int(1/ sonic_config["SIMULATE_DT"]/render_hz)
    control_dt = control_decimal * robot.sim_dt  # = 0.02 s (50 Hz)

    def _teleop_info() -> dict:
        """The teleop settings saved with the episode (TELEOP_EPISODE_KEYS), those in use only: without --vr-view,
        --torso-stick and --spatial-level, the hand mode alone, as before."""
        info = {"hand_mode": hand_mode}
        if view_session:
            info["vr_view"] = view
        if torso_stick:
            info["torso_stick"] = True
        if spatial_level:
            info["spatial_level"] = spatial_level
        if view == "vla":
            info["vla_view"] = {"resize": list(vla_resize), "crop": list(vla_crop)}
        if view == "side":
            info["side_view_log"] = list(side_log)
        return info

    side_cameras = {f"{name}_{eye}" for name in side_cfgs for eye in ("left", "right")}

    def _show_cameras():
        """--vr-view: leave out of rendering what neither the dataset nor the headset needs this step, the side cameras
        not shown and, outside the head view, the head camera's right eye (the dataset records the left one). The
        task's own other cameras render as always."""
        if view_session:
            sonic_env.mujoco.skip_cameras = (side_cameras | {"head_stereo_right"}) - agent.vr_cameras

    def _view_note(v: str | None = None) -> str:
        """The "(<view> view)" note in a --vr-view session's messages; nothing otherwise, as before."""
        return f" ({v or view} view)" if view_session else ""

    def _switch_view(new_view: str):
        nonlocal view, exporter
        view = new_view
        agent.set_vr_view(view)
        _show_cameras()
        exporter = exporters.get(view)
        agent.episodes_saved = saved[view]
        agent.episodes_failed = failed[view]

    def _next_view() -> str:
        """The view whose dataset has the fewest episodes (the first given on a tie), among those this session still
        records: with several views they take turns and their datasets stay the same size. --keep-failures: the
        fewest successful episodes."""
        todo = [v for v in vr_views if saved[v] < num_episodes] or vr_views

        def size(v):
            n_failed = len(load_failed_episodes(run_save_dirs[v])) if keep_failures else 0
            return exporters[v].meta.total_episodes - n_failed

        return min(todo, key=lambda v: (size(v), vr_views.index(v)))

    def _on_episode_reset():
        """In recording mode, reset the WBC pipeline to a consistent initial pose,
        skip elastic band drop, and engage the RL policy immediately."""
        if not record:
            return
        if robot.elastic_band is not None:
            robot.elastic_band.enable = False
        agent._dropping = False
        # Reset the entire WBC pipeline (upper-body interpolation, teleop policy,
        # lower-body RL history) so the robot starts from the default pose.
        # The teleop policy is deactivated — upper body holds default pose
        # until the operator presses the Pico activation button.
        agent.reset_policy()
        # Engage the RL lower-body policy so the robot actively stabilizes
        agent._wbc_policy.lower_body_policy.use_policy_action = True
        # Print the spatialDR initial pose
        """ obs = robot.prepare_obs()
        print("\n=" * 60)
        print("[SpatialDR] Robot initial pose after reset", datetime.now().strftime("%-H:%M:%S"), sim_cnt)
        print("-" * 60)
        print(f"  base_pose (qpos[:7]): {obs['floating_base_pose']}")
        print("=" * 60)
        print("[Record] Episode reset: elastic band skipped, policy reset to initial pose")
        print("[Record] Upper body tracking PAUSED — align arms then press activation button") """

    # stabilized_printed = False
    step_pbar = None  # Progress bar for current recording episode
    ready_announced = False  # "press to start" printed for this episode

    # Initialize telemetry
    from decoupled_wbc.control.utils.telemetry import Telemetry
    telemetry = Telemetry(window_size=100)

    # FIXME DEBUG
    # env.unwrapped._telemetry = telemetry
    # env.unwrapped.mujoco._telemetry = telemetry

    observation, privileged_info = env.reset()
    _on_episode_reset()

    # Read obj_names after first reset so layout is populated by domain randomization.
    obj_names = list(sonic_env.mujoco.mj_objects.keys())

    if record:
        # timestamp = datetime.now().strftime("%m%d%H%M%S")
        for v in vr_views:
            # --vr-view: one dataset per view, under vr-<view>. Otherwise the original place.
            view_dir = f"/vr-{v}" if view_session else ""
            run_save_dirs[v] = f"{os.path.abspath(save_dir)}{view_dir}/{sonic_env.spec.id}/level-{dr_level}" #_{timestamp}
            exporters[v] = _init_exporter(
                run_save_dirs[v],
                task.instruction,
                agent._dwbc_robot_model,
                obj_names,
                robot.joint_names,
                observation["head_stereo_left"].shape,
            )
            if view_session:
                print(f"\n[Record] Exporter initialized for the {v} view, saving to {run_save_dirs[v]} "
                      f"({exporters[v].meta.total_episodes} episodes there)")
            else:
                print(f"\n[Record] Exporter initialized, saving to {run_save_dirs[v]}")
            # A failed episode is marked before it is saved: drop a mark whose save died (the next episode takes its
            # index). Only a dataset recorded with --keep-failures has marks.
            if stale := prune_failed_episodes(run_save_dirs[v], exporters[v].meta.total_episodes):
                print(f"[Record] Dropped the failure marks of episodes {stale}: they were never saved")
            if keep_failures:
                print(f"[Record] {len(load_failed_episodes(run_save_dirs[v]))} of the "
                      f"{exporters[v].meta.total_episodes} episodes there are marked failed ({FAILED_EPISODES_FILE})")
            if count_dataset:  # saved[] and failed[] count the dataset's episodes: those there, then the new ones
                failed[v] = failed_before[v] = len(load_failed_episodes(run_save_dirs[v]))
                saved[v] = saved_before[v] = exporters[v].meta.total_episodes - failed[v]
                print(f"[Record] {saved[v]} of the {num_episodes} successful episodes needed are there{_view_note(v)}")
        if keep_failures:
            print("[Record] --keep-failures: Left Menu + both index triggers, or the reset (Left Menu + both middle "
                  "triggers), ends the episode and saves it as failed. So do the time limit and Ctrl+C.")
        print(f"[Record] Ego view shape: {observation['head_stereo_left'].shape}")
        print(f"[Record] Recording {len(obj_names)} objects: {obj_names}")
        if len(vr_views) > 1:
            view = _next_view()
            print(f"[Record] Views take turns, one saved episode each: {vr_views}. First: {view}")
    _switch_view(view)

    try:
        if count_dataset and all(n >= num_episodes for n in saved.values()):
            print(f"[Record] Already {num_episodes} successful episodes in the dataset"
                  f"{' of every view' if view_session else ''}: nothing to record")
            return
        while True:
            step_start = time.monotonic()

            data_frame = {}

            with telemetry.timer("agent.get_action"):
                action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            _show_cameras()  # --vr-view side: A may have swapped the side camera

            data_frame["target_waist"] = action["target_waist"] # for recording the actual input waist to lower-body policy
            data_frame["observation"] = observation.copy()
            data_frame["action"] = action.copy()
            data_frame["privileged_info"] = privileged_info.copy()

            with telemetry.timer("env.step"):
                # synced_proprio = info.copy()  # capture sim state before action for recording
                # for _ in range(int(control_dt / robot.sim_dt)):
                observation, reward, terminated, truncated, privileged_info = env.step(action)

            # if "proprio" in info:
            #     agent.publish_low_state(info["proprio"])

            reset_requested = agent.reset_requested
            if keep_failures:
                # Either combo ends the episode being recorded, saved as failed below (the scene then resets as after
                # any saved episode). With no episode being recorded, the reset resets and the failure combo does nothing.
                marked = agent.failure_requested
                if exporter is not None and rec_state == RecordingState.RECORDING and (marked or reset_requested):
                    rec_state = RecordingState.EPISODE_DONE
                    failure_reason = "marked" if marked else "reset"
                    reset_requested = False
                    print(f"[Record] {'Failure combo' if marked else 'Reset'}: saving the episode as failed")
                elif marked:
                    print("[Record] Failure combo: no episode is being recorded, nothing to mark")

            if reset_requested:
                # Discard any in-progress recording
                if exporter is not None and rec_state == RecordingState.RECORDING:
                    print("[Record] Reset requested, discarding in-progress episode")
                    exporter.skip_and_start_new_episode()
                    # Close progress bar for discarded episode
                    if step_pbar is not None:
                        step_pbar.close()
                        step_pbar = None

                observation, privileged_info = env.reset()
                # synced_proprio = info.copy()  # capture sim state after reset for recording
                _on_episode_reset()
                ready_announced = False
                obj_names = list(sonic_env.mujoco.mj_objects.keys())
                sim_cnt = 0
                # stabilized_printed = False
                rec_state = RecordingState.WAITING_FOR_LANDING
                initial_target_z = None
                print("[TeleopDecoupled] Environment reset complete")

            with telemetry.timer("update_viewer"):
                if not headless:
                    sonic_env.update_viewer()

            with telemetry.timer("update_reward"):
                sonic_env.update_reward()

            with telemetry.timer("update_render"):
                agent.update_render_caches(observation)

            # After every reset teleop is off until the operator activates it, and the settled scene looks frozen.
            # `_stabilized` is read directly, because the `stabilized` property counts its reads.
            if not ready_announced and robot._stabilized and not agent._teleop_policy.is_active:
                print("[TeleopDecoupled] Robot settled: press Left Menu + Right Trigger to start teleop")
                ready_announced = True

            """ # --- Print once when robot first stabilizes ---
            if robot.stabilized and not stabilized_printed:
                stabilized_printed = True
                obs = robot.prepare_obs()
                print("=" * 60)
                print("[Stabilized] Robot pose (vel converged)", datetime.now().strftime("%-H:%M:%S"), sim_cnt)
                print("-" * 60)
                print(f"  base_pose (qpos[:7]): {obs['floating_base_pose']}")
                print(f"  max |qvel[0:6]|:      {np.max(np.abs(robot.mjData.qvel[0:6])):.4f}")
                print("=" * 60) """

            with telemetry.timer("recode_data"):
                # --- Recording logic (runs at 50Hz) ---
                if exporter is not None:
                    if rec_state == RecordingState.WAITING_FOR_LANDING:
                        # Check if robot has landed (elastic band done) AND
                        # teleop policy is active (operator has re-aligned and
                        # pressed the activation button)
                        elastic_done = (
                            not agent._dropping
                            and (robot.elastic_band is None or not robot.elastic_band.enable)
                        )
                        teleop_active = agent._teleop_policy.is_active
                        if elastic_done and teleop_active and robot.stabilized and agent._cached_target_q is not None:
                            rec_state = RecordingState.RECORDING
                            initial_target_z = None
                            sim_cnt = 0  # Reset sim counter for new episode
                            side_log.clear()
                            failure_reason = None
                            # Create progress bar for this recording episode
                            step_pbar = tqdm(desc=f"Recording episode {episodes_saved + 1}{_view_note()}", unit="frame",
                                            leave=False, position=1, bar_format="{desc} {n_fmt} {rate_fmt}")
                            print(f"[Record] Teleop active, starting episode recording{_view_note()}")

                    if rec_state == RecordingState.RECORDING:
                        frame = _build_frame(agent, obj_names, **data_frame)
                        if view == "side" and (not side_log or side_log[-1][1] != agent.side):
                            side_log.append([exporter.episode_buffer["size"], agent.side])  # this frame's index
                        exporter.add_frame(frame)
                        if step_pbar is not None:
                            step_pbar.update(1)

                        # Track initial target height
                        if initial_target_z is None and "target" in privileged_info:
                            initial_target_z = privileged_info["target"][2]

                        # Check episode end condition: target lifted >= 10cm
                        # if initial_target_z is not None and "target" in privileged_info:
                        #     current_z = privileged_info["target"][2]
                        #     if current_z - initial_target_z >= LIFT_THRESHOLD:
                        #         rec_state = RecordingState.EPISODE_DONE

                        if terminated or truncated :
                            rec_state = RecordingState.EPISODE_DONE
                            if keep_failures and not terminated:
                                failure_reason = "time_limit"  # not ended by the task's success check

                    if rec_state == RecordingState.EPISODE_DONE:
                        # Close progress bar for this episode
                        if step_pbar is not None:
                            step_pbar.close()
                            step_pbar = None

                        ep_idx = exporter.episode_buffer["episode_index"]
                        episode_failed, reason = failure_reason is not None, failure_reason
                        failure_reason = None
                        if episode_failed:
                            mark_failed_episode(exporter.root, ep_idx, reason)  # before the save, see there
                        exporter.save_episode()
                        _save_episode_env_config(exporter, task, ep_idx, _teleop_info(), side_cfgs)
                        episodes_saved += 1
                        if episode_failed:
                            failed[view] += 1
                            agent.episodes_failed = failed[view]
                            print(f"[Record] Episode {episodes_saved} saved as failed ({reason}): episode {ep_idx} "
                                  f"of {run_save_dirs[view]}{_view_note()}")
                        else:
                            saved[view] += 1
                            agent.episodes_saved = saved[view]
                            print(
                                f"[Record] Episode {episodes_saved} saved "
                                + (f"({saved[view]}/{num_episodes} successful episodes in {run_save_dirs[view]}"
                                   f"{_view_note()})" if count_dataset
                                   else f"({view} view: {saved[view]} this session, {run_save_dirs[view]})"
                                   if view_session else f"(target lifted {LIFT_THRESHOLD}m)")
                            )
                        if all(n >= num_episodes for n in saved.values()):
                            what = "successful episodes in the dataset" if count_dataset else "episodes"
                            print(f"[Record] Reached {num_episodes} {what}{' per view' if view_session else ''}, stopping")
                            break
                        if len(vr_views) > 1 and not episode_failed:  # after a failure, the same view again
                            _switch_view(_next_view())  # before the reset, so its first frame renders the new view
                        # Auto-reset for next episode
                        observation, privileged_info = env.reset()
                        _on_episode_reset()
                        ready_announced = False
                        print(f"[Record] Scene reset for the next episode{_view_note()}; wait for the robot to settle")
                        obj_names = list(sonic_env.mujoco.mj_objects.keys())
                        sim_cnt = 0
                        # stabilized_printed = False
                        rec_state = RecordingState.WAITING_FOR_LANDING
                        initial_target_z = None
                        continue  # skip sleep / increment for this iteration

            elapsed = time.monotonic() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Only show timing when loop is slow and verbose_timing is disabled
                telemetry.log_timing_info(context=f"{sim_cnt: >3}: Control loop overran by {-sleep_time:.3f} seconds", threshold=0.001)
                # ...

            sim_cnt += 1
    except KeyboardInterrupt:
        print("Simulator interrupted by user.")
        # Close progress bar if active
        if step_pbar is not None:
            step_pbar.close()
        if exporter is not None and rec_state == RecordingState.RECORDING:
            print("[Record] Saving in-progress episode before exit...")
            try:
                ep_idx = exporter.episode_buffer["episode_index"]
                if keep_failures:  # unfinished: failed
                    mark_failed_episode(exporter.root, ep_idx, "interrupted")
                exporter.save_episode()
                _save_episode_env_config(exporter, task, ep_idx, _teleop_info(), side_cfgs)
                episodes_saved += 1
                if keep_failures:
                    failed[view] += 1
                    print(f"[Record] Episode {episodes_saved} saved as failed (interrupted): episode {ep_idx} of "
                          f"{run_save_dirs[view]}{_view_note()}")
                else:
                    saved[view] += 1
                    agent.episodes_saved = saved[view]
                    print(f"[Record] Episode {episodes_saved} saved (interrupted{f', {view} view' if view_session else ''})")
            except Exception as e:
                print(f"[Record] Failed to save interrupted episode: {e}")
    finally:
        # Ensure progress bar is closed
        if step_pbar is not None:
            step_pbar.close()
        for v, view_exporter in exporters.items():
            view_exporter.stop_video_writers()
            total = f"; {saved[v]}/{num_episodes} successful episodes there now" if count_dataset else ""
            new_ok, new_failed = saved[v] - saved_before[v], failed[v] - failed_before[v]  # this session's
            if keep_failures:
                print(f"[Record] Done. {new_ok} episodes and {new_failed} failed ones saved to {run_save_dirs[v]}"
                      f"{_view_note(v)}{total}")
            else:
                print(f"[Record] Done. {new_ok} episodes saved to {run_save_dirs[v]}{_view_note(v)}{total}")
        env.close()
        agent.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
