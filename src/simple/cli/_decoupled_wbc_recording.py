# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path


EGO_VIEW_FEATURE = "observation.images.ego_view"

# Per-episode teleop settings that teleop_decoupled_wbc writes next to environment_config in meta/episodes.jsonl, and
# replay_decoupled_wbc copies to the episodes it re-records. Only those in use are written: hand_mode always, vr_view
# with --vr-view, torso_stick (true) with --torso-stick, vla_view / side_view_log in those views, spatial_level with
# --spatial-level 1. A missing vr_view is the head view, a missing spatial_level 0.
TELEOP_EPISODE_KEYS = ("hand_mode", "vr_view", "torso_stick", "vla_view", "side_view_log", "spatial_level")

# teleop_decoupled_wbc --keep-failures saves the tries that failed too, and lists them in this file of the dataset:
#   {"failed_episodes": [3, 7], "reasons": {"3": "marked", "7": "interrupted"}}
# Reasons: "marked" (Left Menu + both index triggers), "reset" (Left Menu + both middle triggers), "time_limit",
# "interrupted" (Ctrl+C). Replay keeps these episodes and the list, postprocess_psi0_sonic.py carries the list over to the
# Psi-0 dataset, and merge_psi0_datasets.py --drop-failed leaves them out of a training set. A dataset without the file
# has no failed episodes.
FAILED_EPISODES_FILE = "meta/failed_episodes.json"


def load_failed_episodes(root: str | Path) -> dict[int, str]:
    """{episode index: reason} of the dataset at `root`'s failed episodes, {} if it lists none."""
    path = Path(root) / FAILED_EPISODES_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    reasons = data.get("reasons", {})
    return {int(i): reasons.get(str(i), "failed") for i in data["failed_episodes"]}


def write_failed_episodes(root: str | Path, failed: dict[int, str]) -> None:
    """Make `failed` ({episode index: reason}) the dataset's list of failed episodes (no file if empty). The file is
    replaced in one step, so a crash never leaves half of it."""
    path = Path(root) / FAILED_EPISODES_FILE
    if not failed:
        path.unlink(missing_ok=True)
        return
    indices = sorted(failed)
    data = {"failed_episodes": indices, "reasons": {str(i): failed[i] for i in indices}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(path)


def mark_failed_episode(root: str | Path, episode_index: int, reason: str) -> None:
    """Add episode `episode_index` to the dataset's failed episodes. Called before the episode is saved: should the save
    die halfway, the episode still counts as failed (prune_failed_episodes drops the mark if it never got saved)."""
    write_failed_episodes(root, {**load_failed_episodes(root), int(episode_index): reason})


def prune_failed_episodes(root: str | Path, total_episodes: int) -> list[int]:
    """Drop the marks of episodes the dataset doesn't have (index >= total_episodes): a save that died after the mark.
    Otherwise the next episode saved, at that index, would count as failed. Returns the indices dropped."""
    failed = load_failed_episodes(root)
    stale = [i for i in failed if i >= total_episodes]
    if stale:
        write_failed_episodes(root, {i: r for i, r in failed.items() if i < total_episodes})
    return stale


def _normalize_shape(ego_view_shape) -> list[int] | None:
    if ego_view_shape is None:
        return None

    shape = [int(dim) for dim in ego_view_shape]
    if len(shape) != 3:
        raise ValueError(f"Expected ego-view image shape to be HWC, got {tuple(shape)}")
    return shape


def set_ego_view_feature_shape(features: dict, ego_view_shape) -> None:
    """Match the LeRobot video metadata to the actual egocentric image shape."""
    shape = _normalize_shape(ego_view_shape)
    if shape is None:
        return

    features[EGO_VIEW_FEATURE]["shape"] = shape


def validate_existing_ego_view_feature_shape(save_dir: str | Path, ego_view_shape) -> None:
    """Refuse to append 480p frames to a dataset initialized with old 360p metadata."""
    shape = _normalize_shape(ego_view_shape)
    if shape is None:
        return

    info_path = Path(save_dir) / "meta" / "info.json"
    if not info_path.exists():
        return

    with open(info_path, "r") as f:
        info = json.load(f)

    existing_shape = info.get("features", {}).get(EGO_VIEW_FEATURE, {}).get("shape")
    if existing_shape is None:
        return

    existing_shape = [int(dim) for dim in existing_shape]
    if existing_shape != shape:
        raise ValueError(
            f"Existing LeRobot dataset at {save_dir} declares {EGO_VIEW_FEATURE} "
            f"shape {existing_shape}, but the current ego-view frame is {shape}. "
            "Use a clean --save-dir or remove the old dataset before recording."
        )


# SONIC whole-body "token": the 64-dim encoder output / decoder input that
# drives the 29 body joints.  WBC teleop records it (+ the 14 hand joints) as the
# LeRobot `action` instead of the 43 decoded joint targets.  Fixed at 64 by every
# SONIC release config (encoder: dimension: 64).
WBC_TOKEN_DIM = 64


def apply_wbc_token_action(
    features: dict,
    modality_config: dict,
    joint_names: list[str],
    token_dim: int = WBC_TOKEN_DIM,
) -> None:
    """Redefine the LeRobot ``action`` as [SONIC token(token_dim) | hands(14)].

    Applied ONLY in the WBC recorder/replay (teleop_wbc / replay_wbc), never in
    decoupled_wbc.get_dataset_features, so the decoupled teleop/replay pipeline
    keeps its 43-DOF joint action.  Both the WBC recorder and its replay call
    this so the two exporters declare the identical action schema.

    ``joint_names`` is WHOLE_BODY_JOINTS; its last 14 entries are the L-hand(7)
    + R-hand(7) names, reused verbatim so the hand block matches how the
    decoupled action already encoded the hands.
    """
    hand_names = list(joint_names[-14:])
    features["action"] = {
        "dtype": "float64",
        "shape": (token_dim + 14,),
        "names": [f"token_{i}" for i in range(token_dim)] + hand_names,
    }
    # The token dims are one opaque block; hands keep per-joint slices at the new
    # offsets.  The eef / base_height / navigate entries reference OTHER fields
    # via original_key and are unchanged.  (Confirm the layout GR00T expects.)
    modality_config["action"] = {
        "token": {"start": 0, "end": token_dim},
        "left_hand": {"start": token_dim, "end": token_dim + 7},
        "right_hand": {"start": token_dim + 7, "end": token_dim + 14},
        "left_wrist_pos": {"start": 0, "end": 3, "original_key": "action.eef"},
        "left_wrist_abs_quat": {
            "start": 3, "end": 7, "original_key": "action.eef",
            "rotation_type": "quaternion",
        },
        "right_wrist_pos": {"start": 7, "end": 10, "original_key": "action.eef"},
        "right_wrist_abs_quat": {
            "start": 10, "end": 14, "original_key": "action.eef",
            "rotation_type": "quaternion",
        },
        "base_height_command": {
            "start": 0, "end": 1, "original_key": "teleop.base_height_command",
        },
        "navigate_command": {"start": 0, "end": 4, "original_key": "teleop.navigate_command"},
    }
