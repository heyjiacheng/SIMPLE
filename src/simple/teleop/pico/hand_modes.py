# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""How the PICO controllers close the G1's Dex3 hands in decoupled-WBC teleop.

Each controller has an index trigger and a middle trigger (the grip). decoupled_wbc's PicoStreamer reads both
(pressed = past 0.5) and encodes one of four gestures per hand: none, index trigger only, both, middle trigger only.
G1GripperInverseKinematicsSolver decodes the gesture and returns the hand's 7 joint targets. A hand mode only changes
the pose each gesture asks for. The streamer, the button reading and everything after the solver are shared.

- ``split`` (the default): each trigger closes two of the three fingers. The index trigger pinches the thumb and the
  index finger, the middle trigger the thumb and the middle finger, and both triggers close all three (the binary
  mode's grasp).
- ``binary`` (the original): every gesture closes all three fingers. With one trigger, the thumb turns toward the
  index finger (index trigger) or the middle finger (middle trigger) and meets it, while the other finger curls in
  alongside. In practice the hand is open or closed. ``split`` uses the same poses, but leaves open the finger the
  thumb doesn't meet.

Pressing both middle triggers also resets the episode. In ``split`` mode they close fingers, so PicoDecoupledAgent
then also wants the left menu button held for a reset.

Recording, replay and the trained policies use the 7 joint targets per hand, not the gestures. So a dataset replays
and trains the same whichever mode collected it, and a policy needs no hand mode at inference: it outputs finger
angles and reproduces whatever grasps its demonstrations used. The teleop CLI writes the mode into each episode's
metadata so that datasets can be told apart.
"""

from __future__ import annotations

import numpy as np
from decoupled_wbc.control.teleop.solver.hand.g1_gripper_ik_solver import G1GripperInverseKinematicsSolver
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)

HAND_MODES = ("binary", "split")

# Slots of the 7 joint targets that move each physical finger in sim. The command is ordered thumb, index, middle, but
# G1Sonic writes it onto the MJCF actuators, which are ordered thumb, middle, index. So slots 3:5 move the middle
# finger and 5:7 the index finger. If that swap is ever fixed, swap these two.
_MIDDLE = slice(3, 5)
_INDEX = slice(5, 7)


class G1SplitFingerSolver(G1GripperInverseKinematicsSolver):
    """The ``split`` hand mode: index trigger -> thumb + index finger, middle trigger -> thumb + middle finger,
    both -> all three fingers."""

    def _get_index_close_q_desired(self) -> np.ndarray:  # index trigger only
        q = super()._get_index_close_q_desired()  # the thumb meets the index finger
        q[_MIDDLE] = 0.0
        return q

    def _get_ring_close_q_desired(self) -> np.ndarray:  # middle trigger only
        q = super()._get_ring_close_q_desired()  # the thumb meets the middle finger
        q[_INDEX] = 0.0
        return q

    # Both triggers: the inherited _get_middle_close_q_desired closes all three fingers.


def make_hand_ik_solvers(
    mode: str = "split",
) -> tuple[G1GripperInverseKinematicsSolver, G1GripperInverseKinematicsSolver]:
    """The left and right hand solvers for hand mode `mode`."""
    if mode == "binary":
        return instantiate_g1_hand_ik_solver()
    if mode == "split":
        return G1SplitFingerSolver("left"), G1SplitFingerSolver("right")
    raise ValueError(f"unknown hand mode {mode!r}, expected one of {HAND_MODES}")
