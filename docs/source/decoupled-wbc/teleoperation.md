# Teleoperation

Stage 1 of the decoupled whole-body control data pipeline: human-in-the-loop
demonstration collection.

Teleoperation is run inside the lightweight **MuJoCo** engine only. This keeps
the control loop latency minimal and allows high-frequency physical interaction
while the demonstration is being tracked. Photorealistic rendering happens later,
in [Replay & Render](replay_render.md).

For headset, controller and motion-tracker configuration, see
[Teleop Setup](teleop_decoupled_setup.md) first.

## Environment variables

Adjust these to match your host topology before running:

```bash
export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES="0"
export DISPLAY=":1"
```

## Example usage

```bash
export TASK_NAME=G1WholebodyOpenTrashCanTeleop-v0

python -m simple.cli.teleop_decoupled_wbc \
  simple/$TASK_NAME \
  --target=graspnet1b:0 \
  --sim-mode=mujoco \
  --record \
  --no-headless \
  --success-criteria=2
```

The same command is installed as the `teleop-decoupled-wbc` entry point, so
`uv run teleop-decoupled-wbc simple/$TASK_NAME ...` is equivalent.

Recorded trajectories are written under
`data/teleop_decoupled_wbc/simple/$TASK_NAME/level-0/`, which is the input
directory for the replay stage.

### Hand modes

`--hand-mode` picks how the controller triggers close the three-finger hands:

* `split` (default): each trigger closes two of the three fingers. The index trigger pinches thumb + index finger,
  the middle trigger pinches thumb + middle finger, and both close all three. The manual reset is `Left Menu` +
  both middle triggers (see [Teleop Setup](teleop_decoupled_setup.md)).
* `binary` (the original mode): either trigger closes the whole hand, and both middle triggers reset the episode.

The mode only changes teleop. Datasets store the finger joint targets, so replay, training and policy inference
work the same for both modes. Each episode's entry in `meta/episodes.jsonl` records its `hand_mode`, and replay
copies it. To keep the two kinds of demos apart, record them to different `--save-dir`s.

### Headset views

`--vr-view` picks what the headset shows. The dataset always records the head camera's left eye
(`observation.images.ego_view`), whatever the view. Without `--vr-view` and `--torso-stick`, teleop is exactly as
before, for every task: the head view, the same cameras, dataset place and episode metadata, and the right joystick
turns the robot.

* `head` (default): the head camera's stereo pair.
* `vla`: the recorded head-camera image exactly as the VLA policy receives it: Psi-0's resize to 320 × 180
  (nearest-neighbour), then its center crop. Both eyes show it, and there is no text on it while teleop runs.
  For another policy, set `--vla-resize H W` and `--vla-crop H W`. The images are MuJoCo's; Psi-0 trains on the
  Isaac re-render made by the replay stage, from the same camera pose and with the same field of view.
* `side`: a fixed stereo camera beside the workspace, at the head camera's 640 × 360. The `A` button swaps
  between the left and the right camera. Only tasks that define `side_view_cfgs` have it (the uncap-bottle tasks).

Repeat the option to collect one dataset per view in one session. The views take turns, one saved episode each,
and the next view is always the one whose dataset holds fewer episodes. A view other than `head` records to
`<save-dir>/vr-<view>/simple/$TASK_NAME/level-0/`. `--num-episodes` counts per view. Each episode records its
`vr_view` in `meta/episodes.jsonl`, plus `vla_view` (the resize and crop) or `side_view_log` (`[frame, side]` at each
swap). Replay copies these.

`--torso-stick` turns the right joystick into a torso control, so the head camera can look past the hands. Push
left or right to lean sideways, forward or back to bend; let go and the torso returns upright. The robot can't turn
on the spot then. At full deflection, the camera moves about 7 cm sideways, 10 cm forward or 4 cm back. The hands
drift a few centimetres while the torso leans. Each episode then records `torso_stick: true`.

The two uncap-bottle datasets, one session (head-camera view as the policy sees it, and the side view):

```bash
python -m simple.cli.teleop_decoupled_wbc simple/G1WholebodyUncapBottleTeleop-v0 \
  --sim-mode=mujoco --record --no-headless --success-criteria=0.6 \
  --vr-view vla --vr-view side --torso-stick --keep-failures
```

The uncap-bottle task needs 80 successful episodes per dataset (`teleop_target_episodes`), and for it
`--num-episodes` counts the whole dataset, including the episodes already there: rerun the command after a break, and
it stops when the dataset has 80, or says there is nothing left to record. The headset shows the dataset's count
(`63/80`). For other tasks `--num-episodes` (100) counts the successful episodes of the session.

### Keeping failed tries

By default a dataset holds only successful episodes: the task's success check ends and saves an episode, and the
manual reset throws away the episode being recorded. With `--keep-failures` the failed tries are saved as well and
listed, so the dataset can be kept whole as an archive, and a training set made from it without them:

* `Left Menu` + `Left index Trigger` + `Right index Trigger` ends the episode being recorded and saves it as failed.
* The manual reset (`Left Menu` + both middle triggers) does the same, instead of throwing the episode away.
* An episode cut off by the time limit (`--max-episode-steps`) or by Ctrl+C is saved as failed too.

The failed episodes are listed, with the reason, in the dataset's `meta/failed_episodes.json`:

```json
{"failed_episodes": [3, 7], "reasons": {"3": "marked", "7": "reset"}}
```

The reasons are `marked` (the index-trigger combo), `reset`, `time_limit` and `interrupted`. `--num-episodes` counts
only the successful episodes, and the headset shows how many failed after the saved count (`3/100, 1 failed`). With
several `--vr-view`s, a failed try is retried in the same view.

`Left Menu` + one index trigger is already a combo: with the right trigger it starts and stops teleop, with the left
one it switches the lower-body policy on and off. The two triggers of the failure combo are never pressed at the same
instant, so with `--keep-failures` such a press waits 0.25 s for the other trigger before taking effect, and teleop
starts up to 0.25 s after `Left Menu` + `Right index Trigger`. Without the option, none of this applies.

[Replay](replay_render.md) keeps the failed episodes, whether or not their replay succeeds, and lists them in its
output. They don't count in the replay success rate. `postprocess_psi0_sonic.py` carries the list over to the Psi-0
dataset, and one command makes the training set without them:

```bash
python scripts/merge_psi0_datasets.py --drop-failed \
  --out data/processed_psi0/${TASK_NAME}_train data/processed_psi0/$TASK_NAME
```

### Wider object placement

`--spatial-level 1` spreads the task's objects (target, articulated object, container) further to either side: 5 cm
more each way than the task's own range, for any task. A task can define its own wider rule instead: the uncap
bottle (2.5 cm to either side by default) goes up to 3 cm, and up to 6 cm when it stands the furthest from the robot,
since a bottle further out gets knocked over by the robot's thumbs while it settles. The robot, the table and the
clutter stay as they are. The level is saved with each episode, and a replay places everything as recorded.

```bash
python -m simple.cli.teleop_decoupled_wbc simple/G1WholebodyUncapBottleTeleop-v0 --sim-mode=mujoco --record \
  --no-headless --success-criteria=0.6 --spatial-level 1
```

Give the test set the same level: `dr_decoupled_wbc ... --spatial-level 1` (see
[Domain randomization](../dr/index.md)). Without the option, nothing changes.

> 💡 To list all customizable options, run
> `python -m simple.cli.teleop_decoupled_wbc --help`.

## Supported decoupled whole-body teleop tasks

* `simple/G1WholebodyOpenTrashCanTeleop-v0`
* `simple/G1WholebodyBendPickTeleop-v0`
* `simple/G1WholebodyBendPickAndPlaceTeleop-v0`
* `simple/G1WholebodyBendHandoverTeleop-v0`
* `simple/G1WholebodyPushOfficeChairTeleop-v0`
* `simple/G1WholebodyUncapBottleTeleop-v0`
* `simple/G1WholebodyUncapSmallBottleTeleop-v0`
* `simple/G1WholebodyOpenFaucetTeleop-v0`
* `simple/G1WholebodyOpenOvenTeleop-v0`
* `simple/G1WholebodyCloseDoorTeleop-v0`
* `simple/G1WholebodyXMovePickTeleop-v0`
* `simple/G1WholebodyXMoveBendCarryBoxSonic-v0`
* `simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`
* `simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0`
* `simple/G1WholebodyHandoverTeleop-v0`
