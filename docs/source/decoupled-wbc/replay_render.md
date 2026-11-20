# Replay & Render

Stage 2 of the decoupled whole-body control data pipeline: photorealistic replay
and Isaac Sim rendering.

Once raw trajectories have been captured with
[Teleoperation](teleoperation.md), pass them into the `replay_decoupled_wbc`
suite. With `--sim-mode=mujoco_isaac` the recorded actions are replayed in
MuJoCo while **Isaac Sim** is driven simultaneously as a synchronized rendering
engine. This turns the raw stream into a standard dataset in **LeRobot** format.

## Example usage

```bash
# Ensure $TASK_NAME matches the task used during teleoperation
python -m simple.cli.replay_decoupled_wbc \
  simple/$TASK_NAME \
  --data-dir=data/teleop_decoupled_wbc/simple/$TASK_NAME/level-0/ \
  --sim-mode=mujoco_isaac \
  --no-headless \
  --render-hz=50 \
  --save-dir=data/replay_decoupled_wbc_output \
  --record \
  --resume \
  --success-criteria=0.2
```

The `replay-decoupled-wbc` entry point runs the same command.

> 💡 **Tip:** If the replay success rate is low, try lowering
> `--success-criteria` first.

Replay saves an episode only if its replay succeeds, with one exception: an
episode the teleop dataset lists as failed (`meta/failed_episodes.json`, written
by teleop `--keep-failures`) is saved whether its replay succeeds or not, and
listed as failed in the output under its new index. These episodes don't count
in the success rate.

## Replaying with another table

The replay rebuilds each scene from the `environment_config` stored in
`meta/episodes.jsonl`, so editing that config changes the scene both in MuJoCo
physics and in the Isaac Sim rendering. `scripts/make_round_table_dataset.py`
writes a copy of a teleop dataset with another table, for the faucet task:

* `--variant big` — round, radius 0.65–0.85 m; the clutter objects move with
  the table.
* `--variant small` — round, radius 0.07–0.09 m, just enough for the faucet
  base; the clutter objects are removed.
* `--variant rect` — rectangular, 0.5–1.8 m wide and 0.5–1.5 m deep, with the
  faucet anywhere along its near edge; the clutter objects move with the table
  if they fit on it.
* `--variant triangle` / `small_triangle` — an isosceles triangle with its base
  toward the robot, 0.9–1.8 m wide or just big enough for the faucet base.

Each table keeps the recorded top height, keeps its edge in front of the faucet
where the recorded edge was (moving it toward the robot only as far as needed to
keep the whole faucet base on the table), and is shifted sideways at random.
Tables that the recorded robot motion would come closer to than it came to the
recorded table are rejected. The chosen parameters are stored per episode under
`environment_config.table_edit`.

```bash
python scripts/make_round_table_dataset.py \
  --src=data/teleop_decoupled_wbc/simple/$TASK_NAME/level-0 \
  --out=data/round_table/big/$TASK_NAME/level-0 \
  --variant=big --seed=0

python -m simple.cli.replay_decoupled_wbc simple/$TASK_NAME \
  --data-dir=data/round_table/big/$TASK_NAME/level-0 \
  --sim-mode=mujoco_isaac --headless --no-webrtc --render-hz=50 \
  --save-dir=data/replay_decoupled_wbc_round_big --record --success-criteria=0.2
```

`--variant mixed` picks the recorded box table, a big or a small round table
per episode. `--rooms rooms.json` also puts each episode in a random HSSD room
(`{"scene1": ["box", "big", "small"], ...}` lists the table types a room fits
without the table running into its furniture;
`{"scene6": {"tables": ["rect", "small"], "y_limit": 0.55}}` also bounds how far
sideways a rectangle or triangle may reach), with a random room offset and
rotation, re-sampled lighting and an opaque table material. These change only
the rendering, so the same episodes succeed as without them. `--plan plan.json`
instead lists exactly which episodes to write, from one or several datasets,
each with its variant and room.

To train on several renders at once, merge their Psi-0 outputs (see below) with
`scripts/merge_psi0_datasets.py`; `--task` gives every episode the same
instruction, and `--select` keeps only chosen episodes of each input.

`bash scripts/faucet_v2/make_dataset.sh` runs all of this for the faucet task:
from the raw teleop recordings it builds a 120-episode training set (big round,
small round and rectangular tables in 10 rooms) and a 30-episode test set whose
table shape (triangles), rooms and teleop recordings never appear in training.
The steps are listed at the top of the script.

## Post-processing

To make the rendered data compatible with the [Psi-0](https://github.com/physical-superintelligence-lab/Psi0)
training pipeline, run `postprocess_psi0_sonic.py`. It accepts wildcards (`*`)
so several replay sessions can be merged into one dataset.

```bash
python scripts/postprocess_psi0_sonic.py \
  --sim-root="data/replay_decoupled_wbc_output*/simple/G1WholebodyPushOfficeChairTeleop-v0/level-0/" \
  --out-dir=data/processed_psi0/G1WholebodyPushOfficeChairTeleop-v0 \
  --skip=0 \
  --total_episodes=100
```

**Key arguments**

* `--sim-root` — input directory holding the generated dataset. Quote the value
  when using wildcards so the shell does not expand them early.
* `--out-dir` — output directory for the Psi-0 compatible dataset.
* `--skip` — number of initial frames to skip (useful to bypass static setup
  frames).
* `--total_episodes` — cap on the number of valid episodes to merge.

Episodes listed as failed in an input's `meta/failed_episodes.json` (teleop
`--keep-failures`) are converted like the others, and listed in the output's
`meta/failed_episodes.json` under their new indices. They count toward
`--total_episodes`. The output is then the whole dataset, for the archive. The
training set without the failed episodes, with its statistics recomputed, is one
command:

```bash
python scripts/merge_psi0_datasets.py --drop-failed \
  --out data/processed_psi0/${TASK_NAME}_train data/processed_psi0/$TASK_NAME
```

Several inputs can be given, to merge them into one training set.
