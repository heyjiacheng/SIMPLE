# G1WholebodyOpenFaucetTeleop-v0 — v2 test set (30 episodes)

Held-out test set for checkpoints trained on
`G1WholebodyOpenFaucetTeleop-v0_v2_train`. Everything a table- and
room-invariant policy should ignore is new here:

- **Table shape:** isosceles triangles with the base toward the robot and the
  faucet on that edge. Training has only rectangles and round tables.
- **Rooms:** 6 HSSD rooms no training episode uses (scene10, 13, 17, 23, 28, 42),
  5 episodes each.
- **Faucet placements:** 15 teleop recordings no training episode uses. They
  span −0.087 to +0.098 m sideways and 0.36–0.49 m ahead of the robot.

Same format as the training set (Psi-0 LeRobot v2.1, 50 fps, ego camera 640x360,
`states` 32-D, `action` 36-D, "move forward to the faucet and open it").
30 episodes, 10,049 frames.

## Episodes

Episodes `2k` and `2k+1` replay the same recording, in two different rooms.
Recordings are ordered from the faucet furthest to the robot's right (k = 0) to
furthest to its left (k = 14).

- **Even episodes:** a big triangle, base 1.0–1.8 m and height 0.9–1.5 m.
- **Odd episodes:** a small triangle just big enough for the faucet base
  (about 0.2 m).
- **Clutter:** no test table has clutter objects; they do not fit on the
  narrowing triangle. In training, the big round tables and some rectangles
  have them.

| k | big triangle (episode 2k) | small triangle (episode 2k+1) | faucet sideways |
|---|---|---|---|
| 0 | scene23 | scene28 | −0.087 m |
| 1 | scene10 | scene17 | −0.078 m |
| 2 | scene13 | scene42 | −0.073 m |
| 3 | scene28 | scene23 | −0.061 m |
| 4 | scene17 | scene10 | −0.043 m |
| 5 | scene42 | scene13 | −0.034 m |
| 6 | scene23 | scene28 | −0.022 m |
| 7 | scene10 | scene17 | −0.012 m |
| 8 | scene13 | scene42 | +0.002 m |
| 9 | scene28 | scene23 | +0.027 m |
| 10 | scene17 | scene10 | +0.033 m |
| 11 | scene42 | scene13 | +0.048 m |
| 12 | scene23 | scene28 | +0.060 m |
| 13 | scene10 | scene17 | +0.079 m |
| 14 | scene13 | scene42 | +0.098 m |

Each episode's `environment_config` holds its scene, which the SIMPLE evaluation
loads, and `table_edit` records the triangle and the room.

## Closed-loop evaluation

Serve the checkpoint from the Psi-0 repository; the server listens on port 22085:

```bash
bash scripts/deploy/serve_psi0_simple.sh <run dir> <checkpoint step>
```

Then run the evaluation from the SIMPLE repository:

```bash
TASK_NAME=G1WholebodyOpenFaucetTeleop-v0 uv run eval-decoupled-wbc \
    simple/G1WholebodyOpenFaucetTeleop-v0 psi0_decoupled_wbc train \
    --data-format lerobot \
    --data-dir data/processed_psi0/G1WholebodyOpenFaucetTeleop-v0_v2_test \
    --num-episodes 30 \
    --host 127.0.0.1 --port 22085 --headless
```

- The success threshold defaults to the task's 0.2, the same one the training
  replays had to pass. Every test episode passes it when replaying its recording.
- Videos go to
  `data/evals/psi0_decoupled_wbc/<checkpoint>.G1WholebodyOpenFaucetTeleop-v0/G1WholebodyOpenFaucetTeleop-v0_v2_test/episode_<i>/`.

This set needs the triangle (`prism`) table support added to SIMPLE together
with it.
