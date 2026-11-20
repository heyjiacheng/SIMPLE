# G1WholebodyOpenFaucetTeleop-v0 — v2 training set (120 episodes)

Training set for one VLA checkpoint on "open the faucet" that should rely on the
faucet's position relative to the robot, not on the table or the room. Test it on
`G1WholebodyOpenFaucetTeleop-v0_v2_test`, whose table shape, rooms and faucet
placements never appear here.

- 120 episodes, 44,821 frames, 50 fps, ego camera 640x360 (`observation.images.egocentric`)
- `states` 32-D, `action` 36-D (Psi-0 LeRobot v2.1, same layout as the released SIMPLE data)
- one instruction: "move forward to the faucet and open it" (the text the SIMPLE evaluation sends)
- statistics (`meta/stats_psi0.json`: min/max and q01/q99) are computed over these
  120 episodes. The released faucet recipe normalizes with min/max (`action_norm_type=bounds`).

## Composition

| rooms | big round | small round | rectangle | per room |
|---|---|---|---|---|
| scene1, scene2, scene3, scene9, scene25 | 8 each | 2 each | 2 each | 12 |
| scene6, scene7, scene12, scene14, scene15 | – | 6 each | 6 each | 12 |
| total | 40 | 40 | 40 | 120 |

Every table keeps the recorded top height and keeps the edge in front of the
faucet at the recorded distance from the robot, with the whole faucet base on
the table.

- **Big round:** radius 0.65–0.85 m, shifted sideways up to ±0.25 m. The clutter
  objects stay on it.
- **Small round:** radius 0.07–0.09 m, just big enough for the faucet base. No
  clutter.
- **Rectangle:** 0.5–1.8 m wide and 0.5–1.5 m deep. The faucet sits anywhere
  from 20 % to 84 % of the way along the near edge, and the table shifts
  sideways by up to ±0.39 m. The clutter stays on the table when it fits
  (otherwise the table is bare).
  - In the 5 tight kitchens (scene6, 7, 12, 14, 15), tables stay within ±0.55 m
    of the robot's path, so wide rectangles there cannot shift far.
- **Rooms:** 10 HSSD rooms, 12 episodes each. Each episode has a random room
  offset and rotation, lighting and opaque table material.
- **Faucet placements:** 91 different teleop recordings, each used once or twice.
  They cover the recorded range: 0.36–0.49 m ahead of the robot, ±0.10 m
  sideways, ±7° yaw.

## Provenance

| episodes | copied from |
|---|---|
| 0–48 | random-room renders (`..._round_random_room`), big and small round |
| 49–109 | v2 renders (`..._v2_train_renders`): all rectangles, plus round tables for rooms that lacked them |
| 110–117 | big-table renders in scene2 (`..._round_big`) |
| 118–119 | small-table renders in scene2 (`..._round_small`) |

These are intermediate renders of the build; this set holds its own copy of every
episode. Each entry of `meta/episodes.jsonl` names its source render and episode
(`source`). Its `environment_config.table_edit` records the table and room; the
scene2 renders carry no `room` field there, because they keep the recorded room.

## How it was made

`bash scripts/faucet_v2/make_dataset.sh` builds this set and the test set from the
raw teleop recordings (`scripts/faucet_v2/make_dataset.sh` lists the steps):
- `scripts/make_round_table_dataset.py` writes the scene configs with the new tables.
- `scripts/faucet_v2/plan.py` picks the slots and recordings.
- `simple.cli.replay_decoupled_wbc` replays them in MuJoCo and renders them in Isaac Sim;
  `scripts/postprocess_psi0_sonic.py` converts the renders to Psi-0 format.
- `scripts/faucet_v2/assemble.py` selects the episodes and merges them.
