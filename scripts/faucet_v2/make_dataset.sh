#!/usr/bin/env bash
# Build the v2 faucet datasets from the raw teleop recordings in
# data/psi-data/simple-teleop/G1WholebodyOpenFaucetTeleop-v0:
#   data/processed_psi0/G1WholebodyOpenFaucetTeleop-v0_v2_train   120 episodes
#   data/processed_psi0/G1WholebodyOpenFaucetTeleop-v0_v2_test     30 episodes
#
#   bash scripts/faucet_v2/make_dataset.sh
#
# Intermediate files go to data/faucet_v2_build and are removed once both datasets
# are written (KEEP=1 keeps them). Re-running after an interruption resumes: each
# step is skipped when its output is already there.
#
#   1. scene configs with new tables for the big round, small round and random-room
#      renders of every recording (make_round_table_dataset.py, MuJoCo only)
#   2. MuJoCo replay + Isaac Sim render of those configs
#   3. the v2 plan (plan.py) and its scene configs: rectangles and round tables for
#      training, triangles in unseen rooms for the test set
#   4. MuJoCo replay + Isaac Sim render of the v2 configs
#   5. conversion of the renders to Psi-0 format, selection and merge (assemble.py)
#
# Steps 2 and 4 replay 572 episodes, about 2.5 min each (about a day), one Isaac
# process at a time (11-18 GB RAM). Replays that do not open the faucet are dropped.
set -uo pipefail
cd "$(dirname "$0")/../.."

TASK=G1WholebodyOpenFaucetTeleop-v0
RAW=data/psi-data/simple-teleop/$TASK
SETS=(G1WholebodyOpenFaucetTeleop-v0 G1WholebodyOpenFaucetTeleop2-v0 G1WholebodyOpenFaucetTeleop3-v0)
HERE=scripts/faucet_v2
BUILD=data/faucet_v2_build
OUT_ROOT=${OUT_ROOT:-data/processed_psi0}
CONFIGS=$BUILD/configs
REPLAYS=$BUILD/replays
PSI=$BUILD/psi0
LOGS=$BUILD/logs
export MUJOCO_GL=egl
mkdir -p "$LOGS"

log() { echo "[$(date +%F' '%T)] $*"; }
die() { log "ERROR: $*"; exit 1; }

for part in train test; do
  [ -e "$OUT_ROOT/${TASK}_v2_$part" ] && die "$OUT_ROOT/${TASK}_v2_$part exists; move it away first"
done

make_configs() {  # make_configs OUT LOG ARGS...: scene configs, unless OUT exists
  local out=$1 log=$2; shift 2
  [ -d "$out" ] && return
  log "configs -> $out"
  .venv/bin/python scripts/make_round_table_dataset.py --out "$out" "$@" > "$LOGS/$log" 2>&1 \
    || { rm -rf "$out"; die "make_round_table_dataset.py failed, see $LOGS/$log"; }
}

replay() {  # replay SRC SAVE LOG: replay and render SRC, unless its meta/replay.txt is complete
  local src=$1 save=$2 log=$3
  local total; total=$(wc -l < "$src/meta/episodes.jsonl")
  [ -f "$src/meta/replay.txt" ] && [ "$(wc -l < "$src/meta/replay.txt")" -ge "$total" ] && return
  log "replay $src ($total episodes) -> $save"
  PYTHONUNBUFFERED=1 .venv/bin/python -u src/simple/cli/replay_decoupled_wbc.py "simple/$TASK" \
    --data-dir "$src" --sim-mode mujoco_isaac --headless --no-webrtc --render-hz 50 \
    --record --resume --success-criteria 0.2 --save-dir "$save" >> "$LOGS/$log" 2>&1
  [ "$(wc -l < "$src/meta/replay.txt")" -ge "$total" ] || die "replay of $src stopped early, see $LOGS/$log; re-run to resume"
  log "  $(grep -c True "$src/meta/replay.txt")/$total opened the faucet"
}

convert() {  # convert REPLAYS OUT: Psi-0 format; REPLAYS may be a glob
  log "convert $1 -> $2"
  rm -rf "$2"
  .venv/bin/python scripts/postprocess_psi0_sonic.py --sim-root="$1/simple/$TASK/level-0" \
    --out-dir="$2" --skip=0 --total_episodes=100000 > "$LOGS/convert_$(basename "$2").log" 2>&1 \
    || { rm -rf "$2"; die "conversion to $2 failed, see $LOGS/convert_$(basename "$2").log"; }
}

# 1-2. big round, small round and random-room tables, for every recording
seed=0
for v in big small; do
  for i in 1 2 3; do
    s=${SETS[$((i - 1))]}
    make_configs "$CONFIGS/$v/$s/level-0" "configs_${v}_set$i.log" --src "$RAW/$s/level-0" --variant "$v" --seed "$seed"
    seed=$((seed + 1))
  done
done
for i in 1 2 3; do
  s=${SETS[$((i - 1))]}
  make_configs "$CONFIGS/random_room/$s/level-0" "configs_random_room_set$i.log" \
    --src "$RAW/$s/level-0" --variant mixed --rooms "$HERE/rooms_random.json" --seed $((99 + i))
done
for v in big small random_room; do
  for i in 1 2 3; do
    replay "$CONFIGS/$v/${SETS[$((i - 1))]}/level-0" "$REPLAYS/round_${v}_set$i" "replay_${v}_set$i.log"
  done
done

# 3-4. v2 plan, its scene configs and renders
if [ ! -f "$BUILD/plan/slots.json" ]; then
  for v in big small random_room; do convert "$REPLAYS/round_${v}_set*" "$PSI/${TASK}_round_$v"; done
  log "plan -> $BUILD/plan"
  .venv/bin/python "$HERE/plan.py" --build "$BUILD" > "$LOGS/plan.log" 2>&1 || die "plan.py failed, see $LOGS/plan.log"
fi
make_configs "$CONFIGS/v2/train/level-0" configs_v2_train.log \
  --plan "$BUILD/plan/train_plan.json" --rooms "$HERE/rooms_train.json" --seed 1
make_configs "$CONFIGS/v2/test/level-0" configs_v2_test.log \
  --plan "$BUILD/plan/test_plan.json" --rooms "$HERE/rooms_test.json" --seed 2
for part in train test; do
  replay "$CONFIGS/v2/$part/level-0" "$REPLAYS/v2_$part" "replay_v2_$part.log"
done

# 5. convert and merge
for v in big small random_room; do
  [ -d "$PSI/${TASK}_round_$v" ] || convert "$REPLAYS/round_${v}_set*" "$PSI/${TASK}_round_$v"
done
for part in train test; do
  convert "$REPLAYS/v2_$part" "$PSI/${TASK}_v2_${part}_renders"
done
log "merge -> $OUT_ROOT"
.venv/bin/python "$HERE/assemble.py" --build "$BUILD" --out-root "$OUT_ROOT" > "$LOGS/assemble.log" 2>&1 \
  || die "assemble.py failed, see $LOGS/assemble.log"
for part in train test; do
  cp "$HERE/README_$part.md" "$OUT_ROOT/${TASK}_v2_$part/README.md"
  log "done: $OUT_ROOT/${TASK}_v2_$part, $(wc -l < "$OUT_ROOT/${TASK}_v2_$part/meta/episodes.jsonl") episodes"
done
[ "${KEEP:-0}" = 1 ] || rm -rf "$BUILD"
