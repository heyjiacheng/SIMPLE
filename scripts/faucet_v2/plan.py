"""Plan the table-invariance training set (120 episodes) and its held-out test set.

Training: 40 big round, 40 small round and 40 rectangular tables over the 10
training rooms. Each of the 5 rooms that fit a big round table gets 8 big,
2 small and 2 rect; each of the 5 tighter rooms gets 6 small and 6 rect. Big and
small round episodes are reused from the earlier renders where one exists for
that room; the rest is rendered anew. Recordings are drawn least-used first, so
the set spans as many recordings (faucet placements) as possible.

Test: 15 recordings that no training episode uses, spread over the faucet's
sideways offset, each with a big and a small triangle table, in rooms that no
training episode uses.

Reads the earlier renders from the build directory (<build>/configs and <build>/psi0)
and writes <build>/plan/{train_plan,test_plan,slots}.json: plans for
make_round_table_dataset.py --plan, and every training slot with its source.

    python scripts/faucet_v2/plan.py --build data/faucet_v2_build
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = "data/psi-data/simple-teleop/G1WholebodyOpenFaucetTeleop-v0/{}/level-0"
SETS = ["G1WholebodyOpenFaucetTeleop-v0", "G1WholebodyOpenFaucetTeleop2-v0", "G1WholebodyOpenFaucetTeleop3-v0"]
TEST_RECORDINGS = 15
SPARES_PER_ROOM = 1  # extra new renders per room, in case a replay fails
SEED = 0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def pose_key(env_conf: dict) -> tuple:
    spatial = env_conf["dr_state_dict"]["spatial"]
    return tuple(round(v, 6) for name in ("g1_sonic", "articulated_4")
                 for v in spatial[name]["position"] + spatial[name]["quaternion"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="data/faucet_v2_build", help="build directory, relative to the repo root")
    build = parser.parse_args().build
    psi = f"{build}/psi0/G1WholebodyOpenFaucetTeleop-v0_round_{{}}"
    out = ROOT / build / "plan"

    rng = np.random.default_rng(SEED)
    rooms_train = json.loads((HERE / "rooms_train.json").read_text())
    rooms_test = json.loads((HERE / "rooms_test.json").read_text())

    # recordings (set, episode) and the faucet's sideways offset from the robot
    recording_of, sideways = {}, {}
    for s in SETS:
        for e in load_jsonl(ROOT / RAW.format(s) / "meta/episodes.jsonl"):
            conf = json.loads(e["environment_config"])
            rec = (s, e["episode_index"])
            recording_of[pose_key(conf)] = rec
            sp = conf["dr_state_dict"]["spatial"]
            sideways[rec] = sp["articulated_4"]["position"][1] - sp["g1_sonic"]["position"][1]

    # earlier replays: with which tables each recording opened the faucet
    succeeded = defaultdict(set)
    for variant in ("big", "small", "random_room"):
        for s in SETS:
            for line in open(ROOT / f"{build}/configs/{variant}/{s}/level-0/meta/replay.txt"):
                idx, ok = line.strip().split(": ")
                if ok == "True":
                    succeeded[(s, int(idx))].add(variant)

    # earlier renders in Psi-0 format
    renders = []
    for part in ("big", "small", "random_room"):
        dataset = psi.format(part)
        for e in load_jsonl(ROOT / dataset / "meta/episodes.jsonl"):
            conf = json.loads(e["environment_config"])
            edit = conf.get("table_edit", {})
            renders.append(dict(dataset=dataset, episode=e["episode_index"], recording=recording_of[pose_key(conf)],
                                table=edit.get("variant", "box"), room=edit.get("room", "scene2")))

    # test recordings: ones that opened the faucet with both round tables, one per bin of sideways offset
    robust = sorted((rec for rec, ok in succeeded.items() if {"big", "small"} <= ok), key=lambda rec: sideways[rec])
    test = [robust[rng.choice(b)] for b in np.array_split(np.arange(len(robust)), TEST_RECORDINGS)]
    train_recordings = [rec for rec in sorted(succeeded) if rec not in test]

    uses = Counter()

    def pick(candidates: list[dict], n: int) -> list[dict]:
        """n candidates, least-used recordings first (random among ties)."""
        order = sorted(rng.permutation(len(candidates)), key=lambda i: uses[candidates[i]["recording"]])
        chosen = [candidates[i] for i in order[:n]]
        uses.update(c["recording"] for c in chosen)
        return chosen

    # training slots per room; reuse earlier renders, scarce (room, table) pairs first
    slots = []
    for room, spec in rooms_train.items():
        counts = {"big": 8, "small": 2, "rect": 2} if "big" in spec["tables"] else {"small": 6, "rect": 6}
        slots += [dict(room=room, table=table) for table, n in counts.items() for _ in range(n)]
    groups = defaultdict(list)
    for slot in slots:
        if slot["table"] != "rect":
            groups[(slot["room"], slot["table"])].append(slot)
    available = {key: [r for r in renders if (r["room"], r["table"]) == key and r["recording"] not in test] for key in groups}
    for key in sorted(groups, key=lambda key: len(available[key])):
        for slot, render in zip(groups[key], pick(available[key], len(groups[key]))):
            slot.update(source="reuse", recording=render["recording"], dataset=render["dataset"], episode=render["episode"])

    # new renders for the remaining slots, plus spares; a round table only for recordings that
    # opened the faucet with that table before, a rectangle for ones that did with two tables
    new = [slot for slot in slots if "source" not in slot]
    for room in rooms_train:
        tables = [slot["table"] for slot in new if slot["room"] == room]
        if tables:
            new += [dict(room=room, table=max(set(tables), key=tables.count), spare=True) for _ in range(SPARES_PER_ROOM)]
    for job in new:
        fits = [dict(recording=rec) for rec in train_recordings
                if (len(succeeded[rec] - {"random_room"}) == 2 if job["table"] == "rect" else job["table"] in succeeded[rec])]
        job.update(source="new", recording=pick(fits, 1)[0]["recording"])
    train_plan = [dict(src=RAW.format(job["recording"][0]), episode_index=job["recording"][1], variant=job["table"],
                       room=job["room"], spare=job.get("spare", False)) for job in new]

    # test: each recording with a big and a small triangle, in two different rooms; every room
    # gets both triangle sizes
    rooms = [str(r) for r in rng.permutation(sorted(rooms_test))]
    test_plan = [dict(src=RAW.format(rec[0]), episode_index=rec[1], variant=variant,
                      room=rooms[(i + j * (len(rooms) // 2)) % len(rooms)])
                 for i, rec in enumerate(test) for j, variant in enumerate(("triangle", "small_triangle"))]

    out.mkdir(parents=True, exist_ok=True)
    (out / "train_plan.json").write_text(json.dumps(train_plan, indent=1))
    (out / "test_plan.json").write_text(json.dumps(test_plan, indent=1))
    (out / "slots.json").write_text(json.dumps(dict(test_recordings=test, slots=slots), indent=1))

    distinct = {slot["recording"] for slot in slots}
    print(f"test: {len(test)} recordings, sideways {min(sideways[r] for r in test):+.3f}..{max(sideways[r] for r in test):+.3f} m, "
          f"{len(test_plan)} episodes in {len(rooms)} rooms")
    print(f"training: {len(slots)} slots, {sum(s['source'] == 'reuse' for s in slots)} reused, "
          f"{len(train_plan)} new renders ({sum(j['spare'] for j in train_plan)} spares); "
          f"{len(distinct)} distinct recordings of {len(train_recordings)} usable")
    print("slots:", dict(sorted(Counter((s["room"], s["table"], s["source"]) for s in slots).items())))
    print("recording uses (incl. spares):", dict(sorted(Counter(uses.values()).items())))
    print("sideways coverage of training recordings:", np.round(np.percentile([sideways[r] for r in distinct], [0, 25, 50, 75, 100]), 3))


if __name__ == "__main__":
    main()
