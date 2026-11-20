"""Assemble the v2 training (120 episodes) and test datasets from the renders.

Fills every training slot of <build>/plan/slots.json (written by plan.py): a
reused slot takes its earlier render; a new slot takes its new render or, if
that replay failed, a spare rendered for the same room and table. Then merges
the chosen episodes, and the test renders, with scripts/merge_psi0_datasets.py.

    python scripts/faucet_v2/assemble.py --build data/faucet_v2_build [--out-root DIR]
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = "G1WholebodyOpenFaucetTeleop-v0"
INSTRUCTION = "move forward to the faucet and open it"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def edit_key(episode: dict) -> str:
    """The table edit an episode was rendered with, which is unique per planned episode."""
    return json.dumps(json.loads(episode["environment_config"])["table_edit"], sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="data/faucet_v2_build", help="build directory, relative to the repo root")
    parser.add_argument("--out-root", default="data/processed_psi0", help="where the two datasets are written")
    args = parser.parse_args()
    plan_dir = ROOT / args.build / "plan"
    renders = f"{args.build}/psi0/{TASK}_v2_{{}}_renders"

    slots = json.loads((plan_dir / "slots.json").read_text())["slots"]
    plan = json.loads((plan_dir / "train_plan.json").read_text())
    planned = load_jsonl(ROOT / args.build / "configs/v2/train/level-0/meta/episodes.jsonl")  # plan job k -> episode k
    train_renders = renders.format("train")
    rendered = {edit_key(e): e["episode_index"] for e in load_jsonl(ROOT / train_renders / "meta/episodes.jsonl")}

    # new jobs that rendered, in plan order: (room, table) -> [(job, render episode)]
    done = {k: rendered.get(edit_key(e)) for k, e in enumerate(planned)}
    spares = {}
    for k, job in enumerate(plan):
        if job["spare"] and done[k] is not None:
            spares.setdefault((job["room"], job["variant"]), []).append(done[k])

    select, missing = {}, []
    new_jobs = iter(k for k, job in enumerate(plan) if not job["spare"])
    for slot in slots:
        if slot["source"] == "reuse":
            select.setdefault(slot["dataset"], []).append(slot["episode"])
            continue
        k = next(new_jobs)
        assert (plan[k]["room"], plan[k]["variant"]) == (slot["room"], slot["table"]), (k, slot)
        episode = done[k] if done[k] is not None else (spares.get((slot["room"], slot["table"])) or [None]).pop(0)
        if episode is None:
            missing.append(slot)
            continue
        select.setdefault(train_renders, []).append(episode)
    if missing:
        print(f"{len(missing)} slots without a render: {Counter((s['room'], s['table']) for s in missing)}")

    (plan_dir / "train_select.json").write_text(json.dumps(select, indent=1))
    for part, inputs, extra in (("train", list(select), ["--select", str(plan_dir / "train_select.json")]),
                                ("test", [renders.format("test")], [])):
        out = ROOT / args.out_root / f"{TASK}_v2_{part}"
        if out.exists():
            sys.exit(f"{out} exists; remove it first")
        subprocess.run([sys.executable, str(ROOT / "scripts/merge_psi0_datasets.py"), "--out", str(out),
                        "--task", INSTRUCTION, *extra, *inputs], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
