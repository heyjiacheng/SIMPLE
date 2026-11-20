"""Copy some episodes of a teleop_decoupled_wbc dataset (raw LeRobot v2.1 format) into a new dataset.

The output keeps the teleop format, so ``replay_decoupled_wbc`` (and the Isaac render) read it like the
original: episodes are renumbered 0..N-1 in order, the parquet ``episode_index``/``index`` columns and the
per-episode stats of those two columns follow, videos and every other column and episodes.jsonl key
(environment_config, hand_mode, vr_view, ...) are copied unchanged. ``meta/sources.json`` maps each new
episode to its source episode. The source dataset is only read.

Success dataset (teleop successes that also succeed in replay)::

    python scripts/subset_teleop_dataset.py --drop-failed \\
        --replay-results data/replay_decoupled_wbc_output/mujoco_replay/vr-vla.txt \\
        data/teleop_decoupled_wbc/vr-vla/simple/G1WholebodyUncapBottleTeleop-v0/level-0 \\
        data/teleop_decoupled_wbc/vr-vla-success/simple/G1WholebodyUncapBottleTeleop-v0/level-0

``--drop-failed`` leaves out the episodes meta/failed_episodes.json lists (teleop --keep-failures);
``--replay-results`` (lines ``<episode>: True|False``, a replay's meta/replay.txt) keeps only the True ones;
``--episodes`` keeps only the given ones. Kept failed episodes stay marked, under their new index.
``--replace SRC_EP=DATASET`` takes episode 0 of the teleop-format dataset DATASET (parquet, video, episodes.jsonl
entry, stats) in place of source episode SRC_EP, e.g. a re-rendered fix of a broken episode; sources.json records it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from simple.cli._decoupled_wbc_recording import load_failed_episodes, write_failed_episodes


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_replay_results(path: Path) -> dict[int, bool]:
    results = {}
    for line in open(path):
        if line.strip():
            idx, ok = line.split(":")
            results[int(idx)] = ok.strip() == "True"
    return results


def index_stats(values: np.ndarray, like: dict) -> dict:
    """Stats of an integer column in the per-episode format of ``like`` (lists of one value)."""
    return {**like, "min": [int(values.min())], "max": [int(values.max())], "mean": [float(values.mean())],
            "std": [float(values.std())], "count": [int(values.size)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="teleop dataset dir (the level-0 folder)")
    parser.add_argument("out", type=Path, help="output dataset dir (must not exist)")
    parser.add_argument("--drop-failed", action="store_true", help="leave out the episodes failed_episodes.json lists")
    parser.add_argument("--replay-results", type=Path, help="keep only the episodes with True in this file")
    parser.add_argument("--episodes", type=int, nargs="+", help="keep only these source episodes")
    parser.add_argument("--replace", action="append", default=[], metavar="SRC_EP=DATASET",
                        help="use episode 0 of DATASET in place of source episode SRC_EP (repeatable)")
    args = parser.parse_args()
    src, out = args.src, args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it first")

    info = json.loads((src / "meta" / "info.json").read_text())
    episodes = sorted(load_jsonl(src / "meta" / "episodes.jsonl"), key=lambda e: e["episode_index"])
    stats = {row["episode_index"]: row for row in load_jsonl(src / "meta" / "episodes_stats.jsonl")}
    failed = load_failed_episodes(src)
    total = len(episodes)
    if [e["episode_index"] for e in episodes] != list(range(total)) or info["total_episodes"] != total:
        raise ValueError(f"{src}: episodes.jsonl and info.json disagree")

    keep = set(range(total))
    reasons = {}
    if args.drop_failed:
        keep -= set(failed)
        reasons.update({i: f"failed at teleop ({failed[i]})" for i in failed})
    if args.replay_results:
        results = read_replay_results(args.replay_results)
        missing = sorted(set(range(total)) - set(results))
        if missing:
            raise ValueError(f"{args.replay_results} has no result for episodes {missing}")
        lost = {i for i, ok in results.items() if not ok}
        reasons.update({i: "fails in replay" for i in lost if i in keep})
        keep -= lost
    if args.episodes is not None:
        reasons.update({i: "not in --episodes" for i in keep - set(args.episodes)})
        keep &= set(args.episodes)
    kept = sorted(keep)
    if not kept:
        raise ValueError("no episodes left")

    replace = {}
    for item in args.replace:
        ep_text, path = item.split("=", 1)
        r_dir = Path(path)
        r_info = json.loads((r_dir / "meta" / "info.json").read_text())
        if r_info["features"] != info["features"]:
            raise ValueError(f"{r_dir}: features differ from {src}")
        replace[int(ep_text)] = (r_dir, r_info, load_jsonl(r_dir / "meta" / "episodes.jsonl")[0],
                                 load_jsonl(r_dir / "meta" / "episodes_stats.jsonl")[0])
    unknown = set(replace) - set(range(total))
    if unknown:
        raise ValueError(f"--replace names episodes {sorted(unknown)} that {src} does not have")

    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    chunks = info["chunks_size"]
    new_episodes, new_stats, new_failed, sources = [], [], {}, []
    frame = 0
    for new, old in enumerate(kept):
        ep, ep_dir, ep_info, ep_old, ep_stats = episodes[old], src, info, old, stats[old]
        if old in replace:
            ep_dir, ep_info, ep, ep_stats = replace[old]
            ep_old = ep["episode_index"]
        table = pq.read_table(ep_dir / ep_info["data_path"].format(episode_chunk=ep_old // ep_info["chunks_size"],
                                                                   episode_index=ep_old))
        n = table.num_rows
        if n != ep["length"]:
            raise ValueError(f"{ep_dir}: episode {ep_old} has {n} rows, episodes.jsonl says {ep['length']}")
        episode_col = np.full(n, new, dtype=np.int64)
        index_col = np.arange(frame, frame + n, dtype=np.int64)
        for name, values in (("episode_index", episode_col), ("index", index_col)):
            i = table.schema.get_field_index(name)
            table = table.set_column(i, table.schema.field(i), pa.array(values, type=table.schema.field(i).type))
        table = table.replace_schema_metadata(table.schema.metadata)
        dst = out / info["data_path"].format(episode_chunk=new // chunks, episode_index=new)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst)
        for key in video_keys:
            v_src = ep_dir / ep_info["video_path"].format(episode_chunk=ep_old // ep_info["chunks_size"], video_key=key,
                                                          episode_index=ep_old)
            v_dst = out / info["video_path"].format(episode_chunk=new // chunks, video_key=key, episode_index=new)
            v_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(v_src, v_dst)

        new_episodes.append({**ep, "episode_index": new})
        st = json.loads(json.dumps(ep_stats))
        st["episode_index"] = new
        st["stats"]["episode_index"] = index_stats(episode_col, st["stats"]["episode_index"])
        st["stats"]["index"] = index_stats(index_col, st["stats"]["index"])
        new_stats.append(st)
        if old in failed:
            new_failed[new] = failed[old]
        sources.append({"episode_index": new, "source_episode_index": old,
                        **({"replaced_by": str(ep_dir.resolve())} if old in replace else {})})
        frame += n

    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    write_jsonl(meta / "episodes.jsonl", new_episodes)
    write_jsonl(meta / "episodes_stats.jsonl", new_stats)
    for name in ("tasks.jsonl", "modality.json"):
        if (src / "meta" / name).exists():
            shutil.copy2(src / "meta" / name, meta / name)
    n_eps = len(kept)
    new_info = {**info, "total_episodes": n_eps, "total_frames": frame, "total_videos": n_eps * len(video_keys),
                "total_chunks": (n_eps + chunks - 1) // chunks, "splits": {"train": f"0:{n_eps}"}}
    (meta / "info.json").write_text(json.dumps(new_info, indent=4))
    write_failed_episodes(out, new_failed)  # no file without any
    left_out = {str(i): reasons.get(i, "?") for i in sorted(set(range(total)) - keep)}
    (meta / "sources.json").write_text(json.dumps(
        {"source": str(src.resolve()), "drop_failed": args.drop_failed,
         "replay_results": str(args.replay_results.resolve()) if args.replay_results else None,
         "left_out": left_out, "episodes": sources}, indent=1))
    print(f"{src}: {total} episodes -> {n_eps} ({frame} frames) in {out}"
          + (f", {len(new_failed)} kept marked failed" if new_failed else "")
          + (f", episodes {sorted(replace)} replaced" if replace else "")
          + f"; left out {len(left_out)}: {left_out}")


if __name__ == "__main__":
    main()
