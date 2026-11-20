# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Merge Psi-0 format datasets (as written by postprocess_psi0_sonic.py) into one.

Episodes are re-indexed in input order, the parquet index columns rewritten,
videos copied, and the dataset statistics recomputed over all frames the same
way postprocess_psi0_sonic.py computes them. Each episode records where it came
from under ``source`` in meta/episodes.jsonl, and meta/sources.json lists the
episode range of every input.

Pass ``--task`` to give every episode the same instruction, e.g. when the
inputs were rendered with different task texts for the same task. Pass
``--select`` (JSON ``{"<input dir as given>": [episode indices]}``) to take only
some episodes of an input; inputs it does not list are taken whole.

    python scripts/merge_psi0_datasets.py --out data/processed_psi0/merged \\
        --task "move forward to the faucet and open it" \\
        data/psi-data/simple/G1WholebodyOpenFaucetTeleop-v0 \\
        data/processed_psi0/G1WholebodyOpenFaucetTeleop-v0_round_big

Episodes teleop_decoupled_wbc --keep-failures saved as failed are listed in an
input's meta/failed_episodes.json (postprocess_psi0_sonic.py carries the list
over). ``--drop-failed`` leaves them out: with one input, this makes the
training set of a dataset kept whole as the archive,

    python scripts/merge_psi0_datasets.py --drop-failed \\
        --out data/processed_psi0/uncap-vla-train data/processed_psi0/uncap-vla

Without it they are copied, and listed in the output's meta/failed_episodes.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from simple.cli._decoupled_wbc_recording import load_failed_episodes, write_failed_episodes

STATS_COLUMNS = ("states", "action", "timestamp", "frame_index", "episode_index", "index", "task_index", "next.done")


def _load_postprocess():
    spec = importlib.util.spec_from_file_location("postprocess_psi0_sonic", Path(__file__).with_name("postprocess_psi0_sonic.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def table_variant(episode: dict) -> str:
    """Table type of an episode, from the scene config written by make_round_table_dataset.py."""
    env_conf = episode.get("environment_config")
    if not env_conf:
        return "unknown"
    env_conf = json.loads(env_conf) if isinstance(env_conf, str) else env_conf
    return env_conf.get("table_edit", {}).get("variant", "box")


class DatasetWriter:
    """Writes episodes copied from Psi-0 format datasets as a new dataset.

    Episodes are re-indexed in the order they are added; ``close`` writes the
    meta files, with the statistics recomputed over all frames.
    """

    def __init__(self, out: Path, info: dict):
        if out.exists():
            raise FileExistsError(f"{out} exists; remove it first")
        (out / "meta").mkdir(parents=True)
        self.out, self.info = out, info
        self.video_key = next(k for k, v in info["features"].items() if v["dtype"] == "video").removeprefix("observation.images.")
        self.tasks: list[str] = []
        self.columns = {name: [] for name in STATS_COLUMNS}
        self.episodes: list[dict] = []
        self.episode_stats: list[dict] = []
        self.total_frames = 0
        self.failed: dict[int, str] = {}  # {new episode index: reason}: the failed episodes copied

    def add(self, path: Path, info: dict, ep: dict, ep_stats: dict | None, task: str, **fields) -> None:
        """Copy episode ``ep`` of the dataset at ``path`` (whose meta/info.json is ``info``).

        ``fields`` are added to the episode's meta/episodes.jsonl entry.
        """
        old, new = ep["episode_index"], len(self.episodes)
        old_chunk, new_chunk = old // info["chunks_size"], new // self.info["chunks_size"]
        if task not in self.tasks:
            self.tasks.append(task)
        task_index = self.tasks.index(task)

        table = pq.read_table(path / info["data_path"].format(episode_chunk=old_chunk, episode_index=old))
        n = table.num_rows
        replace = {
            "episode_index": pa.array(np.full(n, new, dtype=np.int64)),
            "index": pa.array(np.arange(self.total_frames, self.total_frames + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
        }
        for name, values in replace.items():
            table = table.set_column(table.schema.get_field_index(name), name, values)
        dst = self.out / self.info["data_path"].format(episode_chunk=new_chunk, episode_index=new)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst)

        video = self.info["video_path"].format(episode_chunk=new_chunk, video_key=self.video_key, episode_index=new)
        (self.out / video).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path / info["video_path"].format(episode_chunk=old_chunk, video_key=self.video_key, episode_index=old), self.out / video)

        for name in self.columns:
            values = table[name].to_pylist()
            self.columns[name].append(np.asarray(values, dtype=np.float32))

        self.episodes.append({**ep, "episode_index": new, "tasks": [task_index],
                              "dataset_from_index": self.total_frames, "dataset_to_index": self.total_frames + n - 1,
                              "instruction": {"task_index": task_index, "task": task}, **fields})
        if ep_stats is not None:
            self.episode_stats.append({**ep_stats, "episode_index": new})
        self.total_frames += n

    def close(self, modality: Path) -> None:
        """Write the meta files; ``modality`` is the meta/modality.json to copy."""
        meta = self.out / "meta"
        write_jsonl(meta / "episodes.jsonl", self.episodes)
        write_jsonl(meta / "episodes_stats.jsonl", self.episode_stats)
        write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": t, "category": "", "description": t} for i, t in enumerate(self.tasks)])
        info = {**self.info, "total_episodes": len(self.episodes), "total_frames": self.total_frames, "total_tasks": len(self.tasks),
                "total_videos": len(self.episodes), "total_chunks": math.ceil(len(self.episodes) / self.info["chunks_size"])}
        (meta / "info.json").write_text(json.dumps(info, indent=4))
        stats_block = _load_postprocess().stats_block
        stats = {name: stats_block(np.concatenate(values)) for name, values in self.columns.items()}
        (meta / "stats.json").write_text(json.dumps(stats, indent=4))
        (meta / "stats_psi0.json").write_text(json.dumps(stats, indent=4))
        (meta / "relative_stats.json").write_text("{}")
        (meta / "lang_map.json").write_text("{}")
        shutil.copy2(modality, meta / "modality.json")
        write_failed_episodes(self.out, self.failed)  # no file without any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Psi-0 format dataset dirs, merged in this order")
    parser.add_argument("--out", required=True, help="output dataset dir (must not exist)")
    parser.add_argument("--task", default=None, help="instruction for every episode (default: keep each input's)")
    parser.add_argument("--select", default=None, help='JSON {"<input dir>": [episode indices]}: only these episodes of that input')
    parser.add_argument("--drop-failed", action="store_true",
                        help="leave out the episodes an input's meta/failed_episodes.json lists (teleop --keep-failures)")
    args = parser.parse_args()

    select = json.loads(Path(args.select).read_text()) if args.select else {}
    unknown = set(select) - set(args.inputs)
    if unknown:
        raise ValueError(f"--select names datasets that are not inputs: {sorted(unknown)}")
    inputs = [Path(p) for p in args.inputs]
    infos = [json.loads((p / "meta" / "info.json").read_text()) for p in inputs]
    for path, info in zip(inputs[1:], infos[1:]):
        for key in ("features", "fps", "data_path", "video_path"):
            if info[key] != infos[0][key]:
                raise ValueError(f"{path}: {key} differs from {inputs[0]}")

    writer = DatasetWriter(Path(args.out), infos[0])
    sources = []
    for name, path, info in zip(args.inputs, inputs, infos):
        src_tasks = {row["task_index"]: row["task"] for row in load_jsonl(path / "meta" / "tasks.jsonl")}
        src_stats = {row["episode_index"]: row for row in load_jsonl(path / "meta" / "episodes_stats.jsonl")}
        episodes = sorted(load_jsonl(path / "meta" / "episodes.jsonl"), key=lambda e: e["episode_index"])
        if name in select:
            keep = set(select[name])
            missing = keep - {ep["episode_index"] for ep in episodes}
            if missing:
                raise ValueError(f"{path} has no episodes {sorted(missing)}")
            episodes = [ep for ep in episodes if ep["episode_index"] in keep]
        failed = load_failed_episodes(path)
        dropped = []
        if args.drop_failed:
            dropped = [ep["episode_index"] for ep in episodes if ep["episode_index"] in failed]
            episodes = [ep for ep in episodes if ep["episode_index"] not in failed]
        first = len(writer.episodes)
        for ep in episodes:
            task = args.task if args.task is not None else src_tasks[ep["tasks"][0]]
            writer.add(path, info, ep, src_stats.get(ep["episode_index"]), task,
                       source={"dataset": str(path), "episode_index": ep["episode_index"]})
            if ep["episode_index"] in failed:
                writer.failed[len(writer.episodes) - 1] = failed[ep["episode_index"]]
        sources.append({"dataset": str(path), "episodes": [first, len(writer.episodes) - 1],
                        "tables": dict(Counter(table_variant(e) for e in writer.episodes[first:]))})
        if args.drop_failed:
            sources[-1]["dropped_failed_episodes"] = dropped
        print(f"{path}: {len(writer.episodes) - first} episodes -> {first}..{len(writer.episodes) - 1}"
              + (f", {len(dropped)} failed ones left out: {dropped}" if args.drop_failed else ""))

    if not writer.episodes:
        raise ValueError(f"no episodes to write (all left out?); remove {writer.out}")
    writer.close(inputs[0] / "meta" / "modality.json")
    (writer.out / "meta" / "sources.json").write_text(json.dumps(sources, indent=2))
    print(f"wrote {len(writer.episodes)} episodes, {writer.total_frames} frames, tasks {writer.tasks} to {writer.out}")


if __name__ == "__main__":
    main()
