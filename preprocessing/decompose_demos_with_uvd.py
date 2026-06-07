"""Run UVD subgoal decomposition over every processed_demo.npz under a root and save subgoals_{cam}.json."""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from uvd.decomp import decomp_trajectories
from uvd.models import get_preprocessor


CAMERA_KEYS = {"top": "top_image", "wrist": "wrist_image"}
PREPROCESSORS = ["vip", "r3m", "liv", "clip", "vc1", "dinov2"]


def find_episodes(root: Path) -> list[tuple[str, Path]]:
    eps = []
    for p in sorted(root.rglob("processed_demo.npz")):
        ep_dir = p.parent
        task = ep_dir.parent.name
        eps.append((task, ep_dir))
    return eps


def filter_episodes(eps, tasks=None, max_per_task=None, max_total=None):
    if tasks is not None:
        tasks_set = set(tasks)
        eps = [(t, d) for (t, d) in eps if t in tasks_set]
    if max_per_task is not None:
        counts: dict[str, int] = defaultdict(int)
        out = []
        for t, d in eps:
            if counts[t] < max_per_task:
                out.append((t, d))
                counts[t] += 1
        eps = out
    if max_total is not None:
        eps = eps[:max_total]
    return eps


def save_debug_png(out_path: Path, frames: np.ndarray, indices: list[int], title: str) -> None:
    import matplotlib.pyplot as plt

    K = len(indices)
    T = int(frames.shape[0])
    fig = plt.figure(figsize=(max(K, 1) * 2.4, 3.4))
    gs = fig.add_gridspec(2, max(K, 1), height_ratios=[5, 1], hspace=0.25)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[idx])
        ax.set_title(f"step {idx}", fontsize=9)
        ax.axis("off")

    ax_tl = fig.add_subplot(gs[1, :])
    ax_tl.barh([0], [T - 1], left=0, color="lightgray", height=0.5)
    for idx in indices:
        ax_tl.axvline(idx, color="red", linewidth=2)
    ax_tl.set_xlim(0, T - 1)
    ax_tl.set_ylim(-0.5, 0.5)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel(f"trajectory step  ({K} subgoals / {T} steps)")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def aggregate_summary(per_episode: list[dict]) -> dict:
    per_task: dict[str, list[int]] = defaultdict(list)
    for ep in per_episode:
        per_task[ep["task"]].append(ep["n_subgoals"])

    per_task_stats = {}
    for t, v in per_task.items():
        per_task_stats[t] = {
            "count": len(v),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "min": int(min(v)),
            "max": int(max(v)),
        }

    all_ks = [ep["n_subgoals"] for ep in per_episode]
    global_stats = {}
    if all_ks:
        global_stats = {
            "count": len(all_ks),
            "mean": float(np.mean(all_ks)),
            "median": float(np.median(all_ks)),
            "min": int(min(all_ks)),
            "max": int(max(all_ks)),
        }
    return {"per_task_n_subgoals": per_task_stats, "global_n_subgoals": global_stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="root containing <task>/<episode>/processed_demo.npz")
    parser.add_argument("--camera", choices=list(CAMERA_KEYS), default="top")
    parser.add_argument("--preprocessor", choices=PREPROCESSORS, default="dinov2")
    parser.add_argument("--tasks", type=str, default=None,
                        help="comma-separated list of task subfolder names; defaults to all")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--save-debug-png", action="store_true",
                        help="also write subgoals_{cam}_debug.png next to each json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary-file", type=Path, default=None,
                        help="defaults to <root>/subgoals_summary_{cam}_{preprocessor}.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("uvd_decompose")

    cam_key = CAMERA_KEYS[args.camera]
    out_name = f"subgoals_{args.camera}.json"

    all_eps = find_episodes(args.root)
    log.info(f"discovered {len(all_eps)} episodes under {args.root}")
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    eps = filter_episodes(all_eps, tasks=tasks,
                          max_per_task=args.max_episodes_per_task,
                          max_total=args.max_episodes)
    log.info(f"after filtering: {len(eps)} episodes to process")
    if not eps:
        log.warning("no episodes after filtering; exiting")
        return

    log.info(f"loading preprocessor='{args.preprocessor}' on device='{args.device}'")
    preprocessor = get_preprocessor(args.preprocessor, device=args.device)

    summary = {
        "preprocessor": args.preprocessor,
        "camera": args.camera,
        "root": str(args.root.resolve()),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "n_eligible": len(eps),
        "n_processed": 0,
        "n_skipped_existing": 0,
        "n_failed": 0,
        "per_episode": [],
    }

    t0 = time.time()
    for i, (task, ep_dir) in enumerate(eps):
        tag = f"[{i + 1}/{len(eps)}] {task}/{ep_dir.name}"
        out_path = ep_dir / out_name
        if out_path.exists() and not args.overwrite:
            log.info(f"{tag}: skip (exists)")
            summary["n_skipped_existing"] += 1
            continue

        try:
            with np.load(ep_dir / "processed_demo.npz", allow_pickle=True) as npz:
                if cam_key not in npz.files:
                    log.error(f"{tag}: missing key '{cam_key}', skipping")
                    summary["n_failed"] += 1
                    continue
                frames = npz[cam_key]
            T = int(frames.shape[0])

            rep = preprocessor.process(frames, return_numpy=True)
            _, decomp_meta = decomp_trajectories("embed", rep)
            indices = [int(x) for x in decomp_meta.milestone_indices]
        except Exception as e:
            log.error(f"{tag}: UVD failed: {e}")
            summary["n_failed"] += 1
            continue

        with open(out_path, "w") as f:
            json.dump({
                "subgoal_indices": indices,
                "preprocessor": args.preprocessor,
                "camera": args.camera,
                "n_steps": T,
            }, f, indent=2)

        if args.save_debug_png:
            try:
                save_debug_png(
                    ep_dir / f"subgoals_{args.camera}_debug.png",
                    frames, indices, title=f"{task} / {ep_dir.name}"
                )
            except Exception as e:
                log.warning(f"{tag}: debug png failed: {e}")

        summary["n_processed"] += 1
        summary["per_episode"].append({
            "task": task,
            "episode": ep_dir.name,
            "n_subgoals": len(indices),
            "n_steps": T,
            "subgoal_indices": indices,
        })

        elapsed = time.time() - t0
        log.info(f"{tag}: {len(indices)} subgoals / {T} steps  (elapsed {elapsed:.1f}s)")

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["duration_seconds"] = round(time.time() - t0, 2)
    summary.update(aggregate_summary(summary["per_episode"]))

    summary_path = (
        args.summary_file
        if args.summary_file is not None
        else args.root / f"subgoals_summary_{args.camera}_{args.preprocessor}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        f"done: processed={summary['n_processed']} "
        f"skipped={summary['n_skipped_existing']} failed={summary['n_failed']} "
        f"duration={summary['duration_seconds']}s"
    )
    log.info(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
