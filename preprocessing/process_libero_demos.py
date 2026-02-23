"""Prepare LIBERO LeRobot episodes for RICL priming.

This script converts a LeRobot-format LIBERO dataset into per-episode
`processed_demo.npz` files containing resized images, embeddings, and
per-step state/action arrays.

Example command: 
```
HF_HOME="/mnt/data/vhoangth2/hf_cache" HF_DATASETS_CACHE="/mnt/data/vhoangth2/hf_cache/datasets" python preprocessing/process_libero_demos.py   --repo-id ryanhoangt/libero-icl-priming   --output-dir ricl_libero_preprocessing/collected_demos_training --group-by-task
```
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi.policies.utils import embed_with_batches
from openpi.policies.utils import load_dinov2
from openpi_client.image_tools import resize_with_pad

logger = logging.getLogger(__name__)


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_uint8_hwc(images: object) -> np.ndarray:
    arr = _to_numpy(images)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _parse_episode_list(raw: str | None, total_episodes: int) -> list[int]:
    if raw is None:
        return list(range(total_episodes))

    raw = raw.strip()
    if not raw:
        return list(range(total_episodes))

    if ":" in raw:
        start_str, end_str = raw.split(":", maxsplit=1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else total_episodes
        return list(range(start, min(end, total_episodes)))

    if "-" in raw:
        start_str, end_str = raw.split("-", maxsplit=1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else total_episodes
        return list(range(start, min(end + 1, total_episodes)))

    return [int(item) for item in raw.split(",") if item.strip()]


def _sanitize_group_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "group"


def _episode_prompt(meta: LeRobotDatasetMetadata) -> dict[int, str]:
    prompts: dict[int, str] = {}
    for episode in meta.episodes:
        if not episode.get("tasks"):
            continue
        prompts[episode["episode_index"]] = episode["tasks"][0]
    return prompts


def _iter_episode_indices(episodes: Iterable[int], max_episodes: int | None) -> Iterable[int]:
    for count, episode_index in enumerate(episodes):
        if max_episodes is not None and count >= max_episodes:
            return
        yield episode_index


def process_episodes(
    *,
    repo_id: str,
    root: Path | None,
    output_dir: Path,
    group_name: str,
    group_by_task: bool,
    episodes: list[int],
    max_episodes: int | None,
    max_episode_per_task: int | None,
    resize: int,
    image_key: str,
    wrist_key: str,
    state_key: str,
    action_key: str,
    local_files_only: bool,
    overwrite: bool,
) -> None:
    metadata = LeRobotDatasetMetadata(repo_id, root=root, local_files_only=local_files_only)
    episode_prompts = _episode_prompt(metadata)

    # -------------------------------------------------------
    # NEW FEATURE: Limit number of episodes per task
    # -------------------------------------------------------
    if group_by_task and max_episode_per_task is not None:
        task_to_episodes: dict[str, list[int]] = {}
        for ep in episodes:
            task_name = episode_prompts.get(ep, "")
            task_name = _sanitize_group_name(task_name or "group")
            task_to_episodes.setdefault(task_name, []).append(ep)

        limited = []
        for task, eps in task_to_episodes.items():
            limited.extend(eps[:max_episode_per_task])
        episodes = limited
        logger.info("Applying per-task episode limit: %d episodes per task", max_episode_per_task)

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        local_files_only=local_files_only,
    )

    selected_lookup = {episode_index: idx for idx, episode_index in enumerate(dataset.episodes)}
    dinov2 = load_dinov2()

    output_dir.mkdir(parents=True, exist_ok=True)
    if group_name:
        output_dir = output_dir / group_name
        output_dir.mkdir(parents=True, exist_ok=True)

    for episode_index in _iter_episode_indices(episodes, max_episodes):
        selected_index = selected_lookup[episode_index]
        start = dataset.episode_data_index["from"][selected_index].item()
        end = dataset.episode_data_index["to"][selected_index].item()
        prompt = episode_prompts.get(episode_index, "")

        group_dir = output_dir
        if group_by_task:
            group_dir = output_dir / _sanitize_group_name(prompt or "group")
            group_dir.mkdir(parents=True, exist_ok=True)

        episode_dir = group_dir / f"episode_{episode_index:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        processed_path = episode_dir / "processed_demo.npz"
        if processed_path.exists() and not overwrite:
            logger.info("Skipping %s (already processed)", episode_dir)
            continue

        top_images: list[np.ndarray] = []
        wrist_images: list[np.ndarray] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []

        for idx in range(start, end):
            item = dataset[idx]
            top_images.append(_ensure_uint8_hwc(item[image_key]))
            wrist_images.append(_ensure_uint8_hwc(item[wrist_key]))
            states.append(_to_numpy(item[state_key]))
            actions.append(_to_numpy(item[action_key]))

        top_images_array = resize_with_pad(np.stack(top_images, axis=0), resize, resize)
        wrist_images_array = resize_with_pad(np.stack(wrist_images, axis=0), resize, resize)
        state_array = np.stack(states, axis=0)
        action_array = np.stack(actions, axis=0)

        top_embeddings = embed_with_batches(top_images_array, dinov2)
        wrist_embeddings = embed_with_batches(wrist_images_array, dinov2)

        np.savez(
            processed_path,
            state=state_array,
            actions=action_array,
            top_image=top_images_array,
            wrist_image=wrist_images_array,
            top_image_embeddings=top_embeddings,
            wrist_image_embeddings=wrist_embeddings,
            prompt=prompt,
        )
        logger.info("Saved %s", processed_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="physical-intelligence/libero")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("ricl_libero_preprocessing/collected_demos_training"))
    parser.add_argument("--group-name", type=str, default="libero_group")
    parser.add_argument("--group-by-task", action="store_true")
    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episode-per-task", type=int, default=None)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--wrist-key", type=str, default="wrist_image")
    parser.add_argument("--state-key", type=str, default="state")
    parser.add_argument("--action-key", type=str, default="actions")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root, local_files_only=args.local_files_only)
    episodes = _parse_episode_list(args.episodes, metadata.total_episodes)

    process_episodes(
        repo_id=args.repo_id,
        root=args.root,
        output_dir=args.output_dir,
        group_name=args.group_name,
        group_by_task=args.group_by_task,
        episodes=episodes,
        max_episodes=args.max_episodes,
        max_episode_per_task=args.max_episode_per_task,  # Pass new arg
        resize=args.resize,
        image_key=args.image_key,
        wrist_key=args.wrist_key,
        state_key=args.state_key,
        action_key=args.action_key,
        local_files_only=args.local_files_only,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()