"""Reorganize existing episode folders by task (prompt) group."""

import re
import shutil
from pathlib import Path


def sanitize_group_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "group"


def reorganize_by_task(input_dir: Path, dry_run: bool = False) -> None:
    import numpy as np

    episode_dirs = sorted(input_dir.glob("episode_*"))
    if not episode_dirs:
        print(f"No episode_* folders found in {input_dir}")
        return

    for episode_dir in episode_dirs:
        npz_path = episode_dir / "processed_demo.npz"
        if not npz_path.exists():
            print(f"Skipping {episode_dir} (no processed_demo.npz)")
            continue

        data = np.load(npz_path, allow_pickle=True)
        prompt = str(data["prompt"])
        group = sanitize_group_name(prompt or "group")

        dest_dir = input_dir / group / episode_dir.name

        print(f"{episode_dir.name}  ->  {group}/")
        if not dry_run:
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(episode_dir), str(dest_dir))

    print("Done." if not dry_run else "Dry run complete — nothing moved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("ricl_libero_preprocessing/collected_demos_training/libero_group"))
    parser.add_argument("--dry-run", action="store_true", help="Print moves without doing them")
    args = parser.parse_args()

    reorganize_by_task(args.input_dir, dry_run=args.dry_run)