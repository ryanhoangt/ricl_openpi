"""Keep only the first 20 episodes per task, move the rest to a remaining folder."""

import shutil
from pathlib import Path


def split_episodes(
    input_dir: Path,
    remaining_dir: Path,
    keep: int = 20,
    dry_run: bool = False,
) -> None:
    task_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not task_dirs:
        print(f"No task folders found in {input_dir}")
        return

    for task_dir in task_dirs:
        episode_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("episode_"))
        to_keep = episode_dirs[:keep]
        to_move = episode_dirs[keep:]

        print(f"\n{task_dir.name}: keeping {len(to_keep)}, moving {len(to_move)}")
        for ep in to_move:
            dest = remaining_dir / task_dir.name / ep.name
            print(f"  MOVE {ep}  ->  {dest}")
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ep), str(dest))

    print("\nDone." if not dry_run else "\nDry run complete — nothing moved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("ricl_libero_preprocessing/collected_demos_training/libero_group"))
    parser.add_argument("--remaining-dir", type=Path, default=Path("ricl_libero_preprocessing/collected_demos_training_remaining/libero_group"))
    parser.add_argument("--keep", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_episodes(args.input_dir, args.remaining_dir, keep=args.keep, dry_run=args.dry_run)