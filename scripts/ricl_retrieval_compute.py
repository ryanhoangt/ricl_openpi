"""Precompute RICL retrieval results for one query episode against a support index."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from autofaiss import build_index


MODALITY_KEYS = {
    "top": ["top_image_embeddings"],
    "wrist": ["wrist_image_embeddings"],
    "both": ["top_image_embeddings", "wrist_image_embeddings"],
}


def find_episodes(root: Path) -> list[tuple[str, Path]]:
    return [
        (str(p.parent.relative_to(root)), p)
        for p in sorted(root.rglob("processed_demo.npz"))
    ]


def load_modality(npz, modality: str) -> np.ndarray:
    return np.concatenate([npz[k] for k in MODALITY_KEYS[modality]], axis=1).astype(np.float32)


def build_support(index_dir: Path, modality: str, exclude_rel: str | None):
    episodes = find_episodes(index_dir)
    if exclude_rel is not None:
        episodes = [(rel, p) for rel, p in episodes if rel != exclude_rel]
    if not episodes:
        raise RuntimeError(f"no episodes found under {index_dir} (after exclusion)")

    rel_names = [rel for rel, _ in episodes]
    npzs = [np.load(p, allow_pickle=True) for _, p in episodes]
    embeddings = np.concatenate([load_modality(n, modality) for n in npzs], axis=0)
    flat_to_ep_step = np.array(
        [(ep_idx, step) for ep_idx, n in enumerate(npzs) for step in range(n["state"].shape[0])]
    )
    return rel_names, npzs, embeddings, flat_to_ep_step


def save_summary_plots(out: Path, distances, retrieved_pairs, rel_names, support_npzs, query_n_steps, max_dist=None):
    import matplotlib.pyplot as plt

    n_steps, k = distances.shape
    top1_ep = retrieved_pairs[:, 0, 0]
    top1_step = retrieved_pairs[:, 0, 1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    for ki in range(k):
        ax.plot(distances[:, ki], label=f"top-{ki + 1}", alpha=0.7)
    if max_dist is not None:
        ax.axhline(max_dist, color="red", linestyle="--", alpha=0.6, label=f"max_dist={max_dist:.1f}")
    ax.set_xlabel("query step")
    ax.set_ylabel("L2 distance (raw)")
    ax.set_title("retrieval distance over query trajectory")
    ax.legend()

    ax = axes[0, 1]
    counts = np.bincount(top1_ep, minlength=len(rel_names))
    ax.bar(range(len(rel_names)), counts)
    ax.set_xlabel("support episode index")
    ax.set_ylabel("times retrieved (top-1)")
    ax.set_title(f"episode coverage ({(counts > 0).sum()}/{len(rel_names)} touched)")

    ax = axes[1, 0]
    norm_query = np.arange(n_steps) / max(n_steps - 1, 1)
    norm_retrieved = np.array(
        [
            top1_step[q] / max(support_npzs[top1_ep[q]]["state"].shape[0] - 1, 1)
            for q in range(n_steps)
        ]
    )
    sc = ax.scatter(norm_query, norm_retrieved, c=top1_ep, cmap="tab20", s=12)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("query phase (normalized)")
    ax.set_ylabel("retrieved phase (normalized)")
    ax.set_title("temporal alignment (top-1, color = source episode)")
    plt.colorbar(sc, ax=ax, label="source ep idx")

    ax = axes[1, 1]
    ax.hist(distances[:, 0], bins=30, alpha=0.7, label="top-1")
    if k > 1:
        ax.hist(distances[:, -1], bins=30, alpha=0.7, label=f"top-{k}")
    if max_dist is not None:
        ax.axvline(max_dist, color="red", linestyle="--", alpha=0.6, label=f"max_dist={max_dist:.1f}")
    ax.set_xlabel("L2 distance (raw)")
    ax.set_ylabel("count")
    ax.set_title("distance distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out / "summary_plots.png", dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query-episode", type=str, required=True,
                        help="relative path under --query-dir (or --index-dir if not set), e.g. 'episode_000253'")
    parser.add_argument("--query-dir", type=Path, default=None,
                        help="if set, query is taken from here; otherwise leave-one-out within --index-dir")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--query-modality", choices=list(MODALITY_KEYS), default="top")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance-file", type=Path, default=Path("assets/max_distance.json"),
                        help="json with {'distances': {'max': float}} for the max_dist reference line")
    args = parser.parse_args()

    same_folder = args.query_dir is None or args.query_dir.resolve() == args.index_dir.resolve()
    exclude = args.query_episode if same_folder else None

    rel_names, support_npzs, all_emb, flat_to_ep_step = build_support(
        args.index_dir, args.query_modality, exclude
    )

    knn_index, _ = build_index(
        embeddings=all_emb,
        save_on_disk=False,
        min_nearest_neighbors_to_retrieve=args.knn_k + 5,
        max_index_query_time_ms=10,
        max_index_memory_usage="25G",
        current_memory_available="50G",
        metric_type="l2",
        nb_cores=8,
    )

    query_root = args.query_dir if args.query_dir else args.index_dir
    query_npz_path = query_root / args.query_episode / "processed_demo.npz"
    if not query_npz_path.exists():
        raise FileNotFoundError(query_npz_path)
    query_npz = np.load(query_npz_path, allow_pickle=True)
    query_emb = load_modality(query_npz, args.query_modality)
    n_steps = query_emb.shape[0]

    sq_distances, flat_indices = knn_index.search(query_emb, args.knn_k)
    distances = np.sqrt(np.clip(sq_distances, 0.0, None))
    retrieved_pairs = flat_to_ep_step[flat_indices]

    max_dist = None
    if args.max_distance_file is not None and args.max_distance_file.exists():
        import json
        max_dist = float(json.load(open(args.max_distance_file))["distances"]["max"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "retrievals.npz",
        distances=distances.astype(np.float32),
        retrieved_ep_local_idx=retrieved_pairs[..., 0].astype(np.int32),
        retrieved_step_idx=retrieved_pairs[..., 1].astype(np.int32),
        rel_names=np.array(rel_names),
        query_episode=args.query_episode,
        query_dir=str(query_root.resolve()),
        index_dir=str(args.index_dir.resolve()),
        query_modality=args.query_modality,
        knn_k=args.knn_k,
        query_n_steps=n_steps,
    )

    save_summary_plots(args.output_dir, distances, retrieved_pairs, rel_names, support_npzs, n_steps, max_dist=max_dist)

    print(f"query: {args.query_episode} ({n_steps} steps, modality={args.query_modality})")
    print(f"index: {len(rel_names)} support episodes, {all_emb.shape[0]} steps total")
    print(f"top-1 raw L2 distance: min={distances[:, 0].min():.3f} mean={distances[:, 0].mean():.3f} max={distances[:, 0].max():.3f}")
    if max_dist is not None:
        norm = distances[:, 0].mean() / max_dist
        print(f"max_dist={max_dist:.3f} (from {args.max_distance_file}); top-1 mean / max_dist = {norm:.3f}")
    print(f"distinct top-1 source episodes: {len(np.unique(retrieved_pairs[:, 0, 0]))}/{len(rel_names)}")
    print(f"saved -> {args.output_dir}/retrievals.npz")
    print(f"saved -> {args.output_dir}/summary_plots.png")


if __name__ == "__main__":
    main()
