"""Visualize TrajPerceiverResampler attention weights to diagnose whether
the perceiver is learning meaningful trajectory retrieval.

Usage:
    uv run scripts/visualize_perceiver_attn.py \
        --checkpoint-dir ./checkpoints/pi0_fast_libero_traj_perceiver/.../14999 \
        --demos-dir ./ricl_libero_preprocessing/collected_demos/libero_group/<task> \
        --out-dir ./attn_vis

Produces per-query heatmaps (num_latents × trajectory_time) and entropy plots.
"""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.policies.policy import TrajPerceiverPolicy


def load_model_and_policy(checkpoint_dir: str, config_name: str, demos_dir: str):
    train_config = _config.get_config(config_name)
    model = train_config.model.load(
        _model.restore_params(pathlib.Path(checkpoint_dir) / "params", dtype=jnp.bfloat16)
    )
    policy = TrajPerceiverPolicy(
        model,
        demos_dir=demos_dir,
        max_traj_len=train_config.model.max_traj_len,
    )
    get_attn = nnx_utils.module_jit(model.get_perceiver_attn_weights)
    return model, policy, get_attn


def build_obs_dict(policy: TrajPerceiverPolicy, query_npz, step_idx: int) -> dict:
    """Build a batched obs_dict for a single query step."""
    import openpi.transforms as _transforms
    from openpi.training import config as _config

    train_config = _config.get_config("pi0_fast_libero_traj_perceiver")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    query_dino_top_emb = query_npz["top_image_embeddings"][step_idx].reshape(64, 768).mean(axis=0).astype(np.float32)
    obs = {
        "query_top_image": query_npz["top_image"][step_idx],
        "query_wrist_image": query_npz["wrist_image"][step_idx],
        "query_state": query_npz["state"][step_idx],
        "query_prompt": query_npz["prompt"].item(),
        "query_dino_top_emb": query_dino_top_emb,
        "traj_state": policy._traj_state,
        "traj_top_emb": policy._traj_top_emb,
        "traj_wrist_emb": policy._traj_wrist_emb,
        "traj_mask": policy._traj_mask,
    }

    # Apply data + model transforms (resize, tokenize)
    for transform in data_config.data_transforms.inputs:
        obs = transform(obs)
    for transform in data_config.model_transforms.inputs:
        obs = transform(obs)

    # Batch
    return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], obs)


def plot_attn_heatmap(attn: np.ndarray, traj_mask: np.ndarray, title: str, out_path: pathlib.Path):
    """attn: [num_latents, T], traj_mask: [T]"""
    valid_len = int(traj_mask.sum())
    attn_valid = attn[:, :valid_len]  # trim padding

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

    # Heatmap: latents × trajectory time
    im = axes[0].imshow(attn_valid, aspect="auto", cmap="viridis", vmin=0)
    axes[0].set_xlabel("Trajectory timestep")
    axes[0].set_ylabel("Latent index")
    axes[0].set_title(f"{title}\nAttention weights [num_latents × T_valid={valid_len}]")
    plt.colorbar(im, ax=axes[0])

    # Per-latent entropy
    eps = 1e-9
    entropy = -(attn_valid * np.log(attn_valid + eps)).sum(axis=-1)  # [num_latents]
    max_entropy = np.log(valid_len)
    axes[1].barh(np.arange(len(entropy)), entropy / max_entropy, color="steelblue")
    axes[1].set_xlabel("Normalised entropy (0=sharp, 1=uniform)")
    axes[1].set_ylabel("Latent index")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Per-latent attention entropy")
    axes[1].axvline(1.0, color="red", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_entropy_over_steps(entropies: list[float], valid_len: int, out_path: pathlib.Path):
    """entropies: mean per-latent normalised entropy at each query step."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(entropies, marker="o", markersize=3)
    ax.set_xlabel("Query step (episode progress)")
    ax.set_ylabel("Mean normalised entropy")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="uniform")
    ax.set_title(f"Attention entropy over episode (traj valid len={valid_len})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config-name", default="pi0_fast_libero_traj_perceiver")
    parser.add_argument("--demos-dir", required=True,
                        help="Directory with demo subdirs (each has processed_demo.npz). "
                             "First demo is used as reference trajectory; second (if present) as query.")
    parser.add_argument("--query-demo-idx", type=int, default=1,
                        help="Which demo to use as the query episode (default: 1, i.e. different from reference).")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of evenly-spaced query steps to visualise.")
    parser.add_argument("--out-dir", default="attn_vis")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and policy...")
    model, policy, get_attn = load_model_and_policy(
        args.checkpoint_dir, args.config_name, args.demos_dir
    )

    # Load query demo
    folders = sorted(f for f in pathlib.Path(args.demos_dir).iterdir() if f.is_dir())
    if args.query_demo_idx >= len(folders):
        print(f"query_demo_idx={args.query_demo_idx} out of range, using 0")
        args.query_demo_idx = 0
    query_npz = np.load(folders[args.query_demo_idx] / "processed_demo.npz")
    ep_len = query_npz["state"].shape[0]
    print(f"Reference demo: {folders[0].name} ({int(policy._traj_mask.sum())} valid frames)")
    print(f"Query demo:     {folders[args.query_demo_idx].name} ({ep_len} steps)")

    # Pick evenly-spaced query steps
    step_indices = np.linspace(0, ep_len - 1, args.num_steps, dtype=int).tolist()

    entropies = []
    valid_len = int(policy._traj_mask.sum())
    max_entropy = np.log(valid_len + 1e-9)

    for i, step_idx in enumerate(step_indices):
        print(f"Processing query step {step_idx}/{ep_len}...")
        obs_dict = build_obs_dict(policy, query_npz, step_idx)
        attn = np.asarray(get_attn(obs_dict))[0]  # [num_latents, T]

        # Per-step heatmap
        plot_attn_heatmap(
            attn, policy._traj_mask,
            title=f"Query step {step_idx}/{ep_len} (demo {args.query_demo_idx})",
            out_path=out_dir / f"attn_step_{step_idx:04d}.png",
        )

        # Track mean entropy
        attn_valid = attn[:, :valid_len]
        eps = 1e-9
        ent = -(attn_valid * np.log(attn_valid + eps)).sum(axis=-1).mean()
        entropies.append(float(ent / max_entropy))

    # Entropy-over-episode plot
    plot_entropy_over_steps(
        entropies, valid_len,
        out_path=out_dir / "entropy_over_episode.png",
    )

    # Summary stats
    print(f"\n=== Summary ===")
    print(f"Mean normalised entropy: {np.mean(entropies):.3f}  (1.0 = fully uniform, 0 = perfectly sharp)")
    print(f"If entropy ≈ 1.0 → perceiver is not learning to retrieve, Q/K alignment is broken")
    print(f"If entropy varies across steps → Q conditioning is working")


if __name__ == "__main__":
    main()
