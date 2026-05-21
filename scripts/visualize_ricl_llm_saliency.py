"""Visualize which RICL context tokens influence LLM action generation most.

Uses input-gradient saliency: computes grad(action_loss) w.r.t. input embeddings
(after SigLIP + text encoding). No modifications to gemma_fast.py needed.

Usage:
    uv run scripts/visualize_ricl_llm_saliency.py \
        --checkpoint-dir ./checkpoints/pi0_fast_libero_ricl/.../14999 \
        --demos-dir ./ricl_libero_preprocessing/collected_demos/libero_group/<task> \
        --out-dir ./saliency_vis

Per block the token layout is:
    [base_img(256)] [wrist_img(256)] [zeros_img(256)] [text(max_token_len)]
    └── images ───────────────────────────────────────┘ └── state+prompt+actions ┘

The saliency map shows, per position, how much that token influences the
action log-likelihood of the query block. Compare image vs. action token regions.
"""

import argparse
import pathlib

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Paper-friendly default style.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.policies.policy_config import create_trained_policy


NUM_IMG_TOKENS_PER_IMAGE = 256  # SigLIP So400m/14 at 224×224 → 16×16 patches
NUM_IMAGES_PER_OBS = 3          # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb (third is zeros)
IMG_TOKENS_PER_BLOCK = NUM_IMG_TOKENS_PER_IMAGE * NUM_IMAGES_PER_OBS  # 768


def load_model(checkpoint_dir: str, config_name: str):
    train_config = _config.get_config(config_name)
    model = train_config.model.load(
        _model.restore_params(pathlib.Path(checkpoint_dir) / "params", dtype=jnp.bfloat16)
    )
    return model, train_config


MODALITY_KEYS = {
    "top": ["top_image_embeddings"],
    "wrist": ["wrist_image_embeddings"],
    "both": ["top_image_embeddings", "wrist_image_embeddings"],
}


def _action_chunk(npz, step, action_horizon):
    actions = npz["actions"]
    end = min(step + action_horizon, len(actions))
    chunk = actions[step:end]
    if len(chunk) < action_horizon:
        chunk = np.concatenate([chunk, np.tile(chunk[-1:], (action_horizon - len(chunk), 1))])
    return chunk


def _fill_slot(data, prefix, npz, step, action_horizon):
    data[f"{prefix}top_image"] = npz["top_image"][step]
    data[f"{prefix}wrist_image"] = npz["wrist_image"][step]
    data[f"{prefix}state"] = npz["state"][step]
    data[f"{prefix}actions"] = _action_chunk(npz, step, action_horizon)
    data[f"{prefix}prompt"] = npz["prompt"].item()


def build_ricl_raw_data_nn(demos_dir: str, num_retrieved: int, action_horizon: int,
                            query_modality: str = "top") -> dict:
    """Build a raw data dict using real DINOv2-embedding NN retrieval.

    Mirrors RiclPolicy.retrieve: demos[0] = query (mid-episode); demos[1:] = support set
    (leave-one-out). Slot i = i-th nearest neighbor by L2 distance over `query_modality`
    embeddings.
    """
    from autofaiss import build_index

    folders = sorted(f for f in pathlib.Path(demos_dir).iterdir() if f.is_dir())
    assert len(folders) >= 2, f"Need at least 2 demo dirs in {demos_dir}, found {len(folders)}"

    keys = MODALITY_KEYS[query_modality]

    def load(folder):
        return np.load(folder / "processed_demo.npz")

    def modality_emb(npz):
        return np.concatenate([npz[k] for k in keys], axis=1).astype(np.float32)

    query_npz = load(folders[0])
    query_step = len(query_npz["state"]) // 2
    query_emb = modality_emb(query_npz)[query_step:query_step + 1]

    support_npzs = [load(f) for f in folders[1:]]
    support_emb = np.concatenate([modality_emb(n) for n in support_npzs], axis=0)
    flat_to_ep_step = np.array(
        [(ep_idx, step) for ep_idx, n in enumerate(support_npzs)
         for step in range(n["state"].shape[0])]
    )

    knn_index, _ = build_index(
        embeddings=support_emb,
        save_on_disk=False,
        min_nearest_neighbors_to_retrieve=max(num_retrieved + 5, 20),
        max_index_query_time_ms=10,
        max_index_memory_usage="25G",
        current_memory_available="50G",
        metric_type="l2",
        nb_cores=8,
    )

    _, flat_indices = knn_index.search(query_emb, num_retrieved)
    retrieved_pairs = flat_to_ep_step[flat_indices[0]]

    data = {}
    for i, (ep_idx, step) in enumerate(retrieved_pairs):
        _fill_slot(data, f"retrieved_{i}_", support_npzs[int(ep_idx)], int(step), action_horizon)

    _fill_slot(data, "query_", query_npz, query_step, action_horizon)
    data["query_prompt"] = query_npz["prompt"].item()
    data["exp_lamda_distances"] = np.ones((num_retrieved + 1, 1), dtype=np.float32)

    print(f"NN retrieval: query={folders[0].name} step={query_step}, "
          f"slots={[(int(e), int(s)) for e, s in retrieved_pairs]}")
    return data


def build_ricl_raw_data(demos_dir: str, num_retrieved: int, action_horizon: int) -> dict:
    """Build a raw data dict from demo npz files.

    Uses demo[0] as query, demos[1..num_retrieved] (cycling) as retrieved.
    """
    folders = sorted(f for f in pathlib.Path(demos_dir).iterdir() if f.is_dir())
    assert len(folders) >= 2, f"Need at least 2 demo dirs in {demos_dir}"

    def load(folder):
        return np.load(folder / "processed_demo.npz")

    query_npz = load(folders[0])
    query_step = len(query_npz["state"]) // 2  # mid-episode for interesting context

    def action_chunk(npz, step):
        actions = npz["actions"]
        end = min(step + action_horizon, len(actions))
        chunk = actions[step:end]
        if len(chunk) < action_horizon:
            chunk = np.concatenate([chunk, np.tile(chunk[-1:], (action_horizon - len(chunk), 1))])
        return chunk

    data = {}
    for i in range(num_retrieved):
        npz = load(folders[(i + 1) % len(folders)])
        step = (i * 7) % len(npz["state"])  # spread across the episode
        prefix = f"retrieved_{i}_"
        data[f"{prefix}top_image"] = npz["top_image"][step]
        data[f"{prefix}wrist_image"] = npz["wrist_image"][step]
        data[f"{prefix}state"] = npz["state"][step]
        data[f"{prefix}actions"] = action_chunk(npz, step)
        data[f"{prefix}prompt"] = npz["prompt"].item()

    data["query_top_image"] = query_npz["top_image"][query_step]
    data["query_wrist_image"] = query_npz["wrist_image"][query_step]
    data["query_state"] = query_npz["state"][query_step]
    data["query_actions"] = action_chunk(query_npz, query_step)
    data["query_prompt"] = query_npz["prompt"].item()
    # no action interpolation needed for saliency
    data["exp_lamda_distances"] = np.ones((num_retrieved + 1, 1), dtype=np.float32)
    return data


def apply_transforms(data: dict, data_config) -> dict:
    for t in data_config.data_transforms.inputs:
        data = t(data)
    for t in data_config.model_transforms.inputs:
        data = t(data)
    return data


def batch_data(data: dict) -> dict:
    return jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], data)


def build_ricl_observation(data: dict, num_retrieved: int) -> _model.RiclObservation:
    return _model.RiclObservation.from_dict(data, num_retrieved_observations=num_retrieved)


def compute_saliency(model, ricl_obs: _model.RiclObservation) -> tuple[np.ndarray, list, dict]:
    """Returns saliency [T], block_ranges list, and token_role_map."""
    from openpi.models.pi0_fast_ricl import make_attn_mask

    num_retrieved = model.num_retrieved_observations
    num_obs = num_retrieved + 1

    # --- Build per-block embeddings and track ranges ---
    list_of_embeddings = []
    list_of_attn_masks = []
    block_ranges = []  # (start, end, is_query, token_loss_mask)
    pos = 0

    for i in range(num_obs):
        prefix = f"retrieved_{i}_" if i < num_retrieved else "query_"
        is_query = (i == num_retrieved)
        obs_i = _model.extract_observation_from_ricl_observation(ricl_obs, prefix)
        obs_i = _model.preprocess_observation_prefix_postfix(
            None, obs_i, train=False, image_keys=list(obs_i.images.keys())
        )
        emb_i, mask_i, ar_i = model.embed_inputs(obs_i)
        attn_i = make_attn_mask(mask_i, ar_i)

        block_len = emb_i.shape[1]
        loss_mask_i = obs_i.token_loss_mask  # [B, max_token_len] — 1 for action tokens

        list_of_embeddings.append(emb_i)
        list_of_attn_masks.append(attn_i)
        block_ranges.append((pos, pos + block_len, is_query, loss_mask_i))
        pos += block_len

    all_embeddings = jnp.concatenate(list_of_embeddings, axis=1)  # [1, T, D]
    batch_size, seq_len = all_embeddings.shape[:2]

    attn_mask = model.combine_attn_masks(list_of_attn_masks, batch_size, seq_len, num_obs)

    # Targets: action tokens from the query block
    query_obs = _model.extract_observation_from_ricl_observation(ricl_obs, "query_")
    query_obs = _model.preprocess_observation_prefix_postfix(
        None, query_obs, train=False, image_keys=list(query_obs.images.keys())
    )
    targets = jax.nn.one_hot(
        jnp.concatenate([
            query_obs.tokenized_prompt_prefix[:, 1:],
            query_obs.tokenized_prompt_postfix
        ], axis=1),
        model.PaliGemma.llm.module.vocab_size,
    )
    loss_mask = query_obs.token_loss_mask[:, 1:]  # shift by 1 for next-token prediction

    # --- Gradient saliency w.r.t. input embeddings ---
    def action_nll(embeddings):
        pre_logits, _, _ = model.PaliGemma.llm(
            embedded_prefix=embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )
        logits, _ = model.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1]:],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.sum(targets * logp * loss_mask[..., None])

    grads = jax.grad(action_nll)(all_embeddings)
    # Input × gradient saliency, L2 over embedding dim → [T]
    saliency = np.asarray(jnp.sqrt(jnp.sum((grads * all_embeddings) ** 2, axis=-1))[0].astype(jnp.float32))

    # --- Build token role map for annotation ---
    token_roles = np.full(seq_len, "pad", dtype=object)
    for idx, (bstart, bend, is_query, tloss) in enumerate(block_ranges):
        img_end = bstart + IMG_TOKENS_PER_BLOCK
        text_start = img_end
        prefix = "query" if is_query else f"ret_{idx}"
        token_roles[bstart:img_end] = f"{prefix}_img"
        # within text region: action tokens vs. state/prompt tokens
        loss_flat = np.asarray(tloss[0])  # [max_token_len]
        for j, l in enumerate(loss_flat):
            tok_pos = text_start + j
            if tok_pos < bend:
                token_roles[tok_pos] = f"{prefix}_action" if l else f"{prefix}_ctx"

    return saliency, block_ranges, token_roles


def _bin_saliency_per_block(saliency: np.ndarray, token_roles: np.ndarray,
                             bstart: int, bend: int, n_bins: int):
    """Bin saliency within a single block. Returns bin centers (in global
    coordinates) and per-role mean values."""
    block_len = bend - bstart
    n_bins = min(n_bins, block_len)
    edges = np.linspace(bstart, bend, n_bins + 1, dtype=int)
    centers = (edges[:-1] + edges[1:]) / 2.0
    per_role = {"img": np.full(n_bins, np.nan), "ctx": np.full(n_bins, np.nan), "action": np.full(n_bins, np.nan)}
    for b in range(n_bins):
        sl = slice(edges[b], edges[b + 1])
        roles_in_bin = token_roles[sl]
        sal_in_bin = saliency[sl]
        for role in per_role:
            mask = np.array([r.endswith(f"_{role}") for r in roles_in_bin])
            if mask.any():
                per_role[role][b] = sal_in_bin[mask].mean()
    return centers, per_role


def plot_saliency_per_position(
    saliency: np.ndarray,
    token_roles: np.ndarray,
    block_ranges: list,
    num_retrieved: int,
    out_path: pathlib.Path,
    bins_per_block: int = 80,
):
    """Compact per-position view (binned within each block separately so lines
    never connect across block boundaries)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.0))

    role_styles = [
        ("img", "#1f77b4", "image"),
        ("action", "#d62728", "action"),
        ("ctx", "#2ca02c", "state/prompt"),
    ]

    # Plot once per (block, role). Use the legend label only on the first
    # block so the legend has one entry per role.
    seen_labels = set()
    for idx, (bstart, bend, is_query, _) in enumerate(block_ranges):
        centers, per_role = _bin_saliency_per_block(saliency, token_roles, bstart, bend, bins_per_block)
        for role, color, label in role_styles:
            y = per_role[role]
            valid = ~np.isnan(y)
            if not valid.any():
                continue
            legend_label = label if label not in seen_labels else None
            ax.plot(centers[valid], y[valid], color=color, linewidth=1.0,
                    label=legend_label)
            seen_labels.add(label)

    # Block boundary verticals + top labels (set after plotting so ylim is known).
    ax.set_xlim(block_ranges[0][0], block_ranges[-1][1])
    ax.margins(y=0.15)
    y_top = ax.get_ylim()[1]
    for idx, (bstart, bend, is_query, _) in enumerate(block_ranges):
        if idx > 0:
            ax.axvline(bstart, color="gray", linewidth=0.4, linestyle="--")
        mid = (bstart + bend) / 2.0
        block_label = "query" if is_query else f"ret_{idx}"
        ax.text(mid, y_top, block_label, ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Token position")
    ax.set_ylabel("Input×Grad saliency")
    ax.set_title("RICL LLM input saliency over token positions", pad=18)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_saliency_means(
    saliency: np.ndarray,
    token_roles: np.ndarray,
    block_ranges: list,
    num_retrieved: int,
    out_path: pathlib.Path,
):
    """Bar chart of mean saliency per (block, role)."""
    means, labels, colors = [], [], []
    for idx, (bstart, bend, is_query, _) in enumerate(block_ranges):
        prefix = "query" if is_query else f"ret_{idx}"
        img_end = bstart + IMG_TOKENS_PER_BLOCK
        means.append(float(saliency[bstart:img_end].mean()))
        labels.append(f"{prefix}\nimg")
        colors.append("steelblue")
        ctx_mask = np.array([r == f"{prefix}_ctx" for r in token_roles])
        if ctx_mask.any():
            means.append(float(saliency[ctx_mask].mean()))
            labels.append(f"{prefix}\nctx")
            colors.append("seagreen")
        act_mask = np.array([r == f"{prefix}_action" for r in token_roles])
        if act_mask.any():
            means.append(float(saliency[act_mask].mean()))
            labels.append(f"{prefix}\naction")
            colors.append("tomato")

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bars = ax.bar(range(len(means)), means, color=colors)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Input×Grad saliency")
    ax.set_title("Mean saliency per (block, role)")

    # Annotate values
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config-name", default="pi0_fast_libero_ricl")
    parser.add_argument("--demos-dir", required=True,
                        help="Directory with demo subdirs, each containing processed_demo.npz")
    parser.add_argument("--out-dir", default="saliency_vis")
    parser.add_argument("--use-nn-retrieval", action="store_true",
                        help="Use real DINOv2-embedding NN retrieval (matches RiclPolicy.retrieve). "
                             "Default off uses the original synthetic context (demos cycled by index).")
    parser.add_argument("--query-modality", choices=list(MODALITY_KEYS), default="top",
                        help="Embedding modality for NN retrieval (only used with --use-nn-retrieval).")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, train_config = load_model(args.checkpoint_dir, args.config_name)
    num_retrieved = model.num_retrieved_observations
    action_horizon = model.action_horizon

    print("Building input from demos...")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if args.use_nn_retrieval:
        raw_data = build_ricl_raw_data_nn(args.demos_dir, num_retrieved, action_horizon,
                                          query_modality=args.query_modality)
    else:
        raw_data = build_ricl_raw_data(args.demos_dir, num_retrieved, action_horizon)
    data = apply_transforms(raw_data, data_config)
    data = batch_data(data)
    ricl_obs = build_ricl_observation(data, num_retrieved)

    print("Computing saliency (this runs a backward pass through the LLM)...")
    saliency, block_ranges, token_roles = compute_saliency(model, ricl_obs)

    print("\n=== Mean saliency by role ===")
    for role in ["img", "ctx", "action"]:
        for idx in list(range(num_retrieved)) + ["query"]:
            prefix = "query" if idx == "query" else f"ret_{idx}"
            key = f"{prefix}_{role}"
            mask = token_roles == key
            if mask.any():
                print(f"  {key:30s}: {saliency[mask].mean():.4f}  (n={mask.sum()})")

    # Overall: retrieved img vs retrieved action
    ret_img_mask = np.array([r.startswith("ret_") and r.endswith("_img") for r in token_roles])
    ret_act_mask = np.array([r.startswith("ret_") and r.endswith("_action") for r in token_roles])
    ret_ctx_mask = np.array([r.startswith("ret_") and r.endswith("_ctx") for r in token_roles])
    print(f"\n  All retrieved img:    {saliency[ret_img_mask].mean():.4f}")
    print(f"  All retrieved action: {saliency[ret_act_mask].mean():.4f}")
    print(f"  All retrieved ctx:    {saliency[ret_ctx_mask].mean():.4f}")
    print("\n  Higher saliency = LLM relies on these tokens more for action generation.")

    plot_saliency_per_position(saliency, token_roles, block_ranges, num_retrieved,
                                out_path=out_dir / "ricl_saliency_positions.png")
    plot_saliency_means(saliency, token_roles, block_ranges, num_retrieved,
                        out_path=out_dir / "ricl_saliency_means.png")


if __name__ == "__main__":
    main()
