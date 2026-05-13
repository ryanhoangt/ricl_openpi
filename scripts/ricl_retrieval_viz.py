"""Gradio app to scrub through a query trajectory and inspect RICL retrievals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrievals", type=Path, required=True,
                        help="path to retrievals.npz produced by ricl_retrieval_compute.py")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("gradio not installed: `pip install gradio`")

    r = np.load(args.retrievals, allow_pickle=True)
    rel_names = r["rel_names"].tolist()
    query_episode = str(r["query_episode"])
    query_dir = Path(str(r["query_dir"]))
    index_dir = Path(str(r["index_dir"]))
    knn_k = int(r["knn_k"])
    n_steps = int(r["query_n_steps"])
    modality = str(r["query_modality"])
    distances = r["distances"]
    retrieved_pairs = np.stack([r["retrieved_ep_local_idx"], r["retrieved_step_idx"]], axis=-1)

    query_npz = np.load(query_dir / query_episode / "processed_demo.npz", allow_pickle=True)
    support_npzs = [np.load(index_dir / rel / "processed_demo.npz", allow_pickle=True) for rel in rel_names]
    query_prompt = str(query_npz["prompt"])

    def render(step):
        step = int(step)
        q_top = query_npz["top_image"][step]
        q_wrist = query_npz["wrist_image"][step]
        top_imgs, wrist_imgs = [], []
        for ki in range(knn_k):
            ep_local, step_idx = retrieved_pairs[step, ki]
            ep_name = rel_names[ep_local]
            ep_len = support_npzs[ep_local]["state"].shape[0]
            cap = f"#{ki + 1} {ep_name} | step {step_idx}/{ep_len - 1} | d={distances[step, ki]:.2f}"
            top_imgs.append((support_npzs[ep_local]["top_image"][step_idx], cap))
            wrist_imgs.append((support_npzs[ep_local]["wrist_image"][step_idx], cap))
        info = (
            f"query episode: {query_episode}\n"
            f"prompt: {query_prompt}\n"
            f"step: {step} / {n_steps - 1}\n"
            f"top-1 distance: {distances[step, 0]:.3f}\n"
            f"top-1 source: {rel_names[retrieved_pairs[step, 0, 0]]} step {retrieved_pairs[step, 0, 1]}\n"
            f"distinct sources in top-{knn_k}: {len(set(retrieved_pairs[step, :, 0].tolist()))}"
        )
        return q_top, q_wrist, top_imgs, wrist_imgs, info

    with gr.Blocks(title=f"RICL retrieval: {query_episode}") as demo:
        gr.Markdown(f"# RICL retrieval viz\n**query**: `{query_episode}` &nbsp; **modality**: `{modality}` &nbsp; **k**: {knn_k}")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Query")
                q_top = gr.Image(label="query top", height=240, width=240)
                q_wrist = gr.Image(label="query wrist", height=240, width=240)
                info = gr.Textbox(label="info", lines=7, interactive=False)
            with gr.Column(scale=4):
                gr.Markdown("### Retrieved (top-k)")
                top_gallery = gr.Gallery(label="retrieved top", columns=knn_k, height=280, object_fit="contain")
                wrist_gallery = gr.Gallery(label="retrieved wrist", columns=knn_k, height=280, object_fit="contain")
        slider = gr.Slider(minimum=0, maximum=n_steps - 1, step=1, value=0, label="query step")
        slider.change(render, inputs=slider, outputs=[q_top, q_wrist, top_gallery, wrist_gallery, info])
        demo.load(render, inputs=slider, outputs=[q_top, q_wrist, top_gallery, wrist_gallery, info])

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
