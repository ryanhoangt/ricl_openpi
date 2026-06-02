"""Gradio app for interactive SAM 2 segmentation of LIBERO preprocessed episodes.

Loads videos from `processed_demo.npz` files (top_image / wrist_image), lets the user
prompt SAM 2 interactively (positive/negative clicks, box, multiple objects), propagates
masklets across the video, and saves results to `sam_masks_{top,wrist}.npz` alongside
each episode. Re-opening an episode replays its saved prompts so refinement can resume.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import gradio as gr
except ImportError:
    raise SystemExit("gradio not installed: `pip install gradio`")


DEFAULT_DATA_ROOT = Path("/teamspace/studios/this_studio/ricl_openpi/ricl_libero_preprocessing/collected_demos")
DEFAULT_SAM_CKPT = Path("/teamspace/studios/this_studio/sam2/checkpoints/sam2.1_hiera_large.pt")
DEFAULT_SAM_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

TAB10 = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]

# Upscale the rendered canvas so the image fills the display container instead of
# being centered with padding (which causes gradio's evt.index to compress click
# coords toward the middle). All click coords are translated back to native space.
RENDER_SCALE = 3


def color_for(obj_id: int) -> tuple[int, int, int]:
    return TAB10[(obj_id - 1) % len(TAB10)] if obj_id > 0 else TAB10[0]


@dataclass
class AppState:
    predictor: object = None
    device: object = None
    data_root: Path | None = None

    npz_path: Path | None = None
    camera: str | None = None
    frames: np.ndarray | None = None              # (T,H,W,3) uint8
    tmp_dir: Path | None = None
    inference_state: object = None
    T: int = 0
    H: int = 0
    W: int = 0
    task_prompt: str = ""

    prompts: dict = field(default_factory=dict)        # obj_id -> frame_idx -> {points,labels,box}
    masks_cache: dict = field(default_factory=dict)    # (obj_id, frame_idx) -> (H,W) bool
    last_video_segments: dict | None = None            # frame_idx -> {obj_id: mask}
    pending_box_corner: tuple | None = None
    prompts_modified_since_propagate: bool = False

    all_episodes: list[Path] = field(default_factory=list)


STATE = AppState()


# ───────────────────────── episode discovery ─────────────────────────

def discover_episodes(root: Path) -> list[Path]:
    return sorted(root.rglob("processed_demo.npz"))


def saved_flags(ep_npz: Path) -> tuple[bool, bool]:
    return ((ep_npz.parent / "sam_masks_top.npz").exists(),
            (ep_npz.parent / "sam_masks_wrist.npz").exists())


def episode_label(ep_npz: Path) -> str:
    rel = ep_npz.parent.relative_to(STATE.data_root)
    t, w = saved_flags(ep_npz)
    mark = f"[top {'✅' if t else '·'} / wrist {'✅' if w else '·'}]"
    return f"{rel} {mark}"


def episode_choices(hide_fully_saved: bool) -> list[tuple[str, str]]:
    out = []
    for ep in STATE.all_episodes:
        t, w = saved_flags(ep)
        if hide_fully_saved and t and w:
            continue
        out.append((episode_label(ep), str(ep)))
    return out


def next_unsaved(current_path: str | None, camera: str) -> str | None:
    if not STATE.all_episodes:
        return None
    paths = [str(p) for p in STATE.all_episodes]
    start = paths.index(current_path) + 1 if current_path in paths else 0
    for p in paths[start:]:
        t, w = saved_flags(Path(p))
        done = t if camera == "top" else w
        if not done:
            return p
    return None


# ───────────────────────── core SAM ops ─────────────────────────

def load_episode(npz_path: Path, camera: str) -> None:
    """Load frames, dump JPEGs, init SAM state, replay saved prompts if present."""
    if STATE.tmp_dir is not None and STATE.tmp_dir.exists():
        shutil.rmtree(STATE.tmp_dir, ignore_errors=True)

    d = np.load(npz_path, allow_pickle=True)
    frames = d[f"{camera}_image"]
    T, H, W, _ = frames.shape

    tmp = Path(tempfile.mkdtemp(prefix=f"sam2_{camera}_"))
    for i, f in enumerate(frames):
        Image.fromarray(f).save(tmp / f"{i:05d}.jpg", quality=95)

    inference_state = STATE.predictor.init_state(video_path=str(tmp))

    STATE.npz_path = npz_path
    STATE.camera = camera
    STATE.frames = frames
    STATE.tmp_dir = tmp
    STATE.inference_state = inference_state
    STATE.T, STATE.H, STATE.W = int(T), int(H), int(W)
    STATE.task_prompt = str(d["prompt"])
    STATE.prompts = {}
    STATE.masks_cache = {}
    STATE.last_video_segments = None
    STATE.pending_box_corner = None
    STATE.prompts_modified_since_propagate = False

    saved_path = npz_path.parent / f"sam_masks_{camera}.npz"
    if saved_path.exists():
        try:
            saved = np.load(saved_path, allow_pickle=True)
            replay = saved["prompts"].item()
            for obj_id, by_frame in replay.items():
                for frame_idx, p in by_frame.items():
                    _submit_prompt(int(obj_id), int(frame_idx), p)
            if "masks" in saved.files and "obj_ids" in saved.files:
                masks = saved["masks"]
                saved_obj_ids = [int(o) for o in saved["obj_ids"].tolist()]
                video_segments: dict[int, dict[int, np.ndarray]] = {}
                for fi in range(masks.shape[0]):
                    video_segments[fi] = {
                        saved_obj_ids[j]: masks[fi, j].astype(bool)
                        for j in range(len(saved_obj_ids))
                    }
                STATE.last_video_segments = video_segments
                for fi, per_obj in video_segments.items():
                    for oid, m in per_obj.items():
                        STATE.masks_cache[(int(oid), int(fi))] = m
            STATE.prompts_modified_since_propagate = False
        except Exception as e:
            print(f"warning: failed to replay prompts from {saved_path}: {e}")


def _submit_prompt(obj_id: int, frame_idx: int, p: dict) -> None:
    """Send the full (points,labels,box) set for (obj_id, frame_idx) to the predictor."""
    pts = p.get("points") or []
    lbls = p.get("labels") or []
    box = p.get("box")

    if not pts and box is None:
        # nothing to submit
        STATE.prompts.setdefault(obj_id, {}).pop(frame_idx, None)
        STATE.masks_cache.pop((obj_id, frame_idx), None)
        return

    kwargs = dict(inference_state=STATE.inference_state, frame_idx=frame_idx, obj_id=obj_id)
    if pts:
        kwargs["points"] = np.asarray(pts, np.float32)
        kwargs["labels"] = np.asarray(lbls, np.int32)
    if box is not None:
        kwargs["box"] = np.asarray(box, np.float32)

    _, out_obj_ids, out_mask_logits = STATE.predictor.add_new_points_or_box(**kwargs)
    for i, oid in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
        STATE.masks_cache[(int(oid), int(frame_idx))] = mask

    STATE.prompts.setdefault(obj_id, {})[frame_idx] = {
        "points": [list(map(float, pt)) for pt in pts],
        "labels": [int(l) for l in lbls],
        "box": [float(v) for v in box] if box is not None else None,
    }
    STATE.prompts_modified_since_propagate = True


def replay_all_prompts() -> None:
    """Reset predictor state and re-submit every prompt in STATE.prompts."""
    STATE.predictor.reset_state(STATE.inference_state)
    STATE.masks_cache = {}
    STATE.last_video_segments = None
    current = STATE.prompts
    STATE.prompts = {}
    for obj_id, by_frame in current.items():
        for frame_idx, p in by_frame.items():
            _submit_prompt(int(obj_id), int(frame_idx), p)
    STATE.prompts_modified_since_propagate = True


def propagate_video() -> None:
    out = {}
    for fi, ids, logits in STATE.predictor.propagate_in_video(STATE.inference_state):
        out[int(fi)] = {int(ids[i]): (logits[i] > 0.0).cpu().numpy()[0] for i in range(len(ids))}
    STATE.last_video_segments = out
    # also refresh masks_cache for all (obj_id, frame_idx) so render uses fresh data
    for fi, per_obj in out.items():
        for oid, m in per_obj.items():
            STATE.masks_cache[(int(oid), int(fi))] = m
    STATE.prompts_modified_since_propagate = False


# ───────────────────────── rendering ─────────────────────────

def render_frame(frame_idx: int) -> np.ndarray:
    if STATE.frames is None:
        return np.zeros((224 * RENDER_SCALE, 224 * RENDER_SCALE, 3), dtype=np.uint8)
    frame_idx = max(0, min(frame_idx, STATE.T - 1))

    base = STATE.frames[frame_idx].astype(np.float32)
    overlay = base.copy()

    masks_to_draw: dict[int, np.ndarray] = {}
    if STATE.last_video_segments and frame_idx in STATE.last_video_segments:
        masks_to_draw.update(STATE.last_video_segments[frame_idx])
    for (oid, fi), m in STATE.masks_cache.items():
        if fi == frame_idx and oid not in masks_to_draw:
            masks_to_draw[oid] = m

    for oid, mask in masks_to_draw.items():
        if mask is None:
            continue
        color = np.array(color_for(oid), dtype=np.float32)
        sel = mask.astype(bool)
        overlay[sel] = 0.55 * overlay[sel] + 0.45 * color

    # Composite at native, then upscale for display so the image fills the container.
    img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    img = img.resize((STATE.W * RENDER_SCALE, STATE.H * RENDER_SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    s = RENDER_SCALE

    for oid, by_frame in STATE.prompts.items():
        if frame_idx not in by_frame:
            continue
        p = by_frame[frame_idx]
        ec = color_for(oid)
        if p.get("box") is not None:
            x0, y0, x1, y1 = (v * s for v in p["box"])
            draw.rectangle([x0, y0, x1, y1], outline=ec, width=2)
        for (x_n, y_n), lbl in zip(p.get("points", []), p.get("labels", [])):
            x, y = x_n * s, y_n * s
            fill = (0, 230, 0) if lbl == 1 else (240, 30, 30)
            # crosshair + white-haloed dot keeps the marker readable over masked regions
            draw.line([x - 10, y, x + 10, y], fill=(255, 255, 255), width=2)
            draw.line([x, y - 10, x, y + 10], fill=(255, 255, 255), width=2)
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=fill, outline=(255, 255, 255), width=2)

    if STATE.pending_box_corner is not None:
        x, y = STATE.pending_box_corner[0] * s, STATE.pending_box_corner[1] * s
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(255, 255, 0), width=3)

    return np.array(img)


def status_text() -> str:
    if STATE.npz_path is None:
        return "no episode loaded — pick an episode and click Load"
    saved_path = STATE.npz_path.parent / f"sam_masks_{STATE.camera}.npz"
    lines = [
        f"episode: {STATE.npz_path.parent.relative_to(STATE.data_root)}",
        f"task: {STATE.task_prompt}",
        f"camera: {STATE.camera}   T={STATE.T}   H×W={STATE.H}×{STATE.W}",
        f"objects: {sorted(STATE.prompts.keys()) or '[]'}",
        f"saved file: {saved_path.name} ({'exists' if saved_path.exists() else 'not yet'})",
        f"propagated: {'yes' if STATE.last_video_segments else 'no'}",
    ]
    if STATE.prompts_modified_since_propagate and STATE.last_video_segments is not None:
        lines.append("⚠ prompts changed since last propagate — re-run Propagate before saving")
    if STATE.pending_box_corner is not None:
        lines.append(f"⚠ box first corner at {STATE.pending_box_corner} — click second corner")
    return "\n".join(lines)


def objects_rows() -> list[list]:
    rows = []
    for oid in sorted(STATE.prompts.keys()):
        bf = STATE.prompts[oid]
        n_frames = len(bf)
        total_pts = sum(len(bf[fi].get("points", [])) for fi in bf)
        n_boxes = sum(1 for fi in bf if bf[fi].get("box") is not None)
        rows.append([oid, n_frames, total_pts, n_boxes])
    return rows


def gallery_items(stride: int) -> list[tuple[np.ndarray, str]]:
    if STATE.frames is None or STATE.last_video_segments is None:
        return []
    items = []
    for fi in range(0, STATE.T, max(1, stride)):
        if fi not in STATE.last_video_segments:
            continue
        base = STATE.frames[fi].astype(np.float32)
        overlay = base.copy()
        for oid, mask in STATE.last_video_segments[fi].items():
            color = np.array(color_for(oid), dtype=np.float32)
            sel = mask.astype(bool)
            overlay[sel] = 0.55 * overlay[sel] + 0.45 * color
        img = np.clip(overlay, 0, 255).astype(np.uint8)
        items.append((img, f"frame {fi}"))
    return items


# ───────────────────────── save ─────────────────────────

def save_current() -> Path | None:
    if STATE.npz_path is None or STATE.last_video_segments is None:
        return None
    obj_ids = sorted({oid for fi in STATE.last_video_segments for oid in STATE.last_video_segments[fi]})
    if not obj_ids:
        return None
    masks = np.zeros((STATE.T, len(obj_ids), STATE.H, STATE.W), dtype=bool)
    for fi, per_obj in STATE.last_video_segments.items():
        for j, oid in enumerate(obj_ids):
            if oid in per_obj:
                masks[fi, j] = per_obj[oid]

    # Sanitize prompts to plain Python types so np.savez round-trips cleanly.
    clean_prompts = {
        int(oid): {
            int(fi): {
                "points": [list(map(float, pt)) for pt in p.get("points", [])],
                "labels": [int(l) for l in p.get("labels", [])],
                "box": [float(v) for v in p["box"]] if p.get("box") is not None else None,
            }
            for fi, p in by_frame.items()
        }
        for oid, by_frame in STATE.prompts.items()
    }

    out_path = STATE.npz_path.parent / f"sam_masks_{STATE.camera}.npz"
    np.savez_compressed(
        out_path,
        source_path=str(STATE.npz_path),
        camera=STATE.camera,
        obj_ids=np.asarray(obj_ids, np.int32),
        masks=masks,
        prompts=np.array(clean_prompts, dtype=object),
        frame_indices=np.arange(STATE.T, dtype=np.int32),
        task_prompt=STATE.task_prompt,
    )
    return out_path


# ───────────────────────── UI ─────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sam-ckpt", type=Path, default=DEFAULT_SAM_CKPT)
    parser.add_argument("--sam-cfg", type=str, default=DEFAULT_SAM_CFG)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # sam2's build_sam refuses to import when CWD contains a sibling `sam2/` directory
    # (e.g. when launched from the repo parent). Hop to a neutral CWD before importing.
    sam2_pkg_parent = Path(__file__).resolve().parent
    if (Path.cwd() / "sam2" / "sam2").is_dir():
        os.chdir(sam2_pkg_parent)
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    STATE.predictor = build_sam2_video_predictor(args.sam_cfg, str(args.sam_ckpt), device=device)
    STATE.device = device
    STATE.data_root = args.data_root
    STATE.all_episodes = discover_episodes(args.data_root)
    if not STATE.all_episodes:
        raise SystemExit(f"no processed_demo.npz files found under {args.data_root}")
    print(f"discovered {len(STATE.all_episodes)} episodes under {args.data_root}")

    # ---------- callbacks ----------
    def cb_refresh_list(hide_saved):
        return gr.update(choices=episode_choices(hide_saved))

    def cb_load(ep_path, camera, stride_val):
        if not ep_path:
            return None, "select an episode first", [], gr.update(), gr.update(), []
        load_episode(Path(ep_path), camera)
        return (
            render_frame(0),
            status_text(),
            objects_rows(),
            gr.update(minimum=0, maximum=STATE.T - 1, value=0),
            gr.update(choices=episode_choices(False)),
            gallery_items(int(stride_val)),
        )

    def cb_change_frame(frame_idx):
        return render_frame(int(frame_idx))

    def cb_canvas_click(evt: gr.SelectData, frame_idx, obj_id, mode):
        if STATE.frames is None:
            return render_frame(0), status_text(), objects_rows()
        frame_idx = int(frame_idx)
        obj_id = int(obj_id)
        x_raw, y_raw = evt.index
        # Click coords come back in the rendered (upscaled) image's natural space —
        # translate to the original frame's native pixel space before storing.
        x = float(max(0, min(x_raw / RENDER_SCALE, STATE.W - 1)))
        y = float(max(0, min(y_raw / RENDER_SCALE, STATE.H - 1)))

        existing = STATE.prompts.get(obj_id, {}).get(frame_idx, {"points": [], "labels": [], "box": None})
        new_points = list(existing.get("points", []))
        new_labels = list(existing.get("labels", []))
        new_box = existing.get("box")

        if mode == "box":
            if STATE.pending_box_corner is None:
                STATE.pending_box_corner = (x, y)
                return render_frame(frame_idx), status_text(), objects_rows()
            x0, y0 = STATE.pending_box_corner
            new_box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            STATE.pending_box_corner = None
        else:
            new_points.append([x, y])
            new_labels.append(1 if mode == "+ click" else 0)

        _submit_prompt(obj_id, frame_idx, {"points": new_points, "labels": new_labels, "box": new_box})
        return render_frame(frame_idx), status_text(), objects_rows()

    def cb_new_obj(_obj_id):
        next_id = (max(STATE.prompts.keys()) + 1) if STATE.prompts else 1
        STATE.pending_box_corner = None
        return next_id

    def cb_change_obj(_obj_id):
        STATE.pending_box_corner = None
        return render_frame(0 if STATE.frames is None else 0), status_text()

    def cb_clear_obj_on_frame(frame_idx, obj_id):
        if STATE.frames is None:
            return render_frame(0), status_text(), objects_rows()
        frame_idx = int(frame_idx)
        obj_id = int(obj_id)
        if obj_id in STATE.prompts and frame_idx in STATE.prompts[obj_id]:
            del STATE.prompts[obj_id][frame_idx]
            if not STATE.prompts[obj_id]:
                del STATE.prompts[obj_id]
            replay_all_prompts()
        STATE.pending_box_corner = None
        return render_frame(frame_idx), status_text(), objects_rows()

    def cb_reset_all(frame_idx):
        if STATE.inference_state is not None:
            STATE.predictor.reset_state(STATE.inference_state)
        STATE.prompts = {}
        STATE.masks_cache = {}
        STATE.last_video_segments = None
        STATE.pending_box_corner = None
        STATE.prompts_modified_since_propagate = False
        return render_frame(int(frame_idx or 0)), status_text(), objects_rows(), []

    def cb_propagate(frame_idx, stride):
        if STATE.inference_state is None or not STATE.prompts:
            return render_frame(int(frame_idx or 0)), status_text(), objects_rows(), []
        propagate_video()
        return (
            render_frame(int(frame_idx or 0)),
            status_text(),
            objects_rows(),
            gallery_items(int(stride)),
        )

    def cb_save(frame_idx):
        if STATE.last_video_segments is None:
            return render_frame(int(frame_idx or 0)), status_text() + "\n⚠ run Propagate before saving", objects_rows(), gr.update()
        out = save_current()
        msg = f"saved → {out}" if out else "nothing to save"
        return (
            render_frame(int(frame_idx or 0)),
            status_text() + f"\n{msg}",
            objects_rows(),
            gr.update(choices=episode_choices(False)),
        )

    def cb_save_next(frame_idx, camera, hide_saved):
        out = save_current() if STATE.last_video_segments is not None else None
        current = str(STATE.npz_path) if STATE.npz_path else None
        nxt = next_unsaved(current, camera)
        msg = f"saved → {out}\n" if out else ("⚠ nothing saved (run Propagate first)\n" if STATE.last_video_segments is None else "")
        if nxt is None:
            return (
                render_frame(int(frame_idx or 0)),
                status_text() + f"\n{msg}no more unsaved episodes for camera={camera}",
                objects_rows(),
                gr.update(choices=episode_choices(hide_saved)),
                gr.update(),
                gr.update(),
            )
        load_episode(Path(nxt), camera)
        return (
            render_frame(0),
            status_text() + f"\n{msg}auto-advanced to next unsaved",
            objects_rows(),
            gr.update(choices=episode_choices(hide_saved), value=nxt),
            gr.update(minimum=0, maximum=STATE.T - 1, value=0),
            [],
        )

    # ---------- layout ----------
    with gr.Blocks(title="SAM 2 LIBERO annotator") as demo:
        gr.Markdown("# SAM 2 annotator for LIBERO preprocessed episodes")

        with gr.Row():
            # left column: episode picker
            with gr.Column(scale=1):
                gr.Markdown("### Episode")
                hide_saved = gr.Checkbox(label="hide fully-saved episodes", value=False)
                episode = gr.Dropdown(label="episode", choices=episode_choices(False), value=None, filterable=True)
                camera = gr.Radio(label="camera", choices=["top", "wrist"], value="top")
                load_btn = gr.Button("Load / re-init", variant="primary")
                status = gr.Textbox(label="status", value=status_text(), lines=9, interactive=False)

            # middle column: canvas
            with gr.Column(scale=2):
                gr.Markdown("### Annotate (click on the image)")
                canvas = gr.Image(label="frame", value=render_frame(0), height=520,
                                  interactive=False, show_label=False)
                frame_slider = gr.Slider(label="frame", minimum=0, maximum=1, step=1, value=0)
                gr.Markdown("### Propagation preview")
                stride = gr.Slider(label="gallery stride", minimum=1, maximum=30, step=1, value=10)
                gallery = gr.Gallery(label="propagated masks", columns=6, height=240, object_fit="contain")

            # right column: prompt controls
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                obj_id = gr.Number(label="active obj_id", value=1, precision=0)
                new_obj_btn = gr.Button("+ new object")
                mode = gr.Radio(label="click mode", choices=["+ click", "− click", "box"], value="+ click")
                with gr.Row():
                    clear_obj_btn = gr.Button("clear obj on frame")
                    reset_btn = gr.Button("reset all", variant="stop")
                propagate_btn = gr.Button("Propagate", variant="primary")
                with gr.Row():
                    save_btn = gr.Button("Save")
                    save_next_btn = gr.Button("Save & next", variant="primary")
                gr.Markdown("### Objects")
                obj_table = gr.Dataframe(
                    headers=["obj_id", "n_frames", "n_points", "n_boxes"],
                    value=objects_rows(),
                    interactive=False,
                )

        # ---------- wiring ----------
        hide_saved.change(cb_refresh_list, inputs=hide_saved, outputs=episode)

        load_btn.click(
            cb_load,
            inputs=[episode, camera, stride],
            outputs=[canvas, status, obj_table, frame_slider, episode, gallery],
        )

        frame_slider.change(cb_change_frame, inputs=frame_slider, outputs=canvas)
        stride.change(lambda s: gallery_items(int(s)), inputs=stride, outputs=gallery)

        canvas.select(
            cb_canvas_click,
            inputs=[frame_slider, obj_id, mode],
            outputs=[canvas, status, obj_table],
        )

        new_obj_btn.click(cb_new_obj, inputs=obj_id, outputs=obj_id)
        obj_id.change(cb_change_obj, inputs=obj_id, outputs=[canvas, status])
        mode.change(lambda _m: (setattr(STATE, "pending_box_corner", None) or status_text()),
                    inputs=mode, outputs=status)

        clear_obj_btn.click(cb_clear_obj_on_frame, inputs=[frame_slider, obj_id],
                            outputs=[canvas, status, obj_table])
        reset_btn.click(cb_reset_all, inputs=frame_slider,
                        outputs=[canvas, status, obj_table, gallery])

        propagate_btn.click(cb_propagate, inputs=[frame_slider, stride],
                            outputs=[canvas, status, obj_table, gallery])
        save_btn.click(cb_save, inputs=frame_slider, outputs=[canvas, status, obj_table, episode])
        save_next_btn.click(cb_save_next, inputs=[frame_slider, camera, hide_saved],
                            outputs=[canvas, status, obj_table, episode, frame_slider, gallery])

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
