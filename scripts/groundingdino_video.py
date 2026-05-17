import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def build_prompt(nouns):
    return ". ".join(n.strip().lower() for n in nouns if n.strip()) + "."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--nouns", required=True, nargs="+", help="List of noun queries, e.g. --nouns cat dog 'red cup'")
    parser.add_argument("--output-frames-dir", required=True, help="Directory to write per-frame annotated images")
    parser.add_argument("--frame-ext", default="jpg", choices=["jpg", "png"], help="Per-frame image extension")
    parser.add_argument("--output-json", required=True, help="Output detections JSON")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for debugging")
    args = parser.parse_args()

    text_prompt = build_prompt(args.nouns)
    print(f"Prompt: {text_prompt!r}")

    print(f"Loading {args.model_id} on {args.device}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device)
    model.eval()
    print("Done.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.2f} fps, {n_frames} frames")

    os.makedirs(args.output_frames_dir, exist_ok=True)
    out_dir_j = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir_j:
        os.makedirs(out_dir_j, exist_ok=True)

    pad = max(6, len(str(max(n_frames, 1))))

    rng = np.random.default_rng(0)
    color_map = {
        n.strip().lower(): tuple(int(c) for c in rng.integers(40, 230, 3))
        for n in args.nouns
        if n.strip()
    }

    results_json = {
        "video": os.path.abspath(args.video),
        "nouns": args.nouns,
        "prompt": text_prompt,
        "model_id": args.model_id,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "fps": float(fps),
        "width": width,
        "height": height,
        "frames_dir": os.path.abspath(args.output_frames_dir),
        "frame_ext": args.frame_ext,
        "frames": [],
    }

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        scores = result["scores"].cpu().numpy()
        labels = result["labels"]
        boxes = result["boxes"].cpu().numpy()

        detections = []
        for score, label, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            label_key = (label or "").strip().lower()
            color = color_map.get(label_key, (0, 255, 0))

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            caption = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text_top = max(0, y1 - th - 4)
            cv2.rectangle(frame_bgr, (x1, y_text_top), (x1 + tw + 2, y1), color, -1)
            cv2.putText(
                frame_bgr,
                caption,
                (x1 + 1, max(th + 1, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            detections.append(
                {
                    "score": float(round(float(score), 5)),
                    "label": label,
                    "box_xyxy": [x1, y1, x2, y2],
                }
            )

        frame_name = f"frame_{frame_idx:0{pad}d}.{args.frame_ext}"
        frame_path = os.path.join(args.output_frames_dir, frame_name)
        if not cv2.imwrite(frame_path, frame_bgr):
            raise RuntimeError(f"Failed to write frame: {frame_path}")

        results_json["frames"].append(
            {"frame_idx": frame_idx, "image": frame_name, "detections": detections}
        )

        if frame_idx % 25 == 0:
            print(f"Frame {frame_idx}/{n_frames}: {len(detections)} detections -> {frame_name}")
        frame_idx += 1

    cap.release()

    with open(args.output_json, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Processed {frame_idx} frames")
    print(f"Wrote annotated frames to: {args.output_frames_dir}")
    print(f"Wrote detections JSON: {args.output_json}")


if __name__ == "__main__":
    main()
