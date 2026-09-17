#!/usr/bin/env python3
"""
cv_cli.py — A command-line computer vision tool.

Supports object detection, image classification, and segmentation using
pretrained YOLOv8 models (via the `ultralytics` package). Works on single
images, folders of images, video files, or a live webcam feed.

Usage examples:
    # Object detection on an image, saves an annotated copy
    python cv_cli.py detect photo.jpg

    # Detection with a custom confidence threshold, print results as JSON
    python cv_cli.py detect photo.jpg --conf 0.4 --json

    # Classify an image (top-5 predicted classes)
    python cv_cli.py classify photo.jpg --topk 5

    # Segment objects in an image
    python cv_cli.py segment photo.jpg

    # Run detection on a whole folder of images
    python cv_cli.py detect ./my_photos/ --save-dir ./results

    # Run detection on a video file
    python cv_cli.py detect clip.mp4

    # Run detection live on a webcam (device 0), press 'q' to quit
    python cv_cli.py detect 0 --webcam

First run will auto-download the chosen pretrained model weights (small
files, a few MB to ~50MB depending on model size).
"""

import argparse
import json
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Default pretrained checkpoints (Ultralytics auto-downloads these).
DEFAULT_MODELS = {
    "detect": "yolov8n.pt",       # nano detection model, trained on COCO (80 classes)
    "segment": "yolov8n-seg.pt",  # nano segmentation model
    "classify": "yolov8n-cls.pt", # nano classification model, trained on ImageNet
}


def _require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        sys.exit(
            "The 'ultralytics' package is required but not installed.\n"
            "Install dependencies first:\n\n"
            "    pip install -r requirements.txt\n"
        )


def gather_sources(source: str, webcam: bool):
    """Resolve the --source argument into something YOLO's predict() accepts."""
    if webcam:
        return int(source)  # webcam device index, e.g. 0

    path = Path(source)
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            sys.exit(f"No supported images found in directory: {source}")
        return [str(p) for p in files]

    if path.is_file():
        return str(path)

    # Fall back: let YOLO handle URLs / raw strings itself
    return source


def run_task(task: str, args: argparse.Namespace):
    YOLO = _require_ultralytics()

    model_name = args.model or DEFAULT_MODELS[task]
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    source = gather_sources(args.source, args.webcam)

    save_dir = args.save_dir or "cv_cli_output"

    predict_kwargs = dict(
        source=source,
        conf=args.conf,
        save=not args.no_save,
        project=save_dir,
        name="run",
        exist_ok=True,
        stream=isinstance(source, int) or (isinstance(source, str) and Path(source).suffix.lower() in VIDEO_EXTS),
        show=args.show,
        verbose=False,
    )

    results_iter = model.predict(**predict_kwargs)

    all_results = []
    for result in results_iter:
        entry = {"source": str(getattr(result, "path", source))}

        if task in ("detect", "segment"):
            boxes = result.boxes
            detections = []
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    detections.append({
                        "class": result.names[cls_id],
                        "confidence": round(float(box.conf[0]), 4),
                        "box_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
                    })
            entry["detections"] = detections
            if not args.json:
                label = entry["source"]
                print(f"\n{label}: {len(detections)} object(s) found")
                for d in detections:
                    print(f"  - {d['class']:<15} conf={d['confidence']:.2f}  box={d['box_xyxy']}")

        elif task == "classify":
            probs = result.probs
            topk = args.topk
            top_indices = probs.top5[:topk]
            top_confs = probs.top5conf[:topk].tolist()
            predictions = [
                {"class": result.names[i], "confidence": round(float(c), 4)}
                for i, c in zip(top_indices, top_confs)
            ]
            entry["predictions"] = predictions
            if not args.json:
                print(f"\n{entry['source']}:")
                for p in predictions:
                    print(f"  {p['class']:<20} {p['confidence']*100:5.1f}%")

        all_results.append(entry)

    if args.json:
        print(json.dumps(all_results, indent=2))

    if not args.no_save and not args.webcam and all_results:
        print(f"\nAnnotated output saved under: runs/{task}/{save_dir}/run/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv_cli.py",
        description="Run computer vision models (detection, classification, segmentation) from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="task", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("source", help="Image path, video path, folder path, or webcam index (with --webcam)")
    common.add_argument("--model", default=None, help="Path or name of a model checkpoint (defaults to a pretrained YOLOv8-nano model)")
    common.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1), default 0.25")
    common.add_argument("--save-dir", default=None, help="Directory to save annotated output (default: cv_cli_output/)")
    common.add_argument("--no-save", action="store_true", help="Don't save annotated output files")
    common.add_argument("--show", action="store_true", help="Display results in a window as they're processed")
    common.add_argument("--webcam", action="store_true", help="Treat 'source' as a webcam device index (e.g. 0)")
    common.add_argument("--json", action="store_true", help="Print results as JSON instead of human-readable text")

    p_detect = sub.add_parser("detect", parents=[common], help="Detect and localize objects (bounding boxes)")
    p_segment = sub.add_parser("segment", parents=[common], help="Detect and segment objects (pixel masks)")
    p_classify = sub.add_parser("classify", parents=[common], help="Classify the whole image into a category")
    p_classify.add_argument("--topk", type=int, default=5, help="Number of top predictions to show (classify only)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_task(args.task, args)


if __name__ == "__main__":
    main()
