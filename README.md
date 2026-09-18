# Computer Vision from the Command Line

## Student Information

| Field | Details |
|---|---|
| **Name** | Harsh Singh |
| **Roll Number** | 24BAI10795 |
| **Slot** | F11 + F12 |
| **Date of Submission** | 18-09-2026 |

A single-file Python CLI that runs pretrained computer vision models
(object detection, image classification, and instance segmentation)
on images, folders, videos, or a live webcam feed.

It's built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
using small "nano" checkpoints pretrained on COCO (80 object classes) for
detection/segmentation and ImageNet (1000 classes) for classification.
Weights are downloaded automatically the first time you use each mode
(a few MB to ~50MB depending on which model size you pick).

## 1. Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

This pulls in `ultralytics`, which in turn installs `torch`, `opencv-python`,
and `numpy`. On a fresh machine this download is a few hundred MB the first
time — that's normal.

## 2. Usage

```bash
python cv_cli.py <task> <source> [options]
```

**Tasks:** `detect`, `classify`, `segment`

**Source** can be:
- a path to a single image (`photo.jpg`)
- a path to a video file (`clip.mp4`)
- a path to a folder of images (`./my_photos/`)
- a webcam device index, combined with `--webcam` (e.g. `0`)

### Examples

```bash
# Detect objects in an image — draws boxes, prints results, saves annotated copy
python cv_cli.py detect photo.jpg

# Raise/lower the confidence threshold (default 0.25)
python cv_cli.py detect photo.jpg --conf 0.4

# Classify an image, show top 5 predicted categories
python cv_cli.py classify photo.jpg --topk 5

# Segment objects (pixel masks instead of boxes)
python cv_cli.py segment photo.jpg

# Run on every image in a folder
python cv_cli.py detect ./my_photos/ --save-dir ./results

# Run on a video file
python cv_cli.py detect clip.mp4

# Live webcam (device 0) with a preview window — press 'q' to quit
python cv_cli.py detect 0 --webcam --show

# Get machine-readable output instead of printed text
python cv_cli.py detect photo.jpg --json --no-save
```

### Options

| Flag | Description |
|---|---|
| `--model NAME` | Use a different checkpoint, e.g. `yolov8s.pt` / `yolov8m.pt` for higher accuracy (slower), or your own fine-tuned `.pt` file |
| `--conf FLOAT` | Confidence threshold, 0–1 (default `0.25`) |
| `--save-dir DIR` | Where annotated output goes (default `cv_cli_output`) |
| `--no-save` | Skip writing annotated image/video files |
| `--show` | Pop up a window with results as they're processed |
| `--webcam` | Treat `source` as a webcam device index |
| `--json` | Print machine-readable JSON instead of formatted text |
| `--topk N` | (classify only) how many top predictions to print |

Annotated output is saved under `runs/<task>/<save-dir>/run/`.

## 3. Using a bigger/more accurate model

The default models (`yolov8n*.pt`, "nano") are small and fast but less
accurate. For better accuracy at the cost of speed, pass a larger variant:

```bash
python cv_cli.py detect photo.jpg --model yolov8s.pt   # small
python cv_cli.py detect photo.jpg --model yolov8m.pt   # medium
python cv_cli.py detect photo.jpg --model yolov8l.pt   # large
python cv_cli.py detect photo.jpg --model yolov8x.pt   # extra-large
```

## 4. Using your own trained model

If you've fine-tuned a YOLOv8 model on your own data (e.g. with
`ultralytics`'s training CLI), just point `--model` at your `.pt` file:

```bash
python cv_cli.py detect photo.jpg --model /path/to/my_model.pt
```

## Notes

- GPU is used automatically if PyTorch detects CUDA; otherwise it runs on CPU.
- The 80 COCO classes detection covers include things like person, car,
  dog, cat, bicycle, chair, bottle, laptop, etc. Run `python cv_cli.py detect --help`
  or check `model.names` for the full list.
