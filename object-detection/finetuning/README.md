# YOLO Fine-Tuning

Fine-tunes Ultralytics YOLO11 on our labeled basketball dataset hosted in Roboflow. The dataset is downloaded locally (no API at inference time); the trained `.pt` is then plugged into the rest of the pipeline.

## Setup

From the repository root:

```bash
uv sync --extra train
cp object-detection/.env.example .env
```

Fill in `.env`:

```
ROBOFLOW_API_KEY=<your key>
ROBOFLOW_WORKSPACE=boriss-workspace-vbstp
ROBOFLOW_PROJECT=basketball_cv_project
ROBOFLOW_VERSION=1
```

## 1. Download the dataset (YOLOv8 format)

**This probably won't work, I'd use the plain cURL I sent on discord instead**

```bash
uv run python object-detection/data.py
```

Default destination: `object-detection/data/yolov8/<project>-<version>/`. The directory contains `data.yaml`, `train/`, `valid/`, and `test/`.

To pull the same labels in COCO for RF-DETR later:

```bash
uv run python object-detection/data.py --format coco
```

## 2. Fine-tune YOLO11s

```bash
uv run python object-detection/finetuning/train.py \
  --data object-detection/data/yolov8/basketball_cv_project-1/data.yaml \
  --model yolo11s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16
```

- Device defaults to `auto`: CUDA if available, otherwise MPS, otherwise CPU.
- Outputs land in `object-detection/finetuning/runs/basketball_yolo11s/`.
- Best weights: `runs/basketball_yolo11s/weights/best.pt`.

For a sanity smoke test, drop `--epochs 80` to `--epochs 2` and `--batch` to something the machine can handle. macOS / MPS works for this; full training is best done on a CUDA GPU (Colab, university cluster).

## 3. Smoke-test the trained checkpoint

```bash
uv run python object-detection/finetuning/predict.py \
  --weights object-detection/finetuning/runs/basketball_yolo11s/weights/best.pt \
  --source object-detection/data/raw/clip_001.mp4 \
  --save
```

Annotated output lands in `object-detection/finetuning/runs/predict/exp/`.

## Notes

- `yolo11s.pt` is auto-downloaded by Ultralytics on first run; subsequent runs reuse the local copy.
- Training writes `args.yaml`, per-epoch metrics (`results.csv`), PR curves, and a confusion matrix into the run directory — useful for the report.
- Dataset structure expected by Ultralytics:

  ```
  basketball_cv_project-1/
    data.yaml
    train/{images,labels}/
    valid/{images,labels}/
    test/{images,labels}/
  ```

  This is exactly what `data.py --format yolov8` produces.

- This directory is training-only. To use the trained `best.pt` from `basketball-detect`, we still need a local YOLO backend in `src/detection.py` (parallel to `RoboflowHostedClient`). That wiring is a follow-up — see `docs/models_and_finetuning.md` Option B step "Integration into this repo".
