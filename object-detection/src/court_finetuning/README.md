# Court Keypoint Fine-Tuning

Fine-tunes a local Ultralytics YOLO pose model for basketball court keypoints, then plugs the trained checkpoint into the existing court mapping pipeline.

## Setup

From the repository root:

```bash
uv sync --extra train
cp object-detection/.env.example .env
```

Set:

```text
ROBOFLOW_API_KEY=your_key_here
```

The downloader defaults to:

```text
workspace: roboflow-jvuqo
project: basketball-court-detection-2
version: 19
format: yolov11
```

## Download Dataset

```bash
uv run basketball-court-finetune-download
```

Default output:

```text
object-detection/data/court-yolo/basketball-court-detection-2-19/data.yaml
```

If Roboflow does not provide `yolov11` from the SDK, retry with:

```bash
uv run basketball-court-finetune-download --format yolov8
```

## Train

Fast baseline on a CUDA GPU:

```bash
uv run basketball-court-finetune-train \
  --data object-detection/data/court-yolo/basketball-court-detection-2-19/data.yaml \
  --model yolo11m-pose.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --patience 20 \
  --device 0 \
  --name court_yolo11m_pose_v19
```

Smoke test:

```bash
uv run basketball-court-finetune-train \
  --data object-detection/data/court-yolo/basketball-court-detection-2-19/data.yaml \
  --model yolo11m-pose.pt \
  --epochs 2 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --name smoke_court_pose
```

The default augmentations intentionally disable flips and mosaic:

```text
fliplr=0.0
flipud=0.0
mosaic=0.0
degrees=0.0
translate=0.05
scale=0.20
perspective=0.0005
```

Court keypoint IDs matter for homography, so geometric augmentations should stay conservative unless the dataset YAML has a correct `flip_idx`.

## Predict

```bash
uv run basketball-court-finetune-predict \
  --weights object-detection/court_finetuning/runs/court_yolo11m_pose_v19/weights/best.pt \
  --source object-detection/data/raw/clip_001.mp4 \
  --device 0 \
  --save
```

## Use In Court Mapping

For the shared fine-tuned checkpoint, download from Hugging Face:

```bash
mkdir -p object-detection/models/court_yolo11m_pose_v19
uvx hf download AnzeZ/basketball-court-yolo11m-pose best.pt \
  --revision 4499465f27dfa29ffa6fe621d71124394375c1d8 \
  --local-dir object-detection/models/court_yolo11m_pose_v19
```

Replace Roboflow API keypoint inference with local YOLO:

```bash
uv run basketball-court-keypoints \
  --backend yolo \
  --weights object-detection/models/court_yolo11m_pose_v19/best.pt \
  --device 0 \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/court_keypoints/clip_001_yolo_keypoints.json \
  --debug-frame-dir object-detection/outputs/debug_court_keypoints/clip_001_yolo \
  --frame-stride 15
```

Then run the existing projection and visualization commands with the new keypoint JSON.
