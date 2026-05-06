# Basketball Object Detection And Tracking

This directory implements the object detection, player tracking, and 2D court movement section of the basketball computer vision pipeline:

```text
video -> Roboflow detections -> SAM2 tracking -> court keypoints -> 2D court movement
```

The downstream team-clustering and jersey-number modules should consume the JSON, masks, and crops generated here.

## Setup

Run commands from the repository root. The root `pyproject.toml` owns dependencies and scripts.

```bash
uv sync --extra dev
cp object-detection/.env.example .env
```

Set `ROBOFLOW_API_KEY` in `.env`.

SAM2 is not bundled by PyPI in a stable way for every platform. It's declared as the `sam2` extra in `pyproject.toml` (sourced from Meta's GitHub repo via `[tool.uv.sources]`), so a single sync installs the Python package:

```bash
SAM2_BUILD_CUDA=0 uv sync --extra dev --extra sam2
```

Pass every extra you need on each `uv sync` — the resolver uninstalls anything you omit.

`SAM2_BUILD_CUDA=0` is recommended on macOS/MPS machines because the optional CUDA extension needs NVIDIA CUDA. On a CUDA host, drop the env var.

The `sam2` package does not ship the model weights. Clone the upstream repo once just for its checkpoint script and copy the file into the checkpoints dir:

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
mkdir -p object-detection/checkpoints
cd third_party/sam2/checkpoints
./download_ckpts.sh
cd ../../..
cp third_party/sam2/checkpoints/sam2.1_hiera_large.pt object-detection/checkpoints/
```

Verify the install:

```bash
uv run python -c "import sam2, torch, torchvision; print('sam2 ok'); print(torch.__version__, torchvision.__version__)"
```

Use `--tracker-backend mock` for smoke tests before SAM2 is installed.

## Quick Run: Tracking Only

This path produces object detections, SAM2 player tracks/masks, player crops, a tracking QA report, and an annotated tracking video. It does not produce the 2D court view.

```bash
uv run basketball-detect \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/detections/clip_001_detections.json \
  --debug-video object-detection/outputs/videos/clip_001_detections.mp4 \
  --frame-stride 5
```

```bash
uv run basketball-track \
  --video object-detection/data/raw/clip_001.mp4 \
  --detections object-detection/outputs/detections/clip_001_detections.json \
  --sam2-checkpoint object-detection/checkpoints/sam2.1_hiera_large.pt \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --output object-detection/outputs/tracks/clip_001_tracks.json \
  --mask-dir object-detection/outputs/masks/clip_001 \
  --crop-dir object-detection/outputs/crops/clip_001 \
  --qa-report object-detection/outputs/reports/clip_001_tracking_qa.json
```

```bash
uv run basketball-visualize-tracks \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_tracks.json \
  --output object-detection/outputs/videos/clip_001_tracks.mp4
```

The main visual output is:

```text
object-detection/outputs/videos/clip_001_tracks.mp4
```

## Quick Run: Tracking + 2D Court View

Run the tracking-only commands first. Then add ball detection, court keypoint detection, court projection, and the side-by-side court visualization.

To use the fine-tuned local court model, download it once:

```bash
mkdir -p object-detection/models/court_yolo11m_pose_v19
uvx hf download AnzeZ/basketball-court-yolo11m-pose best.pt \
  --revision 4499465f27dfa29ffa6fe621d71124394375c1d8 \
  --local-dir object-detection/models/court_yolo11m_pose_v19
```

```bash
uv run basketball-court-keypoints \
  --backend yolo \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/court_keypoints/clip_001_yolo_keypoints.json \
  --debug-frame-dir object-detection/outputs/debug_court_keypoints/clip_001_yolo \
  --frame-stride 15
```

```bash
uv run basketball-ball-detect \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/ball/clip_001_ball.json \
  --frame-stride 1
```

```bash
uv run basketball-project-court \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_tracks.json \
  --court-keypoints object-detection/outputs/court_keypoints/clip_001_yolo_keypoints.json \
  --ball object-detection/outputs/ball/clip_001_ball.json \
  --output object-detection/outputs/court_tracks/clip_001_court_tracks.json \
  --qa-report object-detection/outputs/reports/clip_001_court_mapping_qa.json
```

```bash
uv run basketball-visualize-court \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_tracks.json \
  --court-tracks object-detection/outputs/court_tracks/clip_001_court_tracks.json \
  --output object-detection/outputs/videos/clip_001_2d_court_movement.mp4
```

The main court-view output is:

```text
object-detection/outputs/videos/clip_001_2d_court_movement.mp4
```

## Detection

```bash
uv run basketball-detect \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/detections/clip_001_detections.json \
  --debug-video object-detection/outputs/videos/clip_001_detections.mp4 \
  --frame-stride 5
```

The detector runs on frame `0` and every `frame_stride` frames. It saves all returned predictions, including player, number, ball, referee, and rim classes.

## Optional Frame-0 Corrections

If frame `0` misses a player, create a correction file:

```json
{
  "video": "object-detection/data/raw/clip_001.mp4",
  "frame_id": 0,
  "boxes": [
    {
      "track_hint": "missing left corner player",
      "bbox_xyxy": [100, 80, 180, 310],
      "class_name": "player",
      "note": "manually added"
    }
  ]
}
```

Pass it to tracking:

```bash
uv run basketball-track \
  --video object-detection/data/raw/clip_001.mp4 \
  --detections object-detection/outputs/detections/clip_001_detections.json \
  --frame0-overrides object-detection/outputs/corrections/clip_001_frame0_boxes.json \
  --output object-detection/outputs/tracks/clip_001_tracks.json \
  --mask-dir object-detection/outputs/masks/clip_001
```

If the override file exists and has boxes, those boxes replace detector frame-0 boxes.

## Visualization

```bash
uv run basketball-visualize-tracks \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_tracks.json \
  --output object-detection/outputs/videos/clip_001_tracks.mp4
```

## 2D Court Movement Mapping

This extension projects tracked players and ball movement onto a top-down NBA court. It uses Roboflow's `basketball-court-detection-2` court keypoint model as a calibration source, then computes homographies with OpenCV. The default config currently targets version `19`; pass `--model-version` to pin a different Roboflow version.

For local, non-API court keypoint inference, download the fine-tuned YOLO11m pose checkpoint from Hugging Face:

```bash
mkdir -p object-detection/models/court_yolo11m_pose_v19
uvx hf download AnzeZ/basketball-court-yolo11m-pose best.pt \
  --revision 4499465f27dfa29ffa6fe621d71124394375c1d8 \
  --local-dir object-detection/models/court_yolo11m_pose_v19
```

Model page: https://huggingface.co/AnzeZ/basketball-court-yolo11m-pose

Then use `--backend yolo` for court keypoint detection:

```bash
uv run basketball-court-keypoints \
  --backend yolo \
  --weights object-detection/models/court_yolo11m_pose_v19/best.pt \
  --device auto \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/court_keypoints/clip_001_yolo_keypoints.json \
  --debug-frame-dir object-detection/outputs/debug_court_keypoints/clip_001_yolo \
  --frame-stride 15
```

If you omit `--weights`, the config default points at `object-detection/models/court_yolo11m_pose_v19/best.pt`.

Run court keypoint detection every 15 frames:

```bash
uv run basketball-court-keypoints \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/court_keypoints/clip_001_keypoints.json \
  --frame-stride 15
```

Run ball detection every frame and smooth short gaps:

```bash
uv run basketball-ball-detect \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/ball/clip_001_ball.json \
  --frame-stride 1
```

Project tracked players and ball to court coordinates:

```bash
uv run basketball-project-court \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_sam2_tiny_full_tracks.json \
  --court-keypoints object-detection/outputs/court_keypoints/clip_001_keypoints.json \
  --ball object-detection/outputs/ball/clip_001_ball.json \
  --output object-detection/outputs/court_tracks/clip_001_court_tracks.json \
  --qa-report object-detection/outputs/reports/clip_001_court_mapping_qa.json
```

Render a side-by-side video:

```bash
uv run basketball-visualize-court \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_sam2_tiny_full_tracks.json \
  --court-tracks object-detection/outputs/court_tracks/clip_001_court_tracks.json \
  --output object-detection/outputs/videos/clip_001_2d_court_movement.mp4
```

The court projection uses the bottom-center of each tracked player bbox as the foot location. Ball movement uses the detected ball bbox center, with short missing gaps interpolated.

## Local Court Keypoint Fine-Tuning

The court keypoint API can be replaced by a local YOLO pose checkpoint trained on `basketball-court-detection-2`.

Install train dependencies:

```bash
uv sync --extra train
```

Download Roboflow's court keypoint dataset:

```bash
uv run basketball-court-finetune-download
```

Train the recommended baseline:

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

Run local YOLO court keypoints in the existing mapping pipeline:

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

See `object-detection/src/court_finetuning/README.md` for smoke tests and prediction commands.

## Outputs

- `object-detection/outputs/detections/*_detections.json`: sampled object detections.
- `object-detection/outputs/tracks/*_tracks.json`: per-frame track IDs, cleaned masks, and boxes.
- `object-detection/outputs/masks/<clip>/`: one binary PNG mask per frame and track.
- `object-detection/outputs/crops/<clip>/`: 1 FPS player crops for team clustering.
- `object-detection/outputs/court_keypoints/*_keypoints.json`: sampled court keypoints.
- `object-detection/outputs/ball/*_ball.json`: smoothed ball detections.
- `object-detection/outputs/court_tracks/*_court_tracks.json`: players and ball in court coordinates.
- `object-detection/outputs/reports/*_tracking_qa.json`: tracking QA counts.
- `object-detection/outputs/reports/*_court_mapping_qa.json`: court homography and projection QA counts.

## Tests

```bash
uv run --extra dev pytest
```
