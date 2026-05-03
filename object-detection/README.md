# Basketball Object Detection And Tracking

This directory implements the object detection and player tracking section of the basketball computer vision pipeline:

```text
video -> Roboflow detections -> SAM2 tracking -> mask cleanup -> handoff outputs
```

The downstream team-clustering and jersey-number modules should consume the JSON, masks, and crops generated here.

## Setup

Run commands from the repository root. The root `pyproject.toml` owns dependencies and scripts.

```bash
uv sync --extra dev
cp object-detection/.env.example .env
```

Set `ROBOFLOW_API_KEY` in `.env`.

SAM2 is not bundled by PyPI in a stable way for every platform. Install it from Meta's repository or your course-provided setup, then set the checkpoint and config paths:

```bash
git clone https://github.com/facebookresearch/sam2.git object-detection/sam2
SAM2_BUILD_CUDA=0 uv pip install -e "object-detection/sam2"
mkdir -p object-detection/checkpoints
cd object-detection/sam2/checkpoints
./download_ckpts.sh
cd ../../..
cp object-detection/sam2/checkpoints/sam2.1_hiera_large.pt object-detection/checkpoints/
```

`SAM2_BUILD_CUDA=0` is recommended on macOS/MPS machines because the optional CUDA extension needs NVIDIA CUDA.

Verify the install:

```bash
uv run python -c "import sam2, torch, torchvision; print('sam2 ok'); print(torch.__version__, torchvision.__version__)"
```

Then run real SAM2 tracking:

```bash
uv run basketball-track \
  --video object-detection/data/raw/clip_001.mp4 \
  --detections object-detection/outputs/detections/clip_001_detections.json \
  --sam2-checkpoint object-detection/checkpoints/sam2.1_hiera_large.pt \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --output object-detection/outputs/tracks/clip_001_tracks.json \
  --mask-dir object-detection/outputs/masks/clip_001
```

Use `--tracker-backend mock` for smoke tests before SAM2 is installed.

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

## Outputs

- `object-detection/outputs/detections/*_detections.json`: sampled object detections.
- `object-detection/outputs/tracks/*_tracks.json`: per-frame track IDs, cleaned masks, and boxes.
- `object-detection/outputs/masks/<clip>/`: one binary PNG mask per frame and track.
- `object-detection/outputs/crops/<clip>/`: 1 FPS player crops for team clustering.
- `object-detection/outputs/reports/*_tracking_qa.json`: tracking QA counts.

## Tests

```bash
uv run --extra dev pytest
```
