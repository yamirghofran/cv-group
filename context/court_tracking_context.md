# Court Tracking And 2D Court Mapping Context

This context file is for an agent running this project on another machine, especially an NVIDIA GPU machine, to test the full object detection, SAM2 tracking, ball detection, court keypoint detection, and 2D court movement visualization pipeline.

## Current Branch

Use the `court-mapping` branch.

```bash
git clone https://github.com/yamirghofran/cv-group.git
cd cv-group
git checkout court-mapping
```

The relevant implementation lives under `object-detection/`. The root `pyproject.toml` owns the Python package, dependencies, and CLI scripts.

## Goal Of This Section

Input:

```text
object-detection/data/raw/clip_001.mp4
```

Output:

```text
Roboflow basketball detections
SAM2 player tracks and masks
Roboflow court keypoints
smoothed ball detections
2D projected player and ball court coordinates
side-by-side source video + top-down court movement video
```

This builds on the earlier object detection/tracking work. It does not do shot detection, team clustering, jersey number recognition, or final player identity resolution.

## Environment

Use Python 3.11. The repo is configured for:

```text
requires-python = >=3.11,<3.13
```

Install project dependencies from the repo root:

```bash
uv sync --extra dev
```

Set the Roboflow key:

```bash
cp object-detection/.env.example .env
```

Then edit `.env`:

```text
ROBOFLOW_API_KEY=your_key_here
```

## SAM2 Setup On NVIDIA GPU

Install SAM2 into `third_party/sam2`. On an NVIDIA machine, do not use the macOS-only `SAM2_BUILD_CUDA=0` setting unless CUDA build issues force you to.

```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
uv pip install -e "third_party/sam2"
mkdir -p object-detection/checkpoints
cd third_party/sam2/checkpoints
./download_ckpts.sh
cd ../../..
cp third_party/sam2/checkpoints/sam2.1_hiera_large.pt object-detection/checkpoints/
```

Verify SAM2 and PyTorch:

```bash
uv run python -c "import sam2, torch, torchvision; print('sam2 ok'); print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
```

Expected on NVIDIA:

```text
sam2 ok
...
True
```

If CUDA is unavailable, check the installed PyTorch build and NVIDIA driver/CUDA runtime. The pipeline can run on CPU, but it will be much slower.

## Important SAM2 Path Rule

Do not clone SAM2 into `object-detection/sam2`. Running Python from the parent of a folder named `sam2` can shadow the installed package and trigger this error:

```text
RuntimeError: You're likely running Python from the parent directory of the sam2 repository...
```

Use `third_party/sam2` as shown above.

## Full Pipeline Commands

Run commands from the repo root.

### 1. Detect Basketball Objects

This uses the Roboflow basketball detector configured in `object-detection/config/default.yaml`.

```bash
uv run basketball-detect \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/detections/clip_001_detections.json \
  --debug-video object-detection/outputs/videos/clip_001_detections.mp4 \
  --frame-stride 5
```

The detector keeps all relevant classes, including `player`, player-action classes, `number`, `ball`, `referee`, and `rim`. SAM2 tracking uses only player-like classes from frame 0.

### 2. Track Players With SAM2

For the fastest full-video test on NVIDIA, start with the tiny SAM2 checkpoint if available:

```bash
cp third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt object-detection/checkpoints/
```

Then run:

```bash
uv run basketball-track \
  --video object-detection/data/raw/clip_001.mp4 \
  --detections object-detection/outputs/detections/clip_001_detections.json \
  --tracker-backend sam2 \
  --sam2-checkpoint object-detection/checkpoints/sam2.1_hiera_tiny.pt \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --output object-detection/outputs/tracks/clip_001_sam2_tiny_full_tracks.json \
  --mask-dir object-detection/outputs/masks/clip_001_sam2_tiny_full \
  --crop-dir object-detection/outputs/crops/clip_001_sam2_tiny_full \
  --qa-report object-detection/outputs/reports/clip_001_sam2_tiny_full_tracking_qa.json
```

For better quality, use the large model:

```bash
uv run basketball-track \
  --video object-detection/data/raw/clip_001.mp4 \
  --detections object-detection/outputs/detections/clip_001_detections.json \
  --tracker-backend sam2 \
  --sam2-checkpoint object-detection/checkpoints/sam2.1_hiera_large.pt \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --output object-detection/outputs/tracks/clip_001_sam2_large_full_tracks.json \
  --mask-dir object-detection/outputs/masks/clip_001_sam2_large_full \
  --crop-dir object-detection/outputs/crops/clip_001_sam2_large_full \
  --qa-report object-detection/outputs/reports/clip_001_sam2_large_full_tracking_qa.json
```

### 3. Visualize SAM2 Tracks

```bash
uv run basketball-visualize-tracks \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_sam2_tiny_full_tracks.json \
  --output object-detection/outputs/videos/clip_001_sam2_tiny_full_tracks.mp4
```

Open the output video and check:

- Track IDs are stable.
- Masks are on players, not background.
- Frame 0 has all intended players.
- There are no obvious empty-mask failures.

### 4. Detect Court Keypoints

This uses Roboflow `basketball-court-detection-2`, currently defaulting to version `19` in config.

```bash
uv run basketball-court-keypoints \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/court_keypoints/clip_001_keypoints.json \
  --debug-frame-dir object-detection/outputs/debug_court_keypoints/clip_001 \
  --frame-stride 15
```

Inspect the debug frames in:

```text
object-detection/outputs/debug_court_keypoints/clip_001/
```

The keypoints should sit on court line intersections. If they are poor, the homography and 2D projection will also be poor.

### 5. Detect And Smooth Ball Movement

```bash
uv run basketball-ball-detect \
  --video object-detection/data/raw/clip_001.mp4 \
  --output object-detection/outputs/ball/clip_001_ball.json \
  --frame-stride 1
```

This calls the existing basketball object detector every frame by default, keeps the best `ball` detection, rejects large jumps, interpolates short missing gaps, and applies a small moving average.

### 6. Project Players And Ball To 2D Court Coordinates

Use whichever track file you generated. This example uses the tiny full-video run:

```bash
uv run basketball-project-court \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_sam2_tiny_full_tracks.json \
  --court-keypoints object-detection/outputs/court_keypoints/clip_001_keypoints.json \
  --ball object-detection/outputs/ball/clip_001_ball.json \
  --output object-detection/outputs/court_tracks/clip_001_court_tracks.json \
  --qa-report object-detection/outputs/reports/clip_001_court_mapping_qa.json
```

Projection details:

- Player location is the bottom-center of each tracked player bbox.
- Ball location is the center of the detected ball bbox.
- Homographies are computed from sampled court keypoint frames with OpenCV RANSAC.
- Each track frame uses the nearest sampled homography.
- Output court coordinates are in feet on an NBA court: `94 x 50`.

### 7. Render Side-By-Side Court Movement Video

```bash
uv run basketball-visualize-court \
  --video object-detection/data/raw/clip_001.mp4 \
  --tracks object-detection/outputs/tracks/clip_001_sam2_tiny_full_tracks.json \
  --court-tracks object-detection/outputs/court_tracks/clip_001_court_tracks.json \
  --output object-detection/outputs/videos/clip_001_2d_court_movement.mp4
```

Expected output:

```text
object-detection/outputs/videos/clip_001_2d_court_movement.mp4
```

Left side: original video with masks/IDs.

Right side: top-down NBA court with player trails and ball trail.

## Main Files To Inspect

```text
object-detection/src/detection.py
object-detection/src/tracking.py
object-detection/src/mask_cleanup.py
object-detection/src/visualize_tracks.py
object-detection/src/court_geometry.py
object-detection/src/court_keypoints.py
object-detection/src/ball_tracking.py
object-detection/src/project_to_court.py
object-detection/src/visualize_court.py
object-detection/src/schemas.py
object-detection/config/default.yaml
```

## Tests

Run:

```bash
uv run --extra dev pytest
```

Current expected result:

```text
19 passed
```

The tests include mocked Roboflow/SAM2 smoke coverage, so they do not need real API calls or GPU.

## Known Pitfalls

- Roboflow API inference is cloud-hosted. You need `ROBOFLOW_API_KEY`.
- Full-frame ball detection with `--frame-stride 1` can make many Roboflow API calls.
- SAM2 tracking depends heavily on good frame-0 player boxes.
- If frame 0 misses a player, create a correction JSON and pass `--frame0-overrides`.
- Long clips with players entering/leaving may need re-prompting; the current implementation assumes the short-clip setup where visible players are initialized from frame 0.
- Court mapping quality depends on camera view and court keypoint quality. Broadcast camera cuts or very oblique views can break homography.
- The court keypoint class IDs are assumed to match the configured 33-point court geometry. If a different Roboflow model version changes keypoint ordering, update `class_id_offset` or the geometry mapping.

## Good Acceptance Checks

- `clip_001_detections.mp4` shows reasonable player, number, ball, rim, and referee boxes.
- `clip_001_sam2_*_tracks.mp4` shows stable player IDs and clean masks.
- `clip_001_keypoints.json` has enough valid sampled frames.
- Court keypoint debug images show points on real court intersections.
- `clip_001_court_mapping_qa.json` has nonzero homographies and low invalid point counts.
- `clip_001_2d_court_movement.mp4` shows plausible player and ball movement on the 2D court.

