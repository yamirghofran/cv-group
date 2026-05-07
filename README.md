# Basketball CV Pipeline

End-to-end basketball video analysis: player detection → SAM2 tracking → team clustering → jersey number OCR → annotated video.

## What it does

Given a short basketball clip, `demo.py` produces an annotated video where each player has:
- A coloured mask overlay (blue = Team A, red = Team B)
- A bounding box labelled with their jersey number and team

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (package manager)
- [ffmpeg](https://ffmpeg.org/download.html) — only needed to trim videos

**GPU strongly recommended for SAM2 tracking.** On CPU it takes ~12 hours for a 15-second clip. On a CUDA GPU it takes a few minutes.

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd cv-group
uv sync --extra clustering
```

### 2. Download the SAM2 checkpoint (~900 MB)

```bash
mkdir -p object-detection/checkpoints
# Linux / macOS
curl -L -o object-detection/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" `
  -OutFile "object-detection\checkpoints\sam2.1_hiera_large.pt"
```

### 3. Set up environment variables

```bash
cp .env.sample .env        # Linux / macOS
# or on Windows:
Copy-Item .env.sample .env
```

Open `.env` and replace `replace-with-your-key` with your [Roboflow API key](https://app.roboflow.com) → Settings → API Keys. The workspace and project fields are already filled in.

> The YOLO weights (`object-detection/finetuning/`) and OCR checkpoint (`numbers/models/resnet34.pth`) are already included in the repo — no extra downloads needed.

## Running the demo

### Get a video clip

Any 10–30 second basketball `.mp4` works. To download one from YouTube:

```bash
pip install yt-dlp
# download and trim to 15 seconds starting at second 10
yt-dlp -f "best[height<=720]" --output "clip.mp4" "https://www.youtube.com/watch?v=_C0o7sJ7lMA"
ffmpeg -i clip.mp4 -ss 10 -t 15 -c copy clip_short.mp4
```

### Full pipeline

```bash
uv run python demo.py \
  --video clip_short.mp4 \
  --sam2-checkpoint object-detection/checkpoints/sam2.1_hiera_large.pt
```

On Windows:

```powershell
uv run python demo.py --video clip_short.mp4 --sam2-checkpoint object-detection\checkpoints\sam2.1_hiera_large.pt
```

### Re-running without repeating slow steps

SAM2 tracking and SigLIP clustering each take several minutes. Use these flags to skip steps whose outputs already exist in `demo_output/`:

```bash
# skip both SAM2 and clustering
uv run python demo.py --video clip_short.mp4 \
  --sam2-checkpoint object-detection/checkpoints/sam2.1_hiera_large.pt \
  --skip-tracking --skip-clustering

# skip only SAM2
uv run python demo.py --video clip_short.mp4 \
  --sam2-checkpoint object-detection/checkpoints/sam2.1_hiera_large.pt \
  --skip-tracking
```

## Output

All files are written to `demo_output/` (or the directory set with `--output-dir`):

| File | Description |
|---|---|
| `annotated.mp4` | Final video with masks, boxes, jersey numbers, team labels |
| `detections.json` | YOLO detections per frame |
| `tracks.json` | SAM2 player tracks + mask paths |
| `matches.json` | IoS matches linking number detections to track IDs |
| `teams.json` | SigLIP team cluster assignments per player crop |
| `masks/` | Per-frame binary player mask images |
| `crops/` | Per-frame player crop images used for team clustering |

Terminal summary at the end:

```
── Results ──────────────────────────────
  Track  1 → Team A, jersey #23
  Track  2 → Team B, jersey #11
  Track  3 → Team A, jersey #5
  ...
```

## Pipeline overview

```
clip.mp4
  │
  ├─[1] YOLO detection      → detections.json   (players, numbers, ball, rim)
  ├─[2] SAM2 tracking       → tracks.json + masks/
  ├─[3] Crop extraction     → crops/             (one image per player per frame)
  ├─[4] Team clustering     → teams.json         (SigLIP → UMAP → KMeans)
  ├─[5] IoS matching + OCR  → matches.json       (number bbox → track_id → digit)
  └─[6] Render              → annotated.mp4
```

**IoS (Intersection over Smaller area)** is used instead of IoU to match number detections to players: it measures what fraction of the small number bounding box is covered by the (much larger) player mask, which handles partial occlusions better.

## All CLI options

```
--video               Input video path (required)
--sam2-checkpoint     SAM2 model weights (.pt) — omit to run detection only
--sam2-model-cfg      SAM2 config yaml (default: configs/sam2.1/sam2.1_hiera_l.yaml)
--ocr-checkpoint      ResNet OCR weights (default: numbers/models/resnet34.pth)
--yolo-weights        YOLO weights (default: object-detection/finetuning/.../best.pt)
--ios-threshold       Minimum IoS score to accept a match (default: 0.9)
--frame-stride        Process every N frames for detection (default: 5)
--output-dir          Where to write outputs (default: demo_output)
--device              auto | cpu | cuda | mps (default: auto)
--skip-tracking       Reuse existing tracks.json — skip SAM2
--skip-clustering     Reuse existing teams.json — skip SigLIP clustering
```
