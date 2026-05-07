# Basketball CV Pipeline

End-to-end basketball video analysis: player detection → SAM2 tracking → team clustering → jersey number OCR → annotated video + 2D court minimap.

## What it does

Given a short basketball clip, `demo.py` produces:
- **`annotated_tracks.mp4`** — each player has a coloured mask overlay (Team A / Team B), a bounding box, and their jersey number
- **`annotated_court.mp4`** — side-by-side view: broadcast video + 2D NBA court minimap showing player movement and ball trail *(requires court-keypoint weights — see step 3)*

**GPU strongly recommended for SAM2 tracking.** On CPU it takes ~12 hours for a 15-second clip. On a CUDA GPU it takes a few minutes.

---

## Setup (do this once)

### Step 1 — Install uv

[uv](https://docs.astral.sh/uv/getting-started/installation/) is the package manager used by this project.

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2 — Clone and install dependencies

```bash
git clone <repo-url>
cd cv-group
uv sync --extra clustering --extra sam2
```

> The YOLO detection weights (`object-detection/finetuning/`) and OCR checkpoint (`numbers/models/resnet34.pth`) are already in the repo — no extra downloads needed for those.

### Step 3 — Download the SAM2 checkpoint (~900 MB)

SAM2 is used for player tracking. This file is not in the repo because it is too large.

```bash
# Linux / macOS
mkdir -p object-detection/checkpoints
curl -L -o object-detection/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Windows (PowerShell)
New-Item -ItemType Directory -Force object-detection\checkpoints
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" `
  -OutFile "object-detection\checkpoints\sam2.1_hiera_large.pt"
```

### Step 4 — Set up the environment file

```bash
# Linux / macOS
cp .env.sample .env

# Windows (PowerShell)
Copy-Item .env.sample .env
```

Open `.env` and replace `replace-with-your-key` with your [Roboflow API key](https://app.roboflow.com) → Settings → API Keys.

> **Note:** The demo pipeline uses local YOLO models for detection, ball tracking, and court keypoints — no Roboflow API calls are made at inference time. The API key is only required if you use the Roboflow-backed CLI tools directly.

### Step 5 — Get a video clip

Any 10–30 second basketball `.mp4` works. To download one from YouTube:

```bash
pip install yt-dlp
yt-dlp -f "best[height<=720]" --output "clip.mp4" "https://www.youtube.com/watch?v=_C0o7sJ7lMA"
```

Trim to 15 seconds starting at second 10 (requires [ffmpeg](https://ffmpeg.org/download.html)):

```bash
ffmpeg -i clip.mp4 -ss 10 -t 15 -c copy clip_short.mp4
```

---

## Running the demo

```bash
uv run python demo.py --video clip_short.mp4
```

That's it. The script detects the SAM2 checkpoint automatically and runs all 9 pipeline stages. If the checkpoint is missing it prints the download command and runs detection only.

**On Windows (PowerShell):**

```powershell
uv run python demo.py --video clip_short.mp4
```

### Re-running without repeating slow steps

SAM2 tracking and SigLIP clustering each take several minutes. Use these flags to skip steps whose outputs already exist in `demo_output/`:

```bash
# Skip SAM2 and clustering — re-run only OCR and rendering (fastest)
uv run python demo.py --video clip_short.mp4 --skip-tracking --skip-clustering

# Skip SAM2 only
uv run python demo.py --video clip_short.mp4 --skip-tracking

# Skip ball tracking and court minimap as well
uv run python demo.py --video clip_short.mp4 --skip-tracking --skip-clustering --skip-court
```

---

## Output

All files are written to `demo_output/` (or the directory set with `--output-dir`):

| File | Description |
|---|---|
| `annotated_tracks.mp4` | Broadcast view: team-coloured masks + jersey numbers |
| `annotated_court.mp4` | Side-by-side: broadcast + 2D court minimap *(if court weights present)* |
| `detections.json` | YOLO detections per frame |
| `tracks.json` | SAM2 player tracks + mask paths (enriched with team names) |
| `ball.json` | Ball position per frame (YOLO, every frame) |
| `teams.json` | SigLIP team cluster assignments per player crop |
| `jersey_numbers.json` | IoS + OCR matches per frame (track → jersey number) |
| `court_keypoints.json` | Detected court keypoints per frame |
| `court_tracks.json` | Player + ball positions projected to 2D court coordinates |
| `court_mapping_qa.json` | Homography / projection diagnostics |
| `masks/` | Per-frame binary player mask images |
| `crops/` | Per-frame player crop images used for team clustering |

Terminal output at the end:

```
Pipeline complete. Artifacts:
           detections: demo_output/detections.json
                 ball: demo_output/ball.json
               tracks: demo_output/tracks.json
                 ...
```

---

## Pipeline overview

```
clip.mp4
  │
  ├─[1] YOLO detection        → detections.json         (players, numbers, ball, rim)
  ├─[2] Ball tracking (YOLO)  → ball.json               (every frame, smoothed)
  ├─[3] SAM2 tracking         → tracks.json + masks/
  ├─[4] Crop export           → crops/                  (1 FPS player crops)
  ├─[5] Team clustering       → teams.json              (SigLIP → UMAP → KMeans)
  ├─[6] OCR + IoS matching    → jersey_numbers.json     (number bbox → track → digit)
  ├─[7] Court keypoints       → court_keypoints.json    (fine-tuned YOLO11m-pose)
  ├─[8] Court projection      → court_tracks.json       (2D NBA court coordinates)
  └─[9] Render
        ├─ annotated_tracks.mp4   (broadcast view with team colours + jersey numbers)
        └─ annotated_court.mp4    (side-by-side minimap)
```

**IoS (Intersection over Smaller area)** is used instead of IoU to match jersey-number detections to player masks: it measures what fraction of the small number bounding box overlaps the (much larger) player mask, which handles partial occlusions better.

---

## All CLI options

```
--video               Input video path (required)
--sam2-checkpoint     SAM2 model weights (.pt) — default: object-detection/checkpoints/sam2.1_hiera_large.pt
--sam2-model-cfg      SAM2 config yaml (default: from config file)
--yolo-weights        YOLO detection weights (default: from config)
--court-weights       Court-keypoint YOLO weights — if absent, court minimap is skipped automatically
--ocr-model           OCR model: resnet34 | resnet18 | smolvlm2 | stacking (default: resnet34)
--ios-threshold       Minimum IoS score to accept a match (default: 0.9)
--frame-stride        Process every N frames for detection (default: 5)
--output-dir          Where to write outputs (default: demo_output)
--device              cpu | cuda | mps (default: auto-detected)
--skip-tracking       Reuse existing tracks.json — skip SAM2
--skip-clustering     Reuse existing teams.json — skip SigLIP clustering
--skip-court          Skip ball tracking, court keypoints, and court minimap video
```
