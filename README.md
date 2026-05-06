# cv-group

Basketball video analysis pipeline — detect players, track them across frames, and cluster them into teams.

## Pipeline

```
Video
  │
  ├─ 1. YOLO Detection        → detections.json   (per-frame bounding boxes)
  ├─ 2. SAM2 Tracking         → tracks.json       (per-frame masks + track IDs)
  ├─ 3. Crop Export           → crops/track_*/    (sampled player images)
  ├─ 4. Team Clustering       → teams.json        (SigLIP → UMAP → KMeans)
  └─ 5. Annotated Video       → annotated.mp4     (team-coloured overlays)
```

| Stage | Model | Output |
|---|---|---|
| Detection | YOLO (fine-tuned) or Roboflow RF-DETR | Per-frame player/ball/referee bounding boxes |
| Tracking | SAM 2 | Per-player masks propagated across all frames |
| Team Clustering | SigLIP + UMAP + KMeans | Each track labelled Team A / Team B |

## Project Structure

```
object-detection/     Detection, tracking, crop export, visualization
  src/
    detection.py        Run YOLO/Roboflow on video frames
    tracking.py         SAM2 mask tracking from frame-0 boxes
    export_crops.py     Sample player crops from tracked videos
    visualize_tracks.py Render annotated video with masks + team labels
    schemas.py          Pydantic models (DetectionOutput, TrackOutput, …)
clustering/           Team identification
  classifier.py        SigLIP → UMAP → KMeans pipeline
  cli.py               CLI entry point
  assign.py            Majority-vote: per-crop clusters → per-track team
  load_crops.py        Load crops from flat or track_*/ directories
  schemas.py           Pydantic models (TeamOutput, TeamAssignment)
api/                  FastAPI service
  app.py               Upload video → run full pipeline → download result
  pipeline.py          Orchestrates all stages
  config.py            Environment-based settings
```

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# Install all dependencies
uv sync

# With optional stages
uv sync --extra clustering    # team clustering (scikit-learn, transformers, umap-learn)
uv sync --extra sam2          # SAM2 tracking
uv sync --extra train         # Roboflow dataset download
uv sync --extra dev           # pytest

# Environment variables (optional — defaults work out of the box)
cp .env.sample .env           # edit as needed
```

## CLI Usage

Each stage can be run independently:

```bash
# Detection
basketball-detect --video clip.mp4 --output detections.json

# Tracking (requires detection output)
basketball-track --video clip.mp4 --detections detections.json

# Team clustering (requires crop export from tracking)
basketball-team-cluster --crops-dir crops/clip/ --output teams.json

# Visualize with team colours
basketball-visualize-tracks --video clip.mp4 --tracks tracks.json --teams teams.json --output annotated.mp4
```

## API

```bash
uv run basketball-api
# POST /process  — upload a video, get back an annotated mp4
# GET  /healthz  — service status
```

## Tests

```bash
uv run pytest
```