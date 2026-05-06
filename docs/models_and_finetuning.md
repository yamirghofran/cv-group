# Models in This Pipeline And How To Fine-Tune Them

Scope: this document covers only the detection + tracking stage that lives in `object-detection/`. Team clustering, jersey numbers, court keypoints, and shot detection (described in `docs/context.md` §3–6) are downstream and out of scope here.

---

## What models are running today

### A. Object detector — RF-DETR-S, hosted by Roboflow

**Where it lives in this repo**

- Endpoint config: `object-detection/config/default.yaml` (`roboflow.*` block)
- Defaults (also hard-coded in `RoboflowHostedSettings` at `src/roboflow_client.py:15-23`):
  - `model_id: basketball-player-detection-3-ycjdo`
  - `model_version: 13`
  - `confidence: 40`, `overlap: 30`
- HTTP call: `RoboflowHostedClient.infer_frame` at `src/roboflow_client.py:41-64` — JPEG-encodes each frame, base64s it, POSTs to `https://detect.roboflow.com/<model_id>/<version>?api_key=...`.
- Sampling cadence: frame 0 + every `detection.frame_stride` frames (default 5) — see `src/detection.py:78` and `src/detection.py:111-128`.
- Output normalization: `normalize_roboflow_predictions` (`src/roboflow_client.py:67`) converts Roboflow's center-xywh format to clamped xyxy and wraps each prediction in a `DetectionRecord` (`src/schemas.py:12`).

**Architecture (per `docs/context.md` §1)**: RF-DETR-S, DINOv2 backbone, 10 classes — `ball`, `ball-in-basket`, `number`, `player`, `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block`, `referee`, `rim`. Trained at 640×640 with resize-with-fill, no augmentation (DINOv2 already generalizes).

**How its output drives the rest of the pipeline**

1. The full per-frame detections stream is written to `outputs/detections/<clip>_detections.json` and is the only thing the next stage reads.
2. `tracking.py:70-101` (`load_frame0_boxes`) filters to **frame 0 only** and to classes in `detection.player_like_classes` (player + the player-action subclasses). Those boxes become SAM 2 prompts.
3. Everything else (referees, ball, rim, numbers, frames > 0) is currently kept in the JSON for downstream modules but is not used by tracking.

**Why this is the bottleneck for our project.** The hosted model was trained by Roboflow on the 2025 NBA Playoffs. Our footage is different (different camera angle, jersey colors, lighting, possibly amateur play). Misses on frame 0 cascade — a player not detected on frame 0 is never tracked, because there is no re-prompting mechanism (`docs/context.md` §2 explicitly flags this). The `--frame0-overrides` escape hatch papers over that but is manual.

### B. Tracker — SAM 2 (Hiera-Large)

**Where it lives in this repo**

- `src/tracking.py:142-182` (`sam2_track_masks`) — uses `sam2.build_sam2_video_predictor`, calls `add_new_points_or_box` once per frame-0 box, then `propagate_in_video` for the rest of the clip. Each per-frame mask is run through `clean_mask` in `src/mask_cleanup.py` (largest connected component + centroid distance threshold, matching `docs/context.md` §2).
- Config defaults: `tracking.sam2_checkpoint = sam2.1_hiera_large.pt`, `tracking.sam2_model_cfg = configs/sam2.1/sam2.1_hiera_l.yaml`.

**Why we are not fine-tuning SAM 2.** SAM 2 is class-agnostic and uses a temporal memory bank to re-identify objects after occlusion (`docs/context.md` §2). The published behaviour is already strong on basketball; the bottleneck cited in the blog is throughput, not accuracy. If SAM 2 underperforms on our clips, the realistic interventions are the ones called out in `docs/context.md` §0: smaller checkpoint (Hiera-S/B+), lower input resolution, or a re-prompting mechanism. Fine-tuning the segmentation backbone is a much larger project and is not what "the detector isn't good enough" usually means.

---

## Fine-tuning the detector — the two viable paths

The user-facing problem is detector quality on our footage. Two reasonable replacements for the hosted RF-DETR:

| | Option A — Fine-tune RF-DETR | Option B — Fine-tune YOLO (Ultralytics, e.g. YOLO11 or v8) |
|---|---|---|
| Matches reference pipeline | Yes (`context.md` §1) | No, but very close — same task, different architecture |
| Backbone strength out of the box | DINOv2, strong with little data | CSP / C2f, strong but more aug-hungry |
| Typical data needed for our case | ~500–1500 labeled frames | ~1000–3000 labeled frames |
| Training infra | Roboflow Train (cloud, fastest) **or** rfdetr Python package locally | Ultralytics CLI, local GPU is fine |
| License for commercial / academic | Apache-2.0 | AGPL-3.0 (Ultralytics) — fine for a course project, watch if you ever ship |
| Inference cost in this repo | Same as today (one HTTP call) **or** local PyTorch | Local PyTorch, no network |

Pick **A** if you want minimum churn — the deployment swap is changing two strings in `default.yaml`. Pick **B** if you want full local control, no API key, and faster iteration. Both are viable.

---

## Option A — Fine-tune our own RF-DETR

### A.1. Dataset

Labeling target: replicate `context.md` §1 schema as closely as makes sense.

- **Minimum useful schema** for this stage: `player`, `referee`. Drop the action subclasses (`player-jump-shot`, etc.) unless the shot-detection module needs them — they balloon labeling cost and the tracker doesn't use them.
- **Recommended schema** if we want one model to also feed shot detection later: `ball`, `ball-in-basket`, `number`, `player`, `referee`, `rim`. That's 6 classes versus the original 10 — easier to label, still covers downstream needs.
- Image size: 640×640, resize-with-fill (matches `context.md` §1 — keeps aspect ratio so player proportions don't distort).
- Augmentation: per `context.md` §1, **leave it off** for RF-DETR. The DINOv2 backbone already generalizes; aggressive aug usually hurts.
- Label format: COCO JSON if exporting from Roboflow; the `rfdetr` package also accepts COCO directly.
- Dataset size to aim for: 800–1500 annotated frames sampled at ~1 FPS from 4–8 distinct clips (different teams, lighting, camera angles).

### A.2. Training

Two paths, easiest first.

**A.2.a. Roboflow Train (cloud, no GPU needed)**

1. Upload the labeled dataset to a Roboflow workspace project.
2. Generate a version with the resize-with-fill preprocessing and no augmentations.
3. Custom Train → choose **RF-DETR-S** → init from public checkpoint (per `context.md` §1).
4. Roboflow trains, deploys, and assigns a new `model_id` and `model_version`.
5. **Swap into this repo**: change `roboflow.model_id` and `roboflow.model_version` in `object-detection/config/default.yaml`. Nothing else has to change — `RoboflowHostedClient` is generic across Roboflow detection projects.

**A.2.b. Local fine-tune with the `rfdetr` package**

```bash
uv pip install rfdetr
```

Then a minimal training script (sketch):

```python
from rfdetr import RFDETRSmall
model = RFDETRSmall(pretrain_weights="rf-detr-s.pth")
model.train(
    dataset_dir="data/labeled_coco",
    epochs=50,
    batch_size=4,
    lr=1e-4,
    output_dir="runs/rfdetr_basketball",
)
```

To use the resulting weights from this repo we'd have to move off the hosted client. Concretely, add a new backend alongside `RoboflowHostedClient` (e.g. `RFDetrLocalClient`) that loads the checkpoint once, exposes the same `infer_frame(frame, frame_name) -> dict` shape that `normalize_roboflow_predictions` already understands, and select it via a new `roboflow.backend: hosted | local` switch in `default.yaml`. Keep the JSON shape identical so `tracking.py` and the schemas don't change.

### A.3. Evaluation

- mAP@0.5 on a held-out 100–200-frame test set, per class.
- The metric that matters most for **this** repo: **frame-0 player recall**. If the detector misses no players on frame 0 of held-out clips, tracking will work without manual overrides. Build a small eval that runs the detector on frame 0 of every test clip and compares player count against ground truth.
- Track-level: feed the new detector through the existing pipeline end-to-end and check `outputs/reports/<clip>_tracking_qa.json` for `missing_track_entries` and `empty_masks` going down vs. the baseline.

### A.4. What changes downstream

- If recall on frame 0 improves, manual `--frame0-overrides` files become unnecessary.
- If we add classes (`ball`, `rim`, `number`), the detection JSON now carries them and downstream modules (jersey-number IoS pairing in `context.md` §4, shot detection in §6) can consume them later. No tracking-side change because `is_player_like_class` already filters.
- If we *drop* classes, update `detection.player_like_classes` and `detection.relevant_classes` in `default.yaml` so we don't filter against names the model never produces.

---

## Option B — Fine-tune a YOLO model

### B.1. Dataset

Same images and label policy as Option A. Format conversion: YOLO wants `.txt` per image with `class cx cy w h` normalized to [0,1]. Roboflow exports YOLO format directly; if labeling locally with CVAT/Label Studio, convert with `supervision` (already a dependency) or `pylabel`.

Recommended starting model: **YOLO11s** — small enough to train on a single consumer GPU in a few hours, big enough to outperform the hosted baseline given our domain data.

### B.2. Training

```bash
uv pip install ultralytics
yolo detect train \
  model=yolo11s.pt \
  data=data/basketball.yaml \
  imgsz=640 \
  epochs=80 \
  batch=16 \
  patience=15 \
  project=runs/yolo_basketball \
  name=v1
```

`data/basketball.yaml` looks like:

```yaml
path: ../data/labeled_yolo
train: images/train
val: images/val
test: images/test
names:
  0: player
  1: referee
  2: ball
  3: rim
  4: ball-in-basket
  5: number
```

YOLO benefits from augmentation (mosaic, HSV, flip) — leave the Ultralytics defaults on, they are tuned for detection.

### B.3. Integration into this repo

Bigger lift than Option A because we leave Roboflow entirely. Plan:

1. Add `src/yolo_client.py` with a class that mirrors `RoboflowHostedClient.infer_frame`'s signature and produces the same payload shape Roboflow returns (`{"predictions": [{"class": ..., "x": ..., "y": ..., "width": ..., "height": ..., "confidence": ...}, ...]}`). Easiest path: take Ultralytics' `Results` object and translate it. Then `normalize_roboflow_predictions` keeps working unchanged.
2. Add a `detector.backend: roboflow | yolo` switch in `default.yaml` and a couple of lines in `src/detection.py:99` to pick the client by backend. Avoid adding a parallel detection codepath — just two clients behind one interface.
3. Add `ultralytics` to `pyproject.toml` dependencies.
4. Drop the YOLO weights at `object-detection/checkpoints/yolo_basketball.pt` and reference it from `default.yaml`.

### B.4. Evaluation

Same metrics as Option A. Ultralytics also dumps per-class AP and PR curves into `runs/yolo_basketball/v1/`.

### B.5. What changes downstream

Same as Option A.

---

## What fine-tuning will and won't fix

It will help with:

- **Frame-0 recall** on our footage → fewer manual overrides → more clips usable end to end.
- **Class-level precision** for things the hosted model wasn't tuned for (e.g. our referees in different uniforms, our ball colour).
- **Domain shift** in general (jersey colours, court paint, broadcast-vs-amateur framing).

It will not help with:

- **Mid-clip player entry / re-identification.** That is a re-prompting problem on the SAM 2 side (`context.md` §2), not a detector quality problem. Any clip where a new player enters after frame 0 still needs the architectural fix.
- **Throughput / FPS.** Detector cost is small relative to SAM 2 (`context.md` §0 and §2 — SAM 2 is the 1–2 FPS bottleneck on a T4).
- **Occlusion-driven track loss inside SAM 2's window.** Detector quality doesn't enter here; SAM 2's memory bank does.

---

## Suggested next steps

1. Collect 4–8 representative clips from our actual footage and sample ~1 FPS — that gives a starting set of ~500–1500 frames.
2. Label `player` + `referee` only at first. We can extend the schema later if the downstream teams need ball / rim / number.
3. Start with **Option A via Roboflow Train** because the deployment swap is two strings. Treat it as the baseline.
4. Measure frame-0 player recall on held-out clips and `missing_track_entries` from the tracking QA reports vs. the current hosted model.
5. If we still want full local control or hit Roboflow quota limits, port to **Option B (YOLO11s)** behind a `detector.backend` switch — the JSON contract stays the same so nothing downstream changes.
