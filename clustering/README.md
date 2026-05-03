# Clustering — Team Assignment

Stage 2 of the basketball CV pipeline. Takes per-track player crops produced by
the `object-detection` stage and assigns each track to a team using unsupervised
clustering, so the OCR stage can scope jersey-number lookups by team.

```
object-detection/outputs/crops/<clip>/   ──>   clustering   ──>   clustering/outputs/teams/<clip>_teams.json
```

## Status

Scaffold only. Real implementation lands across the next few commits:

- Pydantic schemas for `TeamAssignment` / `TeamOutput` (matching `object-detection/src/schemas.py` style)
- `load_crops` adapter that reads from `object-detection/outputs/crops/<clip>/`
- Baseline `TeamClassifier` (SigLIP + UMAP + KMeans, vendored from `roboflow/sports`)
- Ablation embedders (DINOv2, HSV histogram)
- Eval harness (purity, ARI, per-track flip rate, latency)
- CLIs: `basketball-team-cluster`, `basketball-team-bench`

## Setup

From repo root:

```bash
uv sync --extra clustering
```

## Run (when implemented)

```bash
uv run basketball-team-cluster \
  --crops-dir object-detection/outputs/crops/clip_001 \
  --output clustering/outputs/teams/clip_001_teams.json
```

## Tests

```bash
uv run --extra dev pytest clustering/tests
```
