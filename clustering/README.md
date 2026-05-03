# Clustering — Team Assignment

Stage 2 of the basketball CV pipeline. Takes per-track player crops produced by
the `object-detection` stage and assigns each crop to a team using unsupervised
clustering, so the OCR stage can scope jersey-number lookups by team.

```
object-detection/outputs/crops/<clip>/   ──>   clustering   ──>   clustering/outputs/teams/<clip>_teams.json
```

## Status

Working baseline.

| Module | What it does | Status |
| --- | --- | --- |
| `clustering.classifier` | `TeamClassifier`: SigLIP + UMAP(3D) + KMeans(k=2), vendored from `roboflow/sports@feat/basketball` | Implemented |
| `clustering.schemas` | Pydantic `TeamAssignment` / `TeamOutput` (mirrors `object-detection/src/schemas.py` style) | Implemented |
| `clustering.load_crops` | Reads PNG/JPEG crops from a directory into BGR numpy arrays | Implemented |
| `clustering.cli` — `basketball-team-cluster` | End-to-end: load crops → fit + predict → write JSON | Implemented |
| `clustering.cli` — `basketball-team-bench` | Multi-method ablation runner (purity / ARI / flip-rate) | Stub |
| Embedder ablations (DINOv2, HSV histogram) | Alternatives to SigLIP for the comparison table | Not yet |
| Eval harness | `purity`, `adjusted_rand`, `flip_rate`, `benchmark` | Not yet |
| Cluster → team-name automation | Mean cluster color matched to known team hex | Not yet |

## Setup

From repo root:

```bash
uv sync --extra clustering
```

First run downloads the SigLIP vision model from Hugging Face (~400 MB), cached
in `~/.cache/huggingface/`.

## Run

End-to-end on the synthetic fixture crops checked into the repo (no external
data required, fast smoke test of the full pipeline):

```bash
uv run basketball-team-cluster \
  --crops-dir clustering/tests/fixtures/sample_crops \
  --output clustering/outputs/teams/fixtures_teams.json
```

End-to-end on real player crops produced by the `object-detection` stage's
`basketball-export-crops` CLI:

```bash
uv run basketball-team-cluster \
  --crops-dir object-detection/outputs/crops/clip_001 \
  --output clustering/outputs/teams/clip_001_teams.json
```

Optional flags: `--device cuda` for GPU SigLIP, `--batch-size N` (default 32),
`--method NAME` (recorded in the output JSON; only `siglip+umap+kmeans` is
implemented today).

## Output format

`TeamOutput` JSON, one record per crop in `assignments`:

```json
{
  "crops_dir": "object-detection/outputs/crops/clip_001",
  "classifier_method": "siglip+umap+kmeans",
  "n_teams": 2,
  "cluster_to_team_name": {},
  "assignments": [
    {"crop_path": ".../crop_001.png", "cluster_id": 0, "track_id": null, "team_name": null},
    {"crop_path": ".../crop_002.png", "cluster_id": 1, "track_id": null, "team_name": null}
  ]
}
```

`cluster_to_team_name` and per-assignment `team_name` are populated once a
human (or later, an automated color-matching heuristic) decides which raw
cluster id corresponds to which team.

## Tests

```bash
uv run pytest clustering/tests
```

Tests use synthetic fixtures (red-dominant vs blue-dominant 32×32 crops) and a
stub classifier so they run in milliseconds without loading SigLIP. Real-model
integration tests against actual basketball footage will land separately.
