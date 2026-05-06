"""CLI entry points for the clustering stage.

Registered in the root pyproject as:
    basketball-team-cluster = "clustering.cli:cluster_main"
    basketball-team-bench   = "clustering.cli:bench_main"
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from clustering.load_crops import load_track_crops
from clustering.schemas import TeamAssignment, TeamOutput

DEFAULT_METHOD = "siglip+umap+kmeans"


class _ClassifierLike(Protocol):
    def fit(self, crops: list[np.ndarray]) -> None: ...
    def predict(self, crops: list[np.ndarray]) -> np.ndarray: ...
    def fit_predict(self, crops: list[np.ndarray]) -> np.ndarray: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="basketball-team-cluster",
        description=(
            "Fit a team classifier on a directory of player crops and write "
            "per-crop team assignments as JSON."
        ),
    )
    parser.add_argument(
        "--crops-dir",
        type=Path,
        required=True,
        help="Directory of PNG/JPEG player crops to fit and predict on.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the TeamOutput JSON file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device for SigLIP feature extraction (default: cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SigLIP feature extraction (default: 32).",
    )
    parser.add_argument(
        "--n-teams",
        type=int,
        default=2,
        help=(
            "Number of clusters to fit (default: 2). Use 3 to set up the "
            "referee-rejection ablation."
        ),
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help=(
            "Method identifier recorded in the output JSON. Currently only "
            f"{DEFAULT_METHOD!r} is implemented; the flag exists so future "
            "ablation methods can override it."
        ),
    )
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    classifier_factory: Callable[[], _ClassifierLike] | None = None,
) -> TeamOutput:
    """Load crops, fit + predict, write JSON, return the TeamOutput.

    `classifier_factory` is injected for tests so we can avoid loading SigLIP.
    In production it defaults to constructing a TeamClassifier.

    Supports both flat crop directories and hierarchical ``track_*/``
    sub-directories.  When track sub-directories are found, ``track_id`` is
    parsed from the directory name and populated on each ``TeamAssignment``.
    ``cluster_to_team_name`` is auto-populated as ``{0: "Team A", 1: "Team B",
    ...}``.
    """
    triples = load_track_crops(args.crops_dir)
    if not triples:
        raise ValueError(f"No crops found in {args.crops_dir}")

    paths, images, track_ids = zip(*triples, strict=True)
    images_list = list(images)
    track_id_list = list(track_ids)

    if len(images_list) < args.n_teams:
        raise ValueError(
            f"Need at least {args.n_teams} crops for n_teams={args.n_teams}; "
            f"found {len(images_list)} in {args.crops_dir}"
        )

    if classifier_factory is None:
        # Local import keeps the CLI module importable without transformers.
        from clustering.classifier import TeamClassifier

        classifier_factory = lambda: TeamClassifier(  # noqa: E731
            device=args.device,
            batch_size=args.batch_size,
            n_teams=args.n_teams,
        )

    classifier = classifier_factory()
    cluster_ids = classifier.fit_predict(images_list)

    # Auto-assign human-readable team names: cluster 0 → "Team A", 1 → "Team B", …
    cluster_to_team_name = {
        cid: f"Team {chr(65 + cid)}" for cid in range(args.n_teams)
    }

    output = TeamOutput(
        crops_dir=str(args.crops_dir),
        classifier_method=args.method,
        n_teams=args.n_teams,
        cluster_to_team_name=cluster_to_team_name,
        assignments=[
            TeamAssignment(
                crop_path=str(path),
                cluster_id=int(cid),
                track_id=tid,
                team_name=cluster_to_team_name.get(int(cid)),
            )
            for path, cid, tid in zip(paths, cluster_ids, track_id_list, strict=True)
        ],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output.model_dump_json(indent=2))
    return output


def cluster_main() -> None:
    run(parse_args())


def bench_main() -> None:
    raise NotImplementedError("basketball-team-bench is not implemented yet")
