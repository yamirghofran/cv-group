import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from clustering.cli import DEFAULT_METHOD, parse_args, run

FIXTURES = Path(__file__).parent / "fixtures" / "sample_crops"


class _StubClassifier:
    """Drop-in replacement for TeamClassifier that doesn't load SigLIP.

    Returns deterministic labels derived from the crop's mean-blue minus
    mean-red channel, so the synthetic red/blue fixtures separate cleanly.
    """

    def __init__(self) -> None:
        self.fit_predict_called_with: list[np.ndarray] | None = None

    def fit(self, crops: list[np.ndarray]) -> None:
        return None

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        labels = []
        for crop in crops:
            blue = float(crop[..., 0].mean())
            red = float(crop[..., 2].mean())
            labels.append(0 if red >= blue else 1)
        return np.array(labels)

    def fit_predict(self, crops: list[np.ndarray]) -> np.ndarray:
        self.fit_predict_called_with = crops
        return self.predict(crops)


def test_parse_args_requires_crops_dir(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output", "/tmp/x.json"])
    err = capsys.readouterr().err
    assert "--crops-dir" in err


def test_parse_args_requires_output(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--crops-dir", "/tmp/crops"])
    err = capsys.readouterr().err
    assert "--output" in err


def test_parse_args_defaults() -> None:
    args = parse_args(["--crops-dir", "/tmp/c", "--output", "/tmp/o.json"])
    assert args.device == "cpu"
    assert args.batch_size == 32
    assert args.method == DEFAULT_METHOD


def test_run_writes_json_with_one_assignment_per_crop(tmp_path: Path) -> None:
    args = argparse.Namespace(
        crops_dir=FIXTURES,
        output=tmp_path / "out.json",
        device="cpu",
        batch_size=32,
        method=DEFAULT_METHOD,
    )

    output = run(args, classifier_factory=_StubClassifier)

    assert (tmp_path / "out.json").exists()
    assert len(output.assignments) == 6

    on_disk = json.loads((tmp_path / "out.json").read_text())
    assert on_disk["classifier_method"] == DEFAULT_METHOD
    assert on_disk["n_teams"] == 2
    assert len(on_disk["assignments"]) == 6


def test_run_uses_fit_predict_not_separate_fit_and_predict(tmp_path: Path) -> None:
    """Confirm the CLI uses the single-pass path so SigLIP doesn't run twice."""

    last: dict[str, _StubClassifier] = {}

    def factory() -> _StubClassifier:
        c = _StubClassifier()
        last["c"] = c
        return c

    args = argparse.Namespace(
        crops_dir=FIXTURES,
        output=tmp_path / "out.json",
        device="cpu",
        batch_size=32,
        method=DEFAULT_METHOD,
    )

    run(args, classifier_factory=factory)

    assert last["c"].fit_predict_called_with is not None
    assert len(last["c"].fit_predict_called_with) == 6


def test_run_separates_red_and_blue_fixtures(tmp_path: Path) -> None:
    args = argparse.Namespace(
        crops_dir=FIXTURES,
        output=tmp_path / "out.json",
        device="cpu",
        batch_size=32,
        method=DEFAULT_METHOD,
    )

    output = run(args, classifier_factory=_StubClassifier)

    by_name = {Path(a.crop_path).name: a.cluster_id for a in output.assignments}
    team_a_clusters = {by_name[f"team_a_{i:02d}.png"] for i in range(3)}
    team_b_clusters = {by_name[f"team_b_{i:02d}.png"] for i in range(3)}

    assert len(team_a_clusters) == 1
    assert len(team_b_clusters) == 1
    assert team_a_clusters != team_b_clusters


def test_run_creates_output_parent_directory(tmp_path: Path) -> None:
    args = argparse.Namespace(
        crops_dir=FIXTURES,
        output=tmp_path / "nested" / "deeply" / "out.json",
        device="cpu",
        batch_size=32,
        method=DEFAULT_METHOD,
    )

    run(args, classifier_factory=_StubClassifier)

    assert (tmp_path / "nested" / "deeply" / "out.json").exists()


def test_run_raises_on_empty_crops_dir(tmp_path: Path) -> None:
    args = argparse.Namespace(
        crops_dir=tmp_path,
        output=tmp_path / "out.json",
        device="cpu",
        batch_size=32,
        method=DEFAULT_METHOD,
    )

    with pytest.raises(ValueError, match="No crops found"):
        run(args, classifier_factory=_StubClassifier)
