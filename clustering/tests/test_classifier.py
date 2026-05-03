"""Smoke tests for the vendored TeamClassifier.

These tests verify imports and the public API surface without loading the SigLIP
model (which downloads ~400MB and is slow on CPU). Real fit/predict against
sample crops is exercised by the integration tests once we have crops on disk.
"""

import inspect

import numpy as np


def test_classifier_imports() -> None:
    from clustering.classifier import SIGLIP_MODEL_PATH, TeamClassifier, create_batches

    assert SIGLIP_MODEL_PATH == "google/siglip-base-patch16-224"
    assert callable(create_batches)
    assert inspect.isclass(TeamClassifier)


def test_classifier_public_api() -> None:
    from clustering.classifier import TeamClassifier

    init_params = list(inspect.signature(TeamClassifier.__init__).parameters)
    assert init_params == ["self", "device", "batch_size"]

    fit_params = list(inspect.signature(TeamClassifier.fit).parameters)
    assert fit_params == ["self", "crops"]

    predict_params = list(inspect.signature(TeamClassifier.predict).parameters)
    assert predict_params == ["self", "crops"]

    fit_predict_params = list(inspect.signature(TeamClassifier.fit_predict).parameters)
    assert fit_predict_params == ["self", "crops"]


def test_fit_predict_returns_empty_array_for_empty_input() -> None:
    from clustering.classifier import TeamClassifier

    fp = TeamClassifier.fit_predict.__wrapped__ if hasattr(
        TeamClassifier.fit_predict, "__wrapped__"
    ) else TeamClassifier.fit_predict

    class _Fake:
        pass

    out = fp(_Fake(), [])  # type: ignore[arg-type]
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_create_batches_chunks_correctly() -> None:
    from clustering.classifier import create_batches

    out = list(create_batches(range(10), batch_size=3))
    assert out == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_create_batches_handles_empty_sequence() -> None:
    from clustering.classifier import create_batches

    assert list(create_batches([], batch_size=4)) == []


def test_create_batches_clamps_batch_size_below_one() -> None:
    from clustering.classifier import create_batches

    out = list(create_batches([1, 2, 3], batch_size=0))
    assert out == [[1], [2], [3]]


def test_predict_returns_empty_array_for_empty_input() -> None:
    from clustering.classifier import TeamClassifier

    # We can call predict's empty-input branch without instantiating the model.
    result = TeamClassifier.predict.__wrapped__ if hasattr(
        TeamClassifier.predict, "__wrapped__"
    ) else TeamClassifier.predict

    # Build a fake `self` whose attributes are never touched on the empty path.
    class _Fake:
        pass

    out = result(_Fake(), [])  # type: ignore[arg-type]
    assert isinstance(out, np.ndarray)
    assert out.size == 0
