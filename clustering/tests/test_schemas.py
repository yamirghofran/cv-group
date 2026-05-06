import json

import pytest
from pydantic import ValidationError

from clustering.schemas import TeamAssignment, TeamOutput


def test_team_assignment_minimal_fields() -> None:
    a = TeamAssignment(crop_path="x.png", cluster_id=0)
    assert a.crop_path == "x.png"
    assert a.cluster_id == 0
    assert a.track_id is None
    assert a.team_name is None


def test_team_assignment_all_fields() -> None:
    a = TeamAssignment(crop_path="x.png", cluster_id=1, track_id=42, team_name="Team A")
    assert a.track_id == 42
    assert a.team_name == "Team A"


def test_team_assignment_rejects_negative_cluster_id() -> None:
    with pytest.raises(ValidationError, match="cluster_id must be >= 0"):
        TeamAssignment(crop_path="x.png", cluster_id=-1)


def test_team_assignment_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TeamAssignment(crop_path="x.png", cluster_id=0, made_up_field="oops")


def test_team_output_minimal() -> None:
    out = TeamOutput(
        crops_dir="object-detection/outputs/crops/clip_001",
        classifier_method="siglip+umap+kmeans",
        assignments=[],
    )
    assert out.n_teams == 2
    assert out.cluster_to_team_name == {}
    assert out.assignments == []


def test_team_output_rejects_n_teams_below_two() -> None:
    with pytest.raises(ValidationError, match="n_teams must be >= 2"):
        TeamOutput(
            crops_dir="x",
            classifier_method="m",
            n_teams=1,
            assignments=[],
        )


def test_team_output_round_trip_json() -> None:
    original = TeamOutput(
        crops_dir="object-detection/outputs/crops/clip_001",
        classifier_method="siglip+umap+kmeans",
        n_teams=2,
        cluster_to_team_name={0: "Team A", 1: "Team B"},
        assignments=[
            TeamAssignment(crop_path="a.png", cluster_id=0, track_id=1, team_name="Team A"),
            TeamAssignment(crop_path="b.png", cluster_id=1, track_id=2, team_name="Team B"),
        ],
    )
    serialized = original.model_dump_json()
    decoded = TeamOutput.model_validate_json(serialized)
    assert decoded == original


def test_team_output_json_is_valid_json() -> None:
    out = TeamOutput(
        crops_dir="x",
        classifier_method="m",
        assignments=[TeamAssignment(crop_path="a.png", cluster_id=0)],
    )
    parsed = json.loads(out.model_dump_json())
    assert parsed["assignments"][0]["crop_path"] == "a.png"
