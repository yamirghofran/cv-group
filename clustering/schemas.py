"""Pydantic models for the team-clustering stage.

Mirrors the Pydantic-on-disk pattern established by `object-detection/src/schemas.py`
so each stage of the pipeline reads/writes JSON files with validated structure.

Output of the clustering stage is a `TeamOutput` written to
`clustering/outputs/teams/<clip>_teams.json`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TeamAssignment(PipelineModel):
    """One crop's team assignment.

    Attributes:
        crop_path: Path to the source crop file (relative to repo root).
        cluster_id: Raw KMeans cluster index (0 .. n_teams-1).
        track_id: Tracker id from object-detection, if the crop filename
            encodes one. None for crops not tied to a track.
        team_name: Human-readable name once cluster→name mapping is decided
            (e.g. "Team A"). None until mapping is set.
    """

    crop_path: str
    cluster_id: int
    track_id: int | None = None
    team_name: str | None = None

    @field_validator("cluster_id")
    @classmethod
    def validate_cluster_id(cls, value: int) -> int:
        if value < 0:
            raise ValueError("cluster_id must be >= 0")
        return value


class TeamOutput(PipelineModel):
    """All team assignments for a single run of the clustering stage.

    Attributes:
        crops_dir: Directory of crops the classifier was run against.
        classifier_method: Identifier for the embedder+reducer+clusterer combo
            used (e.g. "siglip+umap+kmeans").
        n_teams: Number of clusters used (KMeans `n_clusters`).
        cluster_to_team_name: Optional mapping from cluster_id to a
            human-readable team name. Populated once a human (or the
            color-matching heuristic) decides which cluster is which team.
        assignments: One `TeamAssignment` per crop, in the same order as the
            input crops.
    """

    crops_dir: str
    classifier_method: str
    n_teams: int = 2
    cluster_to_team_name: dict[int, str] = Field(default_factory=dict)
    assignments: list[TeamAssignment]

    @field_validator("n_teams")
    @classmethod
    def validate_n_teams(cls, value: int) -> int:
        if value < 2:
            raise ValueError("n_teams must be >= 2")
        return value
