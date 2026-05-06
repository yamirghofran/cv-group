"""Tests for clustering.assign — majority-vote track-to-team lookup."""

from clustering.assign import build_track_team_lookup
from clustering.schemas import TeamAssignment, TeamOutput


def _make_output(
    assignments: list[tuple[str, int, int | None]],
    cluster_to_team_name: dict[int, str] | None = None,
) -> TeamOutput:
    """Helper to build a TeamOutput from (crop_path, cluster_id, track_id) tuples."""
    return TeamOutput(
        crops_dir="test",
        classifier_method="siglip+umap+kmeans",
        n_teams=2,
        cluster_to_team_name=cluster_to_team_name or {0: "Team A", 1: "Team B"},
        assignments=[
            TeamAssignment(
                crop_path=path,
                cluster_id=cid,
                track_id=tid,
                team_name=(cluster_to_team_name or {0: "Team A", 1: "Team B"}).get(cid),
            )
            for path, cid, tid in assignments
        ],
    )


def test_empty_assignments_return_empty_lookup() -> None:
    output = _make_output([])
    assert build_track_team_lookup(output) == {}


def test_no_track_ids_return_empty_lookup() -> None:
    output = _make_output([
        ("a.png", 0, None),
        ("b.png", 1, None),
    ])
    assert build_track_team_lookup(output) == {}


def test_single_track_unanimous_votes() -> None:
    output = _make_output([
        ("a1.png", 0, 1),
        ("a2.png", 0, 1),
        ("a3.png", 0, 1),
    ])
    lookup = build_track_team_lookup(output)
    assert lookup == {1: "Team A"}


def test_two_tracks_clear_separation() -> None:
    output = _make_output([
        ("a1.png", 0, 1),
        ("a2.png", 0, 1),
        ("b1.png", 1, 2),
        ("b2.png", 1, 2),
    ])
    lookup = build_track_team_lookup(output)
    assert lookup == {1: "Team A", 2: "Team B"}


def test_majority_vote_with_mixed_crops() -> None:
    """Track 1 has 3 crops in cluster 0 and 1 crop in cluster 1 → majority wins."""
    output = _make_output([
        ("a1.png", 0, 1),
        ("a2.png", 0, 1),
        ("a3.png", 0, 1),
        ("a4.png", 1, 1),  # outlier
    ])
    lookup = build_track_team_lookup(output)
    assert lookup[1] == "Team A"


def test_tie_broken_by_smaller_cluster_id() -> None:
    """Equal votes for cluster 0 and 1 → cluster 0 wins (smaller id)."""
    output = _make_output([
        ("a1.png", 0, 1),
        ("a2.png", 1, 1),
    ])
    lookup = build_track_team_lookup(output)
    assert lookup[1] == "Team A"


def test_fallback_name_when_no_mapping() -> None:
    """When cluster_to_team_name is empty, generate 'Team A', 'Team B', etc."""
    output = _make_output(
        assignments=[("a.png", 0, 1), ("b.png", 1, 2)],
        cluster_to_team_name={},
    )
    lookup = build_track_team_lookup(output)
    assert lookup[1] == "Team A"
    assert lookup[2] == "Team B"


def test_many_tracks() -> None:
    output = _make_output([
        (f"t{t}_f{i}.png", t % 2, t)
        for t in range(1, 6)
        for i in range(3)
    ])
    lookup = build_track_team_lookup(output)
    # track 1 → t%2=1 (odd) → cluster 1 → "Team B"
    assert lookup[1] == "Team B"
    # track 2 → t%2=0 (even) → cluster 0 → "Team A"
    assert lookup[2] == "Team A"


def test_ignores_assignments_without_track_id() -> None:
    output = _make_output([
        ("a.png", 0, 1),
        ("b.png", 1, None),  # no track_id — should be ignored
    ])
    lookup = build_track_team_lookup(output)
    assert lookup == {1: "Team A"}
