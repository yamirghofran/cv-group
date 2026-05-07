"""Build track-level team lookups from the per-crop clustering output.

The clustering stage produces a ``TeamOutput`` with one ``TeamAssignment`` per
crop.  Because each tracked player produces *multiple* crops (sampled over
time), we need to aggregate per-crop predictions into a single team label per
track using majority vote.

This module provides :func:`build_track_team_lookup` which returns a
``{track_id: team_name}`` mapping suitable for enriching ``TrackRecord`` instances
and driving team-aware visualisation.
"""

from __future__ import annotations

from collections import Counter

from clustering.schemas import TeamOutput


def build_track_team_lookup(team_output: TeamOutput) -> dict[int, str]:
    """Return a ``{track_id: team_name}`` mapping via majority vote.

    For each ``track_id`` found in *team_output.assignments*:

    1. Collect the ``cluster_id`` of every crop belonging to that track.
    2. Pick the most common cluster (majority vote).  Ties are broken by
       choosing the smaller cluster id.
    3. Map the winning cluster through ``team_output.cluster_to_team_name``.
       If the mapping has no entry for that cluster, a fallback name
       ``"Team <letter>"`` is generated using ``chr(65 + cluster_id)``.

    Returns an empty dict when no assignments carry a ``track_id`` (e.g. crops
    loaded from a flat directory with no ``track_*/`` structure).
    """
    cluster_to_name = team_output.cluster_to_team_name

    # Group assignments by track_id
    by_track: dict[int, list[int]] = {}
    for assignment in team_output.assignments:
        if assignment.track_id is None:
            continue
        by_track.setdefault(assignment.track_id, []).append(assignment.cluster_id)

    lookup: dict[int, str] = {}
    for track_id, cluster_ids in sorted(by_track.items()):
        # Majority vote: most common cluster_id, ties broken by smaller id
        counts = Counter(cluster_ids)
        winner = min(counts, key=lambda cid: (-counts[cid], cid))
        team_name = cluster_to_name.get(winner, f"Team {chr(65 + winner)}")
        lookup[track_id] = team_name

    return lookup
