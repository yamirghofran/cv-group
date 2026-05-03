"""CLI entry points for the clustering stage.

Registered in the root pyproject as:
    basketball-team-cluster = "clustering.cli:cluster_main"
    basketball-team-bench   = "clustering.cli:bench_main"
"""

from __future__ import annotations


def cluster_main() -> None:
    raise NotImplementedError("basketball-team-cluster is not implemented yet")


def bench_main() -> None:
    raise NotImplementedError("basketball-team-bench is not implemented yet")
