def test_package_imports() -> None:
    import clustering  # noqa: F401


def test_cli_entry_points_exist() -> None:
    from clustering.cli import bench_main, cluster_main

    assert callable(cluster_main)
    assert callable(bench_main)
