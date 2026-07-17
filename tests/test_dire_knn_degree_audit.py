"""CPU tests for the DiRe NN-descent degree audit helpers."""

import argparse

import numpy as np
import pytest
from experiments.audit_dire_knn_degree import (
    graph_contract_summary,
    neighbor_recall_summary,
    parse_graph_configuration,
    remove_query_self,
)


def test_parse_graph_configuration():
    config = parse_graph_configuration("umap-matched:64:128")
    assert config.name == "umap-matched"
    assert config.graph_degree == 64
    assert config.intermediate_graph_degree == 128
    with pytest.raises(argparse.ArgumentTypeError):
        parse_graph_configuration("unsafe/name:64:128")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_graph_configuration("bad:64:32")


def test_remove_query_self_uses_global_query_ids_and_keeps_distances_aligned():
    neighbors = np.asarray([[4, 9, 1, 2], [7, 3, 8, 1]], dtype=np.int64)
    distances = np.asarray([[0.0, 0.1, 0.2, 0.3], [0.0, 0.4, 0.5, 0.6]], dtype=np.float32)
    clean, clean_distances = remove_query_self(
        neighbors,
        np.asarray([4, 7]),
        3,
        distances,
    )
    assert np.array_equal(clean, [[9, 1, 2], [3, 8, 1]])
    assert np.allclose(clean_distances, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_neighbor_recall_summary_and_graph_contract():
    exact = np.asarray([[1, 2, 3], [0, 2, 3]], dtype=np.int64)
    approximate = np.asarray([[1, 4, 3], [0, 2, 4]], dtype=np.int64)
    summary = neighbor_recall_summary(exact, approximate, ks=(1, 3))
    assert summary["mean_recall_at_1"] == 1.0
    assert summary["mean_recall_at_3"] == pytest.approx(2 / 3)
    assert summary["exact_match_fraction_at_3"] == 0.0

    graph = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    assert graph_contract_summary(graph, 3) == {
        "rows": 3,
        "neighbors": 2,
        "self_rows": 0,
        "out_of_range_rows": 0,
        "duplicate_rows": 0,
    }


def test_graph_contract_detects_invalid_rows():
    graph = np.asarray([[0, 1], [2, 2], [1, 9]], dtype=np.int64)
    summary = graph_contract_summary(graph, 3)
    assert summary["self_rows"] == 1
    assert summary["duplicate_rows"] == 1
    assert summary["out_of_range_rows"] == 1
