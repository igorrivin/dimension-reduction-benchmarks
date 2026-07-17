"""CPU-only tests for the single-cell source and preprocessing contract."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments import prepare_single_cell as prep


class FakeAnnData:
    """Small AnnData-shaped object sufficient for pure orchestration tests."""

    def __init__(self, X, obs, var, *, obs_names=None, layers=None):
        pd = pytest.importorskip("pandas")
        self.X = X
        self.obs = pd.DataFrame(obs).copy()
        self.var = pd.DataFrame(var).copy()
        self.obs.index = (
            pd.Index(obs_names, dtype=str)
            if obs_names is not None
            else pd.Index([f"cell-{index}" for index in range(len(self.obs))], dtype=str)
        )
        if self.var.index.equals(pd.RangeIndex(len(self.var))):
            self.var.index = pd.Index(
                [f"gene-{index}" for index in range(len(self.var))], dtype=str
            )
        self.layers = {} if layers is None else layers
        self.obsm = {}
        self.varm = {}

    @property
    def shape(self):
        return self.X.shape

    @property
    def n_obs(self):
        return self.X.shape[0]

    @property
    def n_vars(self):
        return self.X.shape[1]

    @property
    def obs_names(self):
        return self.obs.index

    @property
    def var_names(self):
        return self.var.index

    def copy(self):
        return copy.deepcopy(self)

    def __getitem__(self, key):
        rows, columns = key
        row_indices = np.arange(self.n_obs)[rows]
        column_indices = np.arange(self.n_vars)[columns]
        result = FakeAnnData(
            self.X[np.ix_(np.atleast_1d(row_indices), np.atleast_1d(column_indices))],
            self.obs.iloc[np.atleast_1d(row_indices)].copy(),
            self.var.iloc[np.atleast_1d(column_indices)].copy(),
            obs_names=self.obs.index[np.atleast_1d(row_indices)],
        )
        result.obsm = {
            name: np.asarray(values)[np.atleast_1d(row_indices)].copy()
            for name, values in self.obsm.items()
        }
        result.varm = {
            name: np.asarray(values)[np.atleast_1d(column_indices)].copy()
            for name, values in self.varm.items()
        }
        return result


def _raw_adata(rows: int = 8, genes: int = 5) -> FakeAnnData:
    pd = pytest.importorskip("pandas")
    X = (np.arange(rows * genes).reshape(rows, genes) % 7).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": ["T", "T", "B", "B"] * (rows // 4),
            "cell_type_ontology_term_id": ["CL:1", "CL:1", "CL:2", "CL:2"] * (rows // 4),
            "donor_id": ["d1", "d2", "d1", "d2"] * (rows // 4),
            "tissue": ["lung"] * rows,
            "tissue_general": ["lung"] * rows,
            "assay": ["10x"] * rows,
            "dataset_id": ["dataset-a"] * rows,
            "soma_joinid": np.arange(100, 100 + rows, dtype=np.int64),
            "is_primary_data": [True] * rows,
        }
    )
    var = pd.DataFrame(
        {
            "feature_id": [f"ENSG{index}" for index in range(genes)],
            "feature_name": ["MT-CO1", *[f"G{index}" for index in range(1, genes)]],
        },
        index=[f"ENSG{index}" for index in range(genes)],
    )
    return FakeAnnData(X, obs, var, obs_names=[f"cell-{index}" for index in range(rows)])


def _minimal_scanpy(calls, *, hvg_hook=None):
    """Return a light Scanpy-shaped fake for preprocessing contract tests."""

    class FakePP:
        @staticmethod
        def highly_variable_genes(data, **kwargs):
            calls.append(("hvg", kwargs, data))
            if hvg_hook is not None:
                hvg_hook(data, kwargs)
            selected = min(int(kwargs["n_top_genes"]), data.n_vars)
            data.var["highly_variable"] = np.arange(data.n_vars) < selected
            data.var["highly_variable_rank"] = np.asarray(
                [float(index) if index < selected else np.nan for index in range(data.n_vars)]
            )

        @staticmethod
        def normalize_total(data, **kwargs):
            calls.append(("normalize_total", kwargs, data))

        @staticmethod
        def log1p(data):
            calls.append(("log1p", {}, data))

        @staticmethod
        def scale(data, **kwargs):
            calls.append(("scale", kwargs, data))

        @staticmethod
        def pca(data, **kwargs):
            calls.append(("pca", kwargs, data))
            data.obsm["X_pca"] = np.zeros((data.n_obs, kwargs["n_comps"]), dtype=np.float32)
            data.varm["PCs"] = np.zeros((data.n_vars, kwargs["n_comps"]), dtype=np.float32)

    return SimpleNamespace(pp=FakePP)


def _successful_loess(*_args, **_kwargs):
    """Return a LOESS-shaped no-op so CPU tests do not require scikit-misc."""

    return SimpleNamespace(fit=lambda: None)


def _aggregate_for_tests(adata, *, by, func):
    """Return Scanpy-aggregate-shaped stable mean/variance test output."""

    assert func == ["mean", "var"]
    pd = pytest.importorskip("pandas")
    labels = np.asarray(adata.obs[by], dtype=str)
    unique = np.unique(labels)
    means = []
    variances = []
    counts = []
    for label in unique:
        values = np.asarray(adata.X)[labels == label].astype(np.float64)
        means.append(np.mean(values, axis=0))
        variances.append(np.var(values, axis=0, ddof=1))
        counts.append(values.shape[0])
    return SimpleNamespace(
        obs=pd.DataFrame({by: unique, "n_obs_aggregated": counts}, index=unique),
        layers={"mean": np.asarray(means), "var": np.asarray(variances)},
    )


def _prepared_fixture(rows: int = 8, hvgs: int = 3) -> prep.PreparedRepresentations:
    """Return a small serialization-valid prepared result for execute tests."""

    pd = pytest.importorskip("pandas")
    return prep.PreparedRepresentations(
        pca=np.zeros((rows, 2), dtype=np.float32),
        log_hvg=None,
        metadata=pd.DataFrame({"cell_id": [f"cell-{index}" for index in range(rows)]}),
        genes=pd.DataFrame(
            {
                "gene_id": [f"gene-{index}" for index in range(hvgs)],
                "highly_variable_rank": np.arange(hvgs, dtype=np.int64),
            }
        ),
        pipeline={"hvg": {"selected_genes": hvgs}},
    )


def test_import_does_not_eagerly_load_optional_single_cell_stack():
    code = """
import json
import sys

optional = {"anndata", "scanpy", "cellxgene_census", "pandas", "pyarrow"}
before = {name.split(".", 1)[0] for name in sys.modules}
import experiments.prepare_single_cell  # noqa: F401
after = {name.split(".", 1)[0] for name in sys.modules}
print(json.dumps(sorted((after - before) & optional)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_deterministic_stratified_sample_is_exact_balanced_and_seeded():
    strata = np.asarray(["a"] * 10 + ["b"] * 20 + ["c"] * 30)

    first, details = prep.deterministic_stratified_subsample(strata, 12, seed=41)
    repeat, _ = prep.deterministic_stratified_subsample(strata, 12, seed=41)
    changed, _ = prep.deterministic_stratified_subsample(strata, 12, seed=42)

    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, changed)
    assert np.all(first[:-1] < first[1:])
    assert [row["candidate_cells"] for row in details] == [10, 20, 30]
    assert [row["selected_cells"] for row in details] == [2, 4, 6]
    assert [np.count_nonzero(strata[first] == value) for value in ("a", "b", "c")] == [
        2,
        4,
        6,
    ]


def test_raw_count_qc_matches_for_dense_and_sparse_inputs():
    sparse = pytest.importorskip("scipy.sparse")
    X = np.asarray([[1, 0, 3], [0, 2, 0], [0, 0, 0]], dtype=np.float32)
    names = ["MT-CO1", "ACTB", "mt-ND1"]

    dense_qc = prep.compute_count_qc(X, names)
    sparse_qc = prep.compute_count_qc(sparse.csr_matrix(X), names)

    for column in prep.QC_COLUMNS:
        np.testing.assert_allclose(dense_qc[column], sparse_qc[column])
    np.testing.assert_allclose(dense_qc["total_counts"], [4, 2, 0])
    np.testing.assert_array_equal(dense_qc["n_genes_by_counts"], [2, 1, 0])
    np.testing.assert_allclose(dense_qc["pct_counts_mt"], [100, 0, 0])


def test_census_query_is_pinned_primary_metadata_first_and_row_aligned(monkeypatch):
    pd = pytest.importorskip("pandas")
    calls = []
    candidates = _raw_adata().obs.reset_index(drop=True)
    candidates["library_id"] = ["lib-a", "lib-b"] * 4
    dataset_table = pd.DataFrame(
        {
            "collection_id": [prep.TABULA_SAPIENS_COLLECTION_ID, "another-collection"],
            "dataset_id": ["dataset-a", "dataset-z"],
            "dataset_title": [prep.TABULA_SAPIENS_DATASET_TITLE, "Other"],
            "dataset_total_cell_count": [8, 2],
        }
    )

    class DatasetReader:
        def read(self):
            return self

        def concat(self):
            return self

        def to_pandas(self):
            return dataset_table.copy()

    class Census:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __getitem__(self, key):
            assert key == "census_info"
            return {"datasets": DatasetReader()}

    def get_obs(census, organism, **kwargs):
        calls.append(("get_obs", kwargs))
        assert organism == "Homo sapiens"
        return candidates.copy()

    def get_anndata(census, organism, **kwargs):
        calls.append(("get_anndata", kwargs))
        ids = np.asarray(kwargs["obs_coords"], dtype=np.int64)[::-1]
        positions = [int(value - 100) for value in ids]
        source = _raw_adata()
        source.obs["library_id"] = ["lib-a", "lib-b"] * 4
        return source[np.asarray(positions, dtype=np.int64), :]

    fake_module = SimpleNamespace(
        open_soma=lambda **kwargs: Census(), get_obs=get_obs, get_anndata=get_anndata
    )
    monkeypatch.setitem(sys.modules, "cellxgene_census", fake_module)

    loaded = prep.load_tabula_sapiens_census(
        max_cells=4, sample_seed=17, hvg_batch_fields=("library_id",)
    )

    assert [name for name, _ in calls] == ["get_obs", "get_anndata"]
    get_obs_kwargs = calls[0][1]
    get_anndata_kwargs = calls[1][1]
    assert "is_primary_data == True" in get_obs_kwargs["value_filter"]
    assert "dataset-a" in get_obs_kwargs["value_filter"]
    assert "library_id" in get_obs_kwargs["column_names"]
    assert get_anndata_kwargs["X_name"] == "raw"
    assert get_anndata_kwargs["obs_value_filter"] == get_obs_kwargs["value_filter"]
    np.testing.assert_array_equal(
        np.asarray(loaded.adata.obs["soma_joinid"]),
        np.asarray(get_anndata_kwargs["obs_coords"]),
    )
    assert loaded.source["census_version"] == prep.DEFAULT_CENSUS_VERSION
    assert loaded.source["resolved_dataset_ids"] == ["dataset-a"]
    assert loaded.source["candidate_cells"] == 8
    assert loaded.source["selected_shape"] == [4, 5]
    assert loaded.sampling["selected_cells"] == 4
    assert all(column in loaded.adata.obs for column in prep.QC_COLUMNS)


def test_census_checkpoint_identity_includes_configured_hvg_batch_fields():
    common = {
        "census_version": prep.DEFAULT_CENSUS_VERSION,
        "collection_id": prep.TABULA_SAPIENS_COLLECTION_ID,
        "dataset_title": prep.TABULA_SAPIENS_DATASET_TITLE,
        "max_cells": prep.CENSUS_PILOT_CELLS,
        "sample_seed": prep.DEFAULT_SAMPLE_SEED,
        "stratify_fields": prep.DEFAULT_CENSUS_STRATIFY_FIELDS,
        "metadata_fields": prep.DEFAULT_METADATA_FIELDS,
    }

    configured = prep.census_raw_request_identity(**common, hvg_batch_fields=("library_id",))
    default = prep.census_raw_request_identity(**common)

    assert configured["hvg_batch_fields"] == ["library_id"]
    assert "library_id" in configured["observation_columns"]
    assert prep._sha256_json(configured) != prep._sha256_json(default)


def test_scanpy_pipeline_uses_raw_seurat_v3_donor_assay_and_nonwhitened_pca():
    adata = _raw_adata(rows=8, genes=5)
    loaded = prep.LoadedCells(adata=adata, source={}, sampling={}, qc={})
    calls = []
    raw = adata.X.copy()

    class FakePP:
        @staticmethod
        def highly_variable_genes(data, **kwargs):
            calls.append(("hvg", kwargs, data.X.copy()))
            data.var["highly_variable"] = [True, True, True, False, False]
            data.var["highly_variable_rank"] = [0.0, 1.0, 2.0, np.nan, np.nan]

        @staticmethod
        def normalize_total(data, **kwargs):
            calls.append(("normalize_total", kwargs, data.X.copy()))
            totals = data.X.sum(axis=1, keepdims=True)
            data.X = np.divide(
                kwargs["target_sum"] * data.X,
                totals,
                out=np.zeros_like(data.X),
                where=totals > 0,
            )

        @staticmethod
        def log1p(data):
            calls.append(("log1p", {}, data.X.copy()))
            data.X = np.log1p(data.X)

        @staticmethod
        def scale(data, **kwargs):
            calls.append(("scale", kwargs, data.X.copy()))
            means = data.X.mean(axis=0)
            standard = data.X.std(axis=0)
            standard[standard == 0] = 1
            data.X = np.clip((data.X - means) / standard, -kwargs["max_value"], kwargs["max_value"])

        @staticmethod
        def pca(data, **kwargs):
            calls.append(("pca", kwargs, data.X.copy()))
            data.obsm["X_pca"] = np.ascontiguousarray(
                data.X[:, : kwargs["n_comps"]], dtype=np.float32
            )
            data.varm["PCs"] = np.eye(data.n_vars, kwargs["n_comps"], dtype=np.float32)

    prepared = prep.prepare_representations(
        loaded,
        n_hvgs=3,
        n_pcs=2,
        write_log_hvg=True,
        scanpy_module=SimpleNamespace(pp=FakePP),
        loess_factory=_successful_loess,
        aggregate_func=_aggregate_for_tests,
        max_dense_bytes=1_000_000,
    )

    assert [name for name, *_ in calls] == ["hvg", "normalize_total", "log1p", "scale", "pca"]
    np.testing.assert_array_equal(calls[0][2], raw)
    assert calls[0][1] == {
        "flavor": "seurat_v3",
        "n_top_genes": 3,
        "batch_key": "hvg_batch",
        "span": 0.5,
        "subset": False,
        "inplace": True,
    }
    assert set(loaded.adata.obs["hvg_batch"]) == {"d1|10x", "d2|10x"}
    assert prepared.pipeline["hvg"]["batch_fields"] == ["donor_id", "assay"]
    assert prepared.pipeline["hvg"]["batch_definition"] == "donor_id|assay"
    assert prepared.pipeline["hvg"]["cell_type_or_tissue_labels_used"] is False
    assert calls[-1][1]["zero_center"] is True
    assert calls[-1][1]["svd_solver"] == "randomized"
    assert prepared.pipeline["pca"]["whiten"] is False
    assert prepared.pca.shape == (8, 2)
    assert prepared.pca.dtype == np.float32
    assert prepared.log_hvg.shape == (8, 3)


def test_hvg_span_is_passed_to_scanpy_and_recorded_in_pipeline():
    adata = _raw_adata(rows=8, genes=5)
    loaded = prep.LoadedCells(adata=adata, source={}, sampling={}, qc={})
    calls = []
    loess_calls = []

    def capture_loess(predictors, response, **kwargs):
        loess_calls.append((np.asarray(predictors), np.asarray(response), kwargs))
        return SimpleNamespace(fit=lambda: None)

    prepared = prep.prepare_representations(
        loaded,
        n_hvgs=3,
        n_pcs=2,
        hvg_span=0.625,
        scanpy_module=_minimal_scanpy(calls),
        loess_factory=capture_loess,
        aggregate_func=_aggregate_for_tests,
        max_dense_bytes=1_000_000,
    )

    hvg_calls = [entry for entry in calls if entry[0] == "hvg"]
    assert len(hvg_calls) == 1
    assert hvg_calls[0][1]["span"] == pytest.approx(0.625)
    assert loess_calls
    assert all(call[2] == {"span": 0.625, "degree": 2} for call in loess_calls)
    hvg = prepared.pipeline["hvg"]
    assert hvg["span"] == pytest.approx(0.625)
    assert hvg["degree"] == 2
    assert hvg["span_policy"] == "fixed_global"
    assert hvg["fallback"] == "none"


def test_hvg_batch_diagnostics_are_sorted_counted_and_record_conditioning():
    adata = _raw_adata(rows=12, genes=12)
    rng = np.random.default_rng(440)
    adata.X = rng.integers(0, 31, size=adata.shape, dtype=np.int32).astype(np.float32)
    loaded = prep.LoadedCells(adata=adata, source={}, sampling={}, qc={})

    prepared = prep.prepare_representations(
        loaded,
        n_hvgs=3,
        n_pcs=2,
        scanpy_module=_minimal_scanpy([]),
        loess_factory=_successful_loess,
        aggregate_func=_aggregate_for_tests,
        max_dense_bytes=1_000_000,
    )

    diagnostics = prepared.pipeline["hvg"]["batch_diagnostics"]
    assert [row["batch_label"] for row in diagnostics] == ["d1|10x", "d2|10x"]
    for row in diagnostics:
        assert {
            "batch_label",
            "cell_count",
            "nonconstant_genes",
            "distinct_log10_means",
            "duplicate_log10_mean_fraction",
            "loess_neighborhood_points",
            "status",
        } <= set(row)
        mask = np.asarray(adata.obs["hvg_batch"], dtype=str) == row["batch_label"]
        batch = np.asarray(adata.X)[mask]
        means = batch.mean(axis=0, dtype=np.float64)
        variances = batch.var(axis=0, ddof=1, dtype=np.float64)
        nonconstant = variances > 0
        n_nonconstant = int(nonconstant.sum())
        distinct_log10_means = int(np.unique(np.log10(means[nonconstant])).size)
        assert row["cell_count"] == int(mask.sum())
        assert row["nonconstant_genes"] == n_nonconstant
        assert row["distinct_log10_means"] == distinct_log10_means
        assert row["duplicate_log10_mean_fraction"] == pytest.approx(
            1.0 - distinct_log10_means / n_nonconstant
        )
        assert row["loess_neighborhood_points"] == min(
            n_nonconstant,
            int(np.ceil(prep.DEFAULT_HVG_SPAN * n_nonconstant)),
        )
        assert row["status"] == "ok"


def test_hvg_near_singularity_error_identifies_the_failing_batch():
    adata = _raw_adata(rows=8, genes=5)
    loaded = prep.LoadedCells(adata=adata, source={}, sampling={}, qc={})
    loess_calls = 0

    def fail_for_second_batch(*_args, **_kwargs):
        nonlocal loess_calls
        loess_calls += 1

        def fit():
            if loess_calls == 2:
                raise ValueError("There are other near singularities as well")

        return SimpleNamespace(fit=fit)

    with pytest.raises(RuntimeError) as error:
        prep.prepare_representations(
            loaded,
            n_hvgs=3,
            n_pcs=2,
            scanpy_module=_minimal_scanpy([]),
            loess_factory=fail_for_second_batch,
            aggregate_func=_aggregate_for_tests,
            max_dense_bytes=1_000_000,
        )

    message = str(error.value)
    assert "d2|10x" in message
    assert "fixed span" in message.lower()
    assert "no fallback" in message.lower()
    failed = [row for row in error.value.diagnostics if row["status"] == "failed"]
    assert [row["batch_label"] for row in failed] == ["d2|10x"]
    assert "near singular" in failed[0]["error_message"].lower()


def test_zero_variance_batch_skips_loess_without_inventing_a_failure():
    pd = pytest.importorskip("pandas")
    adata = _raw_adata()
    adata.obs["hvg_batch"] = "constant-batch"

    def aggregate(_adata, *, by, func):
        assert by == "hvg_batch"
        assert func == ["mean", "var"]
        return SimpleNamespace(
            obs=pd.DataFrame({by: ["constant-batch"], "n_obs_aggregated": [_adata.n_obs]}),
            layers={
                "mean": np.zeros((1, _adata.n_vars), dtype=np.float64),
                "var": np.zeros((1, _adata.n_vars), dtype=np.float64),
            },
        )

    diagnostics = prep.diagnose_seurat_v3_batches(
        adata,
        batch_key="hvg_batch",
        span=0.5,
        aggregate_func=aggregate,
        loess_factory=lambda *_args, **_kwargs: pytest.fail(
            "LOESS called for a zero-variance batch"
        ),
    )

    assert diagnostics[0]["status"] == "ok"
    assert diagnostics[0]["nonconstant_genes"] == 0
    assert diagnostics[0]["loess_applied"] is False


def test_biological_hvg_batch_field_is_rejected_before_source_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prep,
        "load_tabula_sapiens_census",
        lambda *_args, **_kwargs: pytest.fail("source loaded before batch-field validation"),
    )
    args = prep.parse_args(
        [
            "tabula-sapiens",
            "--output",
            str(tmp_path / "prepared"),
            "--hvg-batch-fields",
            "donor_id,cell_type",
        ]
    )

    with pytest.raises(ValueError, match="biological.*cell_type"):
        prep.execute(args)


def test_wrong_hvg_count_writes_diagnostics_and_failed_stage_timings(tmp_path, monkeypatch):
    loaded = prep.LoadedCells(
        adata=_raw_adata(),
        source={"mode": "cellxgene_census"},
        sampling={"selected_cells": 8},
        qc={"cell_filters": {"enabled": False}},
    )
    diagnostics = [
        {
            "batch_label": "d1|10x",
            "cell_count": 4,
            "span": 0.5,
            "degree": 2,
            "status": "ok",
        }
    ]

    class WrongCountPP:
        @staticmethod
        def highly_variable_genes(data, **_kwargs):
            data.var["highly_variable"] = [True, True, False, False, False]

    monkeypatch.setitem(sys.modules, "scanpy", SimpleNamespace(pp=WrongCountPP))
    monkeypatch.setattr(
        prep,
        "diagnose_seurat_v3_batches",
        lambda *_args, **_kwargs: copy.deepcopy(diagnostics),
    )
    monkeypatch.setattr(prep, "load_tabula_sapiens_census", lambda *_args, **_kwargs: loaded)
    output = tmp_path / "prepared"
    args = prep.parse_args(
        [
            "tabula-sapiens",
            "--output",
            str(output),
            "--n-hvg",
            "3",
            "--n-pcs",
            "2",
        ]
    )

    with pytest.raises(prep.HVGLoessDiagnosticsError, match="selected 2 HVGs"):
        prep.execute(args)

    failure_path = tmp_path / "prepared.hvg-loess-failure.json"
    failure = json.loads(failure_path.read_text())
    assert failure["fallback"] == "none"
    assert failure["diagnostics"] == diagnostics
    assert failure["stage_status"]["hvg_batch_diagnostics"] == "executed"
    assert failure["stage_status"]["hvg_selection"] == "failed"
    assert failure["stage_status"]["preprocessing"] == "failed"
    assert set(failure["stage_timings_seconds"]) == set(failure["stage_status"])
    assert failure["stage_timings_seconds"]["hvg_selection"] >= 0
    assert not (output / "manifest.json").exists()


def test_real_skmisc_fixed_span_preflight_accepts_well_conditioned_counts():
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("skmisc.loess")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(517)
    rates = np.geomspace(0.2, 30.0, 256)
    matrix = rng.poisson(rates, size=(120, rates.size)).astype(np.float32)
    adata = anndata.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"hvg_batch": ["batch-a"] * matrix.shape[0]},
            index=[f"cell-{index}" for index in range(matrix.shape[0])],
        ),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(matrix.shape[1])]),
    )

    diagnostics = prep.diagnose_seurat_v3_batches(
        adata,
        batch_key="hvg_batch",
        span=prep.DEFAULT_HVG_SPAN,
    )

    assert len(diagnostics) == 1
    assert diagnostics[0]["batch_label"] == "batch-a"
    assert diagnostics[0]["status"] == "ok"
    assert diagnostics[0]["span"] == pytest.approx(0.5)


def test_scanpy_batch_aggregate_preserves_high_integer_variance():
    anndata = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    scanpy = pytest.importorskip("scanpy")
    base = 10**9
    matrix = np.asarray(
        [
            [base, base, base],
            [base + 1, base + 2, base + 3],
            [base + 2, base + 4, base + 6],
            [base + 3, base + 6, base + 9],
        ],
        dtype=np.int64,
    )
    adata = anndata.AnnData(
        X=matrix,
        obs=pd.DataFrame({"hvg_batch": ["batch-a"] * 4}, index=[f"cell-{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"gene-{i}" for i in range(3)]),
    )

    aggregated = scanpy.get.aggregate(adata, by="hvg_batch", func=["mean", "var"])
    np.testing.assert_allclose(
        np.asarray(aggregated.layers["var"])[0],
        np.asarray([5 / 3, 20 / 3, 15], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    diagnostics = prep.diagnose_seurat_v3_batches(
        adata,
        batch_key="hvg_batch",
        span=0.5,
        loess_factory=_successful_loess,
        aggregate_func=scanpy.get.aggregate,
    )
    assert diagnostics[0]["nonconstant_genes"] == 3
    assert diagnostics[0]["status"] == "ok"


def test_real_scanpy_seurat_v3_pipeline_uses_the_fixed_span():
    anndata = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("scanpy")
    sparse = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(518)
    rates = np.geomspace(0.2, 30.0, 256)
    matrix = sparse.csr_matrix(rng.poisson(rates, size=(120, rates.size)).astype(np.float32))
    obs = pd.DataFrame(
        {
            "donor_id": np.repeat(["d1", "d2"], 60),
            "assay": np.tile(np.repeat(["10x", "smart"], 30), 2),
        },
        index=[f"cell-{index}" for index in range(matrix.shape[0])],
    )
    var = pd.DataFrame(
        {"feature_id": [f"gene-{index}" for index in range(matrix.shape[1])]},
        index=[f"gene-{index}" for index in range(matrix.shape[1])],
    )
    loaded = prep.LoadedCells(
        adata=anndata.AnnData(X=matrix, obs=obs, var=var),
        source={"mode": "fixture"},
        sampling={"selected_cells": matrix.shape[0]},
        qc={"cell_filters": {"enabled": False}},
    )

    prepared = prep.prepare_representations(
        loaded,
        n_hvgs=50,
        n_pcs=10,
        hvg_span=0.5,
        max_dense_bytes=10_000_000,
    )

    assert prepared.pca.shape == (120, 10)
    assert prepared.pipeline["hvg"]["selected_genes"] == 50
    assert prepared.pipeline["hvg"]["span"] == pytest.approx(0.5)
    assert prepared.pipeline["hvg"]["fallback"] == "none"
    assert all(row["status"] == "ok" for row in prepared.pipeline["hvg"]["batch_diagnostics"])


def test_hvg_runs_before_any_full_copy_of_the_loaded_matrix():
    adata = _raw_adata(rows=8, genes=5)
    loaded = prep.LoadedCells(adata=adata, source={}, sampling={}, qc={})
    calls = []
    hvg_started = False
    ordinary_copy = adata.copy

    def guarded_copy():
        if not hvg_started:
            raise AssertionError("full source AnnData copied before HVG selection")
        return ordinary_copy()

    def mark_hvg_started(data, _kwargs):
        nonlocal hvg_started
        assert data is adata
        hvg_started = True

    adata.copy = guarded_copy
    prep.prepare_representations(
        loaded,
        n_hvgs=3,
        n_pcs=2,
        scanpy_module=_minimal_scanpy(calls, hvg_hook=mark_hvg_started),
        loess_factory=_successful_loess,
        aggregate_func=_aggregate_for_tests,
        max_dense_bytes=1_000_000,
    )

    assert hvg_started is True
    assert [entry[0] for entry in calls].count("hvg") == 1


def test_scale_guard_accounts_for_scanpy_float64_dense_materialization():
    adata = _raw_adata(rows=8, genes=5)
    loaded = prep.LoadedCells(adata=adata, source={}, sampling={}, qc={})
    expected_bytes = 8 * 3 * np.dtype(np.float64).itemsize

    with pytest.raises(MemoryError, match=f"{expected_bytes:,} bytes"):
        prep.prepare_representations(
            loaded,
            n_hvgs=3,
            n_pcs=2,
            scanpy_module=_minimal_scanpy([]),
            loess_factory=_successful_loess,
            aggregate_func=_aggregate_for_tests,
            max_dense_bytes=expected_bytes - 1,
        )


def test_tenx_local_adapter_records_hash_qc_and_unresolved_batch_metadata(tmp_path):
    matrix_path = tmp_path / "filtered_matrix.h5"
    matrix_path.write_bytes(b"small fake HDF5 fixture identity")
    calls = []

    def read_10x_h5(path, *, gex_only):
        calls.append((path, gex_only))
        return _raw_adata()

    loaded = prep.load_tenx_brain_h5(
        matrix_path,
        max_cells=4,
        sample_seed=9,
        min_cells_per_gene=1,
        scanpy_module=SimpleNamespace(read_10x_h5=read_10x_h5),
    )

    assert calls == [(str(matrix_path.resolve()), True)]
    assert loaded.source["mode"] == "tenx_h5"
    assert loaded.source["sha256"] == hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    assert loaded.source["original_shape"] == [8, 5]
    assert loaded.source["selected_shape"] == [4, 5]
    assert loaded.sampling["selected_cells"] == 4
    assert set(loaded.adata.obs["dataset_id"]) == {prep.TENX_BRAIN_DATASET_ID}
    assert set(loaded.adata.obs["donor_id"]) == {"pooled-two-mice-unresolved"}
    assert set(loaded.adata.obs["assay"]) == {prep.TENX_BRAIN_ASSAY}
    assert loaded.qc["gene_filter"] == {
        "enabled": True,
        "min_cells_per_gene": 1,
        "genes_before": 5,
        "genes_after": 5,
    }


def test_remote_cache_is_atomic_hashed_and_reused_without_a_second_request(tmp_path, monkeypatch):
    import urllib.request

    payload = b"pretend HDF5 bytes"
    requests = []

    class Response:
        status = 200
        headers = {"Content-Length": str(len(payload)), "ETag": '"fixture"'}

        def __init__(self):
            self.remaining = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def geturl(self):
            return "https://example.test/matrix.h5"

        def read(self, _size):
            result, self.remaining = self.remaining, b""
            return result

    def urlopen(request, *, timeout):
        requests.append((request.full_url, timeout))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    expected = hashlib.sha256(payload).hexdigest()

    path, first = prep.cache_remote_file(
        "https://example.test/matrix.h5", tmp_path / "cache", expected_sha256=expected
    )
    reused_path, second = prep.cache_remote_file(
        "https://example.test/matrix.h5", tmp_path / "cache", expected_sha256=expected
    )

    assert requests == [("https://example.test/matrix.h5", 120)]
    assert path == reused_path
    assert path.read_bytes() == payload
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["sha256"] == second["sha256"] == expected
    assert not list(path.parent.glob("*.part"))


def test_standard_artifacts_and_manifest_contain_hashes(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pca = np.arange(12, dtype=np.float32).reshape(6, 2)
    log_hvg = np.arange(18, dtype=np.float32).reshape(6, 3)
    metadata = pd.DataFrame(
        {"cell_id": [f"c{index}" for index in range(6)], "cell_type": ["T"] * 6}
    )
    genes = pd.DataFrame({"gene_id": ["g1", "g2", "g3"], "highly_variable_rank": [0, 1, 2]})
    loaded = prep.LoadedCells(
        adata=None,
        source={"mode": "test", "sha256": "source-hash"},
        sampling={"selected_cells": 6},
        qc={"cell_filters": {"enabled": False}},
    )
    prepared = prep.PreparedRepresentations(
        pca=pca,
        log_hvg=log_hvg,
        metadata=metadata,
        genes=genes,
        pipeline={"configuration_sha256": "pipeline-hash"},
    )

    manifest = prep.write_prepared_outputs(tmp_path, loaded=loaded, prepared=prepared)

    expected = {
        "pca2.npy",
        "log_hvg3.npy",
        "metadata.parquet",
        "genes.parquet",
        "manifest.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    np.testing.assert_array_equal(np.load(tmp_path / "pca2.npy"), pca)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == on_disk
    assert set(on_disk) == {
        "schema_version",
        "dataset",
        "selection",
        "preprocessing",
        "artifacts",
        "software",
    }
    assert on_disk["selection"]["selected_cells"] == 6
    for artifact in on_disk["artifacts"].values():
        payload = (tmp_path / artifact["path"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
        assert artifact["bytes"] == len(payload)
        assert "shape" in artifact
        assert "dtype" in artifact


def test_execute_manifest_records_stage_timings_and_status(tmp_path, monkeypatch):
    output_path = tmp_path / "prepared"
    loaded = prep.LoadedCells(
        adata=_raw_adata(),
        source={"mode": "local_h5ad", "sha256": hashlib.sha256(b"fixture").hexdigest()},
        sampling={"selected_cells": 8},
        qc={"cell_filters": {"enabled": False}},
    )

    monkeypatch.setattr(prep, "load_tabula_sapiens_census", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(
        prep,
        "prepare_representations",
        lambda *_args, **_kwargs: _prepared_fixture(),
    )

    # Keep this orchestration test independent of a local Parquet engine.
    def write_fake_parquet(path, _frame):
        Path(path).write_bytes(b"parquet fixture")

    monkeypatch.setattr(prep, "_atomic_parquet", write_fake_parquet)
    args = prep.parse_args(
        [
            "tabula-sapiens",
            "--output",
            str(output_path),
            "--n-hvg",
            "3",
            "--n-pcs",
            "2",
        ]
    )

    manifest = prep.execute(args)
    on_disk = json.loads((output_path / "manifest.json").read_text())
    assert manifest == on_disk
    timings = manifest["software"]["stage_timings_seconds"]
    statuses = manifest["software"]["stage_status"]
    required = {"source_load", "raw_cache", "preprocessing", "output_write"}
    assert required <= set(timings)
    assert set(timings) == set(statuses)
    allowed_statuses = {"executed", "cache_hit", "written", "disabled"}
    assert set(statuses.values()) <= allowed_statuses
    for stage, value in timings.items():
        if statuses[stage] == "disabled":
            assert value is None
        else:
            assert np.isfinite(value) and value >= 0
    assert statuses["source_load"] == "executed"
    assert statuses["raw_cache"] == "disabled"
    assert statuses["preprocessing"] == "executed"
    assert statuses["output_write"] in {"executed", "written"}


def test_selected_raw_cache_is_identity_keyed_atomic_validated_and_reused(tmp_path, monkeypatch):
    output_path = tmp_path / "prepared"
    cache_path = tmp_path / "selected-raw-cache"
    raw_payload = b"serialized selected raw AnnData"
    expected_source = {
        "mode": "cellxgene_census",
        "census_version": prep.DEFAULT_CENSUS_VERSION,
    }
    expected_sampling = {
        "selected_cells": 8,
        "sample_seed": prep.DEFAULT_SAMPLE_SEED,
    }
    expected_qc = {"cell_filters": {"enabled": False}}
    source_loads = []
    h5ad_writes = []
    h5ad_reads = []
    output_calls = []

    def write_h5ad(filename, *args, **kwargs):
        del args, kwargs
        path = Path(filename)
        h5ad_writes.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_payload)

    writer = write_h5ad

    def selected_adata():
        sparse = pytest.importorskip("scipy.sparse")
        adata = _raw_adata()
        adata.X = sparse.csr_matrix(adata.X)
        adata.obs["sampling_stratum"] = "fixture"
        for column in prep.QC_COLUMNS:
            adata.obs[column] = 0.0
        adata.write_h5ad = writer
        return adata

    def load_source(*args, **kwargs):
        source_loads.append((args, kwargs))
        return prep.LoadedCells(
            adata=selected_adata(),
            source=copy.deepcopy(expected_source),
            sampling=copy.deepcopy(expected_sampling),
            qc=copy.deepcopy(expected_qc),
        )

    def read_h5ad(filename, *args, **kwargs):
        del args, kwargs
        h5ad_reads.append(Path(filename))
        return selected_adata()

    monkeypatch.setattr(prep, "load_tabula_sapiens_census", load_source)
    monkeypatch.setitem(sys.modules, "anndata", SimpleNamespace(read_h5ad=read_h5ad))
    monkeypatch.setattr(
        prep,
        "prepare_representations",
        lambda *_args, **_kwargs: _prepared_fixture(),
    )

    def capture_outputs(*_args, **kwargs):
        output_calls.append(
            {
                "timings": dict(kwargs["stage_timings"]),
                "status": dict(kwargs["stage_status"]),
            }
        )
        return {"selection": {"selected_cells": 8}}

    monkeypatch.setattr(prep, "write_prepared_outputs", capture_outputs)

    argv = [
        "tabula-sapiens",
        "--output",
        str(output_path),
        "--n-hvg",
        "3",
        "--n-pcs",
        "2",
        "--raw-cache-dir",
        str(cache_path),
    ]
    args = prep.parse_args(argv)
    prep.execute(args)
    assert output_calls[-1]["status"]["raw_cache"] == "written"
    assert output_calls[-1]["status"]["source_load"] == "executed"

    cache_entries = [path for path in cache_path.iterdir() if path.is_dir()]
    assert len(cache_entries) == 1
    entry = cache_entries[0]
    checkpoint_path = entry / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    selected_name = checkpoint["selected_raw_h5ad"]["path"]
    assert prep.RAW_H5AD_GENERATION_PATTERN.fullmatch(selected_name)
    selected_raw = entry / selected_name
    entry_names = {path.name for path in entry.iterdir()}
    assert {selected_name, "checkpoint.json"} <= entry_names
    assert not any(".tmp" in name or name.endswith(".part") for name in entry_names)
    assert selected_raw.read_bytes() == raw_payload
    assert len(h5ad_writes) == 1
    assert h5ad_writes[0].resolve() != selected_raw.resolve()
    assert set(checkpoint) == {
        "schema_version",
        "request_identity",
        "identity_sha256",
        "selected_raw_h5ad",
        "semantic",
        "loaded",
    }
    assert len(checkpoint["identity_sha256"]) == 64
    int(checkpoint["identity_sha256"], 16)
    assert checkpoint["identity_sha256"][:16] in entry.name
    assert isinstance(checkpoint["request_identity"], dict)
    assert checkpoint["request_identity"]["hvg_batch_fields"] == ["donor_id", "assay"]
    assert set(prep.DEFAULT_HVG_BATCH_FIELDS) <= set(
        checkpoint["request_identity"]["observation_columns"]
    )
    assert isinstance(checkpoint["semantic"], dict)
    assert checkpoint["selected_raw_h5ad"] == {
        "path": selected_name,
        "sha256": hashlib.sha256(raw_payload).hexdigest(),
        "bytes": len(raw_payload),
    }
    assert set(checkpoint["loaded"]) == {
        "source",
        "sampling",
        "qc",
        "source_sha256",
        "sampling_sha256",
        "qc_sha256",
    }
    assert checkpoint["loaded"]["source"] == expected_source
    assert checkpoint["loaded"]["sampling"] == expected_sampling
    assert checkpoint["loaded"]["qc"] == expected_qc
    for field in ("source", "sampling", "qc"):
        assert checkpoint["loaded"][f"{field}_sha256"] == prep._sha256_json(
            checkpoint["loaded"][field]
        )

    # A process death can leave files not named by the committed manifest.
    # A validated cache hit must retain the committed generation and clean them.
    orphan_generation = entry / f"selected_raw.{'f' * 64}.h5ad"
    if orphan_generation == selected_raw:
        orphan_generation = entry / f"selected_raw.{'e' * 64}.h5ad"
    orphan_generation.write_bytes(b"uncommitted generation")
    orphan_temporary = entry / ".selected_raw.interrupted.tmp.h5ad"
    orphan_temporary.write_bytes(b"partial generation")

    # An identical request must read the committed checkpoint, not reload the source.
    prep.execute(prep.parse_args(argv))
    assert output_calls[-1]["status"]["raw_cache"] == "cache_hit"
    assert output_calls[-1]["status"]["source_load"] == "cache_hit"
    assert len(source_loads) == 1
    assert len(h5ad_reads) == 2
    assert h5ad_reads[-1] == selected_raw
    assert len(h5ad_writes) == 1
    assert not orphan_generation.exists()
    assert not orphan_temporary.exists()

    # Downstream representation settings do not change the selected-raw identity.
    prep.execute(
        prep.parse_args(
            [
                *argv,
                "--output",
                str(tmp_path / "alternate-prepared"),
                "--hvg-span",
                "0.625",
                "--n-pcs",
                "1",
            ]
        )
    )
    assert output_calls[-1]["status"]["raw_cache"] == "cache_hit"
    assert len(source_loads) == 1
    assert len(h5ad_writes) == 1

    # A failed refresh must leave the previously committed pair intact.
    committed_raw = selected_raw.read_bytes()
    committed_checkpoint = checkpoint_path.read_bytes()

    def fail_after_partial_write(filename, *args, **kwargs):
        del args, kwargs
        path = Path(filename)
        h5ad_writes.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial")
        raise RuntimeError("fixture checkpoint write failure")

    writer = fail_after_partial_write
    with pytest.raises(RuntimeError, match="fixture checkpoint write failure"):
        prep.execute(prep.parse_args([*argv, "--refresh-raw-cache"]))
    assert selected_raw.read_bytes() == committed_raw
    assert checkpoint_path.read_bytes() == committed_checkpoint
    entry_names = {path.name for path in entry.iterdir()}
    assert {selected_name, "checkpoint.json"} <= entry_names
    assert not any(".tmp" in name or name.endswith(".part") for name in entry_names)

    # Failure after replacing the H5AD but before committing checkpoint.json
    # must also roll the pair back to the prior committed generation.
    replacement_payload = b"different but semantically mocked selected raw AnnData"

    def write_replacement_h5ad(filename, *args, **kwargs):
        del args, kwargs
        path = Path(filename)
        h5ad_writes.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(replacement_payload)

    writer = write_replacement_h5ad
    original_atomic_json = prep._atomic_json

    def fail_manifest_commit(path, payload):
        if Path(path).name == "checkpoint.json":
            raise RuntimeError("fixture manifest commit failure")
        return original_atomic_json(path, payload)

    monkeypatch.setattr(prep, "_atomic_json", fail_manifest_commit)
    with pytest.raises(RuntimeError, match="fixture manifest commit failure"):
        prep.execute(prep.parse_args([*argv, "--refresh-raw-cache"]))
    monkeypatch.setattr(prep, "_atomic_json", original_atomic_json)
    assert selected_raw.read_bytes() == committed_raw
    assert checkpoint_path.read_bytes() == committed_checkpoint
    entry_names = {path.name for path in entry.iterdir()}
    assert {name for name in entry_names if prep.RAW_H5AD_GENERATION_PATTERN.fullmatch(name)} == {
        selected_name
    }
    assert not any(".tmp" in name for name in entry_names)

    # A committed checkpoint with a bad artifact hash must never be a silent miss.
    selected_raw.write_bytes(b"tampered")
    with pytest.raises(ValueError) as error:
        prep.execute(prep.parse_args(argv))
    assert "invalid selected-raw checkpoint" in str(error.value)
    assert "--refresh-raw-cache" in str(error.value)

    # Explicit refresh is the recovery path and atomically replaces the bad pair.
    writer = write_h5ad
    prep.execute(prep.parse_args([*argv, "--refresh-raw-cache"]))
    assert selected_raw.read_bytes() == raw_payload
    assert output_calls[-1]["status"]["raw_cache"] == "refreshed"

    # Identity metadata is validated independently of the H5AD content hash.
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["identity_sha256"] = "0" * 64
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError) as error:
        prep.execute(prep.parse_args(argv))
    assert "invalid selected-raw checkpoint" in str(error.value)
    assert "--refresh-raw-cache" in str(error.value)


def test_failed_cold_pilot_retries_from_committed_raw_checkpoint(tmp_path, monkeypatch):
    sparse = pytest.importorskip("scipy.sparse")
    monkeypatch.setattr(prep, "CENSUS_PILOT_CELLS", 8)
    cache_path = tmp_path / "raw-cache"
    output = tmp_path / "prepared"
    source_calls = []
    preparation_calls = []
    output_statuses = []

    def selected_adata():
        adata = _raw_adata()
        adata.X = sparse.csr_matrix(adata.X)
        adata.obs["sampling_stratum"] = "fixture"
        for column in prep.QC_COLUMNS:
            adata.obs[column] = 0.0
        adata.write_h5ad = lambda path, **_kwargs: Path(path).write_bytes(
            b"checkpointed raw fixture"
        )
        return adata

    def load_source(*_args, **_kwargs):
        source_calls.append(True)
        return prep.LoadedCells(
            adata=selected_adata(),
            source={"mode": "cellxgene_census"},
            sampling={"selected_cells": 8},
            qc={"cell_filters": {"enabled": False}},
        )

    def prepare(*_args, **_kwargs):
        preparation_calls.append(True)
        if len(preparation_calls) == 1:
            raise prep.HVGLoessDiagnosticsError(
                "fixture fixed-span failure",
                [{"batch_label": "d1|10x", "status": "failed"}],
            )
        return _prepared_fixture()

    monkeypatch.setattr(prep, "load_tabula_sapiens_census", load_source)
    monkeypatch.setitem(
        sys.modules,
        "anndata",
        SimpleNamespace(read_h5ad=lambda *_args, **_kwargs: selected_adata()),
    )
    monkeypatch.setattr(prep, "prepare_representations", prepare)
    monkeypatch.setattr(
        prep,
        "write_prepared_outputs",
        lambda *_args, **kwargs: output_statuses.append(dict(kwargs["stage_status"])) or {},
    )
    args = prep.parse_args(
        [
            "tabula-sapiens",
            "--output",
            str(output),
            "--max-cells",
            "8",
            "--n-hvg",
            "3",
            "--n-pcs",
            "2",
            "--raw-cache-dir",
            str(cache_path),
        ]
    )

    with pytest.raises(prep.HVGLoessDiagnosticsError, match="fixed-span failure"):
        prep.execute(args)
    assert len(source_calls) == 1
    assert len(list(cache_path.glob("tabula-sapiens-*/checkpoint.json"))) == 1
    assert (tmp_path / "prepared.hvg-loess-failure.json").is_file()

    prep.execute(args)
    assert len(source_calls) == 1
    assert output_statuses[-1]["raw_cache"] == "cache_hit"
    assert output_statuses[-1]["source_load"] == "cache_hit"
    assert not (tmp_path / "prepared.hvg-loess-failure.json").exists()


def test_selected_raw_checkpoint_round_trips_real_sparse_anndata(tmp_path, monkeypatch):
    anndata = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    sparse = pytest.importorskip("scipy.sparse")
    obs = pd.DataFrame(
        {
            "soma_joinid": np.arange(10, 14, dtype=np.int64),
            "donor_id": pd.Categorical(["d2", "d1", "d2", "d1"]),
            "assay": pd.Categorical(["10x", "10x", "smart", "smart"]),
            "sampling_stratum": ["a", "b", "a", "b"],
            "total_counts": [3.0, 4.0, 5.0, 6.0],
        },
        index=[f"cell-{index}" for index in range(4)],
    )
    var = pd.DataFrame(
        {"feature_id": ["ENSG1", "ENSG2", "ENSG3"]},
        index=["ENSG1", "ENSG2", "ENSG3"],
    )
    matrix = sparse.csr_matrix(
        np.asarray([[1, 0, 2], [0, 4, 0], [2, 1, 2], [0, 0, 6]], dtype=np.int32)
    )
    loaded = prep.LoadedCells(
        adata=anndata.AnnData(X=matrix, obs=obs, var=var),
        source={"mode": "cellxgene_census", "census_version": prep.DEFAULT_CENSUS_VERSION},
        sampling={"selected_cells": 4, "seed": 11},
        qc={"cell_filters": {"enabled": False}},
    )
    identity = {"fixture": "real-anndata", "schema": prep.RAW_CHECKPOINT_SCHEMA_VERSION}
    required = ("soma_joinid", "donor_id", "assay", "sampling_stratum", "total_counts")

    written = prep.write_selected_raw_checkpoint(
        tmp_path,
        loaded=loaded,
        request_identity=identity,
        required_obs_columns=required,
    )
    cached = prep.load_selected_raw_checkpoint(
        tmp_path,
        request_identity=identity,
        required_obs_columns=required,
    )

    assert cached is not None
    restored, record = cached
    assert written["cache_hit"] is False
    assert record["cache_hit"] is True
    assert written["semantic_sha256"] == record["semantic_sha256"]
    np.testing.assert_array_equal(restored.adata.X.toarray(), matrix.toarray())
    assert list(restored.adata.obs["donor_id"].astype(str)) == ["d2", "d1", "d2", "d1"]
    assert restored.source["selected_raw_checkpoint"] == record
    assert restored.sampling == loaded.sampling
    assert restored.qc == loaded.qc

    def fail_with_memory_pressure(*_args, **_kwargs):
        raise MemoryError("fixture allocation failure")

    monkeypatch.setattr(anndata, "read_h5ad", fail_with_memory_pressure)
    with pytest.raises(MemoryError, match="allocation failure"):
        prep.load_selected_raw_checkpoint(
            tmp_path,
            request_identity=identity,
            required_obs_columns=required,
        )


def test_selected_raw_semantics_include_gene_symbols_and_require_sparse_storage():
    sparse = pytest.importorskip("scipy.sparse")
    adata = _raw_adata()
    required = ("soma_joinid", "donor_id", "assay")

    with pytest.raises(ValueError, match="sparse CSR"):
        prep._selected_raw_semantic_record(adata, required_obs_columns=required)

    adata.X = sparse.csr_matrix(adata.X)
    original = prep._selected_raw_semantic_record(adata, required_obs_columns=required)
    adata.var.loc[adata.var.index[0], "feature_name"] = "changed-symbol"
    changed = prep._selected_raw_semantic_record(adata, required_obs_columns=required)

    assert original["feature_names_sha256"] != changed["feature_names_sha256"]
    assert original["semantic_sha256"] != changed["semantic_sha256"]


def test_cli_defaults_and_exposes_fixed_hvg_span_and_selected_raw_cache(tmp_path):
    common = [
        "tabula-sapiens",
        "--output",
        str(tmp_path / "prepared"),
    ]
    args = prep.parse_args(common)
    assert prep.DEFAULT_HVG_SPAN == pytest.approx(0.5)
    assert args.hvg_span == pytest.approx(0.5)
    assert args.raw_cache_dir is None
    assert args.refresh_raw_cache is False

    configured = prep.parse_args(
        [
            *common,
            "--hvg-span",
            "0.625",
            "--raw-cache-dir",
            str(tmp_path / "raw-cache"),
            "--refresh-raw-cache",
        ]
    )
    assert configured.hvg_span == pytest.approx(0.625)
    assert configured.raw_cache_dir == tmp_path / "raw-cache"
    assert configured.refresh_raw_cache is True


def test_refresh_raw_cache_requires_a_cache_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prep,
        "load_tabula_sapiens_census",
        lambda *_args, **_kwargs: pytest.fail("source loaded before cache CLI validation"),
    )
    argv = [
        "tabula-sapiens",
        "--output",
        str(tmp_path / "prepared"),
        "--refresh-raw-cache",
    ]
    with pytest.raises((SystemExit, ValueError)) as error:
        args = prep.parse_args(argv)
        prep.execute(args)
    if isinstance(error.value, ValueError):
        assert "--raw-cache-dir" in str(error.value)


def test_short_census_pilot_is_rejected_before_checkpoint_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "CENSUS_PILOT_CELLS", 8)
    loaded = prep.LoadedCells(
        adata=_raw_adata(rows=4),
        source={"mode": "cellxgene_census"},
        sampling={"selected_cells": 4},
        qc={},
    )
    checkpoint_calls = []
    monkeypatch.setattr(prep, "load_selected_raw_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prep, "load_tabula_sapiens_census", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(
        prep,
        "write_selected_raw_checkpoint",
        lambda *_args, **_kwargs: checkpoint_calls.append(True),
    )
    monkeypatch.setattr(
        prep,
        "prepare_representations",
        lambda *_args, **_kwargs: pytest.fail("preprocessing reached after a short pilot"),
    )
    args = prep.parse_args(
        [
            "tabula-sapiens",
            "--output",
            str(tmp_path / "prepared"),
            "--max-cells",
            "8",
            "--raw-cache-dir",
            str(tmp_path / "raw-cache"),
        ]
    )

    with pytest.raises(RuntimeError, match="exactly 8 cells"):
        prep.execute(args)

    assert checkpoint_calls == []


def test_cli_rejects_moving_release_and_nonstandard_census_pilot_size(tmp_path):
    with pytest.raises(ValueError, match="exact YYYY-MM-DD"):
        prep.validate_pinned_census_version("latest")
    args = prep.parse_args(
        [
            "tabula-sapiens",
            "--output",
            str(tmp_path),
            "--max-cells",
            "99999",
        ]
    )
    with pytest.raises(ValueError, match="exactly 100000"):
        prep.execute(args)
