Single-cell RNA-seq reducer benchmark protocol
==============================================

.. warning::

   **Status: planned, not run.**  This page specifies the experiment before
   inspecting any outcomes.  It contains no single-cell benchmark results and
   makes no performance claim about PCA, the autoencoder, DiRe, or UMAP.

Material passport
-----------------

``Artifact type``
   Prospective code-experiment protocol.

``Research question``
   At fixed target widths, how do a reconstruction-trained autoencoder, DiRe,
   RAPIDS cuML UMAP, and PCA trade off biological-label preservation, local and
   global geometry, technical-confound retention, runtime, and memory on sparse,
   noisy single-cell RNA-seq data?

``Protocol date``
   2026-07-15.

``Evidence state``
   Official dataset and package documentation has been checked; data have not
   been downloaded, prepared, embedded, or scored.

``Primary sources``
   The versioned `CELLxGENE Census releases`_, the `Tabula Sapiens collection`_,
   the `10x 1.3M brain-cell page`_, and the Scanpy documentation for
   `highly-variable genes`_ and `sparse PCA`_.

Questions and estimands
-----------------------

The main comparison is deliberately fixed-width rather than an intrinsic-
dimension contest.  Every reducer receives the same prepared matrix, labels are
withheld during fitting, and target dimensions ``d=2`` and ``d=8`` are evaluated
independently.  The primary questions are:

* how faithfully each embedding preserves the geometry of its stated input;
* how much annotated cell-type information remains accessible without using
  labels to fit the reducer;
* whether donor, assay, library depth, or other technical effects dominate an
  apparently well-separated layout; and
* how end-to-end time and peak memory change from 100,000 to roughly one million
  cells.

There will be no single composite winner score.  Geometry, biological signal,
technical effects, and systems cost answer different questions and will be
reported as separate profiles.

Datasets and scale tiers
------------------------

Tabula Sapiens: biological benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source is the CELLxGENE collection
``e5f58829-1a66-40b5-a624-9046778e74f5``.  The public collection has expanded
since the original approximately 483,000-cell atlas and currently describes an
all-cells dataset with more than one million cells.  To prevent the contents from
silently changing, preparation must open Census release ``2025-11-08`` rather
than the aliases ``stable`` or ``latest``.  This is a long-term-supported release;
the release documentation identifies ``cellxgene-census`` 1.17.x as its
compatible reader line.

The preparation code must resolve the collection's ``Tabula Sapiens - All
Cells`` dataset inside that pinned Census, retain only primary data, and write the
resolved dataset ID and actual before/after cell counts to ``manifest.json``.
No expected full-cohort count is hard-coded into the analysis.

Two tiers use that one frozen query:

``tabula-pilot-100k``
   Exactly 100,000 eligible cells, selected deterministically with seed
   ``20260715``.  Sampling is stratified jointly by donor and annotated cell type
   while preserving approximate population proportions.  The manifest records
   allocations and any strata that cannot be represented.  This tier is the
   first scientific run and exercises both input arms, both target widths, and
   all three reducer seeds.

``tabula-full``
   Every eligible primary cell returned by the identical pinned query.  It is run
   only after the pilot passes the checks below.  Biological metrics use donor-
   held-out evaluation, so cells from a donor never appear in both training and
   test folds of a label probe.

Tabula Sapiens combines tissues, donors, and assay technologies.  That complexity
is useful, but it makes global donor- or assay-mixing scores easy to misread.  It
is therefore the biological and confounding benchmark, not a pre-integrated
ground truth.

10x 1.3M brain cells: scale benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scale tier uses the official filtered HDF5 feature-barcode matrix from the
`10x 1.3M brain-cell page`_.  The page reports 1,306,127 cells from embryonic
mouse brain and two mice.  The preparation manifest records the actual matrix
shape, response metadata, byte size, and SHA-256 digest rather than assuming the
advertised count.

This dataset is primarily a systems and geometry test.  With only two source
animals and no expert cell-type labels supplied as ground truth on the download
page, Cell Ranger clusters must not be promoted to independent biological labels.
If those clusters are used, they are shown only as descriptive overlays.

The later 5M--10M CELLxGENE or requester-pays atlas tier is explicitly out of
scope for this first experiment.  It should be attempted only after the 1.3M-cell
run establishes storage, graph-construction, and metric-sampling behavior.

Preprocessing contract
----------------------

Source matrices are already distributed as curated/filtered cell matrices.  A
single universal mitochondrial-percentage cutoff across tissues and assays could
remove real biology, so the primary Census protocol applies no additional cell
threshold.  It validates that the count matrix is finite, nonnegative, and
integer-valued and removes genes detected in fewer than three retained cells,
recording the before/after counts.  It then computes and retains total counts,
detected-gene count, and mitochondrial fraction for later leakage checks.  Any
stricter cell QC is a separately named sensitivity analysis.

Preparation then performs these frozen operations:

1. Preserve the integer count matrix in sparse CSR form for provenance and HVG
   selection.
2. Select 2,000 highly variable genes with Scanpy's ``seurat_v3`` method on raw
   counts.  For Tabula Sapiens, construct the batch key as the literal
   concatenation ``donor_id|assay`` so a gene specific to one donor or library
   technology is not automatically preferred; do not batch by tissue or cell
   type.  The 10x matrix uses library identity when that metadata is available.
   ``seurat_v3`` requires ``scikit-misc`` and expects counts rather than log
   values.  Freeze the LOESS span at ``0.5`` (the preparation default and the
   explicit run-book value).  A LOESS error, a non-finite diagnostic, or failure
   to return exactly 2,000 genes is a failed preparation: the program must not
   silently change the span, switch HVG flavor, remove batching, or substitute a
   different feature set.
   The command rejects cell-type and tissue columns passed through
   ``--hvg-batch-fields``; that option is restricted to technical batches.
3. Normalize every cell to a total of 10,000, then apply ``log1p``.
4. Center each HVG, scale it to unit variance, and clip values to ``[-10, 10]``.
   Fit all scaling statistics on the complete transductive cohort and record them.
   Before scaling sparse input, ``--max-dense-bytes`` checks the pinned Scanpy
   float64 dense-result lower bound.  It is a per-matrix guard, not a peak-RSS
   limit; the run must still record measured host memory.
5. Compute 50 non-whitened principal components with a fixed randomized-SVD seed.
   Do not row-L2-normalize the resulting matrix.

The manifest fixes the Seurat-v3 fit as ``span=0.5``, ``degree=2``,
``span_policy=fixed_global``, and ``fallback=none``.  Before fitting, it records
one conditioning row per donor-by-assay batch with ``batch_label``,
``cell_count``, ``nonconstant_genes``, ``distinct_log10_means``,
``duplicate_log10_mean_fraction``, ``loess_neighborhood_points``,
``loess_applied``, and ``status``.  These diagnostics identify the batch
responsible for an invalid LOESS fit; they do not authorize a batch-specific
parameter change or retry.  A
failed fit or an HVG result with the wrong feature count writes
``<output-name>.hvg-loess-failure.json`` beside the requested output directory
with the complete diagnostics, failed-stage timing, and checkpoint identity,
then exits nonzero without a final preparation manifest.

The same retained cells and HVG order must be used in both arms.  Dataset labels,
cell types, tissues, donors, and assay names may guide sampling and post hoc
metrics, but no reducer may see them during fitting.

Prepared input arms
-------------------

``pca50`` (primary)
   The dense ``N x 50`` non-whitened PCA representation.  This is the conventional
   substrate for neighbour-graph construction in single-cell workflows and the
   scalable all-method comparison.  It is written as ``pca50.npy``.

``log_hvg2000`` (sensitivity)
   The dense ``N x 2000`` scaled log-HVG matrix immediately before PCA, written as
   ``log_hvg2000.npy``.  It tests whether the 50-PC prepass changes the method
   ordering.  It is required for the 100k pilot and optional for full-scale tiers
   because its decoder and distance costs are much larger.

These arms define different reference geometries.  A trustworthiness value
relative to ``pca50`` and one relative to ``log_hvg2000`` must not be pooled or
treated as measurements of the same estimand.  Raw counts are not a headline
arm: MSE and Euclidean distance on unnormalized counts largely measure sequencing
depth.  A raw-count run, if added, is named and interpreted as a negative control.

Reducers and paired settings
----------------------------

Each arm is reduced by PCA, the autoencoder, DiRe, and RAPIDS cuML UMAP to ``d=2``
and ``d=8`` with seeds 0, 1, and 2.  The PCA result is the first ``d`` components
of a fresh PCA fit to the arm and supplies a linear baseline.  ``k=30`` is the
primary neighbourhood width; ``k=15`` and ``k=50`` are optional metric-only
sensitivity values.

All methods receive identical ``float32`` values in identical row order.  No
method-specific centering, whitening, row normalization, label supervision, or
pretrained model is allowed unless it becomes a separately named ablation.  The
autoencoder is initialized and trained from scratch for every dataset, arm,
target width, and seed.  Its end-to-end time includes staging, validation-based
epoch selection, the fresh full-data refit, and encoding.

The current implementations are dense-first: DiRe and the autoencoder do not
consume CSR inputs, and RAPIDS UMAP's sparse path does not provide the same
scalable NN-descent path used for dense arrays.  The PCA50 arm is therefore a
fair shared substrate, not an optimization granted to one reducer.  The direct
HVG pilot makes the cost and scientific effect of that choice visible.

Metrics
-------

Geometry relative to the arm input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* trustworthiness, continuity, and exact neighbour overlap at ``k=30`` on one
  fixed evaluation sample shared by every method and seed;
* Spearman correlation and scale-aligned stress on one fixed set of random cell
  pairs; and
* optional Betti-curve summaries on identical point samples, explicitly labelled
  descriptive rather than inferential topology.

The initial harness emits trustworthiness, continuity, exact neighbour overlap,
pair-distance correlation, and stress.  Betti curves and an internal approximate-
graph recall audit are planned follow-ons.  Before approximate graph results can
be interpreted as reducer effects, the follow-on audit must report recall against
an exact reference on a tractable fixed sample; approximation error must not be
silently attributed to the reducer.

Biological-label preservation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* donor-held-out macro-F1 and balanced accuracy for a fixed, simple cell-type
  probe trained only after embedding;
* donor-minimum-support per-cell-type F1 and a sampled cell-type average
  silhouette width, with rare eligible types shown individually rather than
  hidden by a micro-average; and
* local label-enrichment and tissue-stratified summaries where coverage permits
  (planned follow-ons).

These probes are transductive embedding diagnostics, not evidence that a reducer
generalizes to unseen cells.  Folds are grouped by donor rather than randomized by
cell because cells from one donor are not independent replicates.  A label is
eligible for the held-out summary only when it has prespecified minimum support
across at least three donors; all exclusions and support counts are retained.

Technical effects and confounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial harness reports leave-one-donor-out predictability of total counts,
detected genes, and mitochondrial fraction from the embedding, including
``R^2``, normalized MSE, Spearman correlation, fold count, and sampled-row
identity.  Higher technical-covariate predictability means greater leakage.

Donor/assay silhouette, LISI-type diagnostics, and cell-type-by-donor coverage
tables are planned follow-ons rather than outputs promised by the initial
harness.  They must be computed only within cell types having adequate
cross-batch coverage.  Tissue-stratified and LISI summaries become release gates
only after their implementations and tests land.

The reducers are not batch-integration methods.  Global batch mixing is therefore
not rewarded, particularly when donor, tissue, assay, and cell type are
confounded.  Batch metrics are diagnostics of retained technical structure, not
standalone quality scores.

Systems measurements
~~~~~~~~~~~~~~~~~~~~

Record preparation, transfer, graph construction, fitting/training, encoding,
metric, and total wall time separately, together with peak host RAM, peak GPU
memory, output size, and any failure or fallback.  Both reducer-only and complete
download-excluded pipeline times are reported.  Download time is provenance and
operations context, not reducer runtime.  For Census input, the preparation
manifest records separate wall times for dataset resolution, candidate-metadata
selection, deterministic sampling, raw-count materialization, and the combined
alignment/QC/gene-filter step.  Local and 10x source work is represented by the
aggregate ``source_load`` stage.  All sources additionally record count
validation, HVG diagnostics and selection, normalization/logging, scaling, PCA,
output-table construction, and artifact serialization.  The recorded
``output_write`` and ``total_before_manifest`` values stop immediately before
the final manifest is serialized, so neither is labeled complete end-to-end
preparation time.  Cache-hit status is recorded so a
checkpoint-assisted retry cannot be compared to a cold Census read as if the two
timings measured the same work.  The manifest fields
``software.stage_timings_seconds`` and ``software.stage_status`` have identical
stage keys, including at least ``source_load``, ``raw_cache``,
``preprocessing``, and ``output_write``; a skipped stage is represented
explicitly rather than reported as a zero-duration execution.

Replication and interpretation
------------------------------

Reducer seeds are paired across methods, and every stochastic cell/pair sample is
materialized once and reused.  The primary summaries report the mean, sample
standard deviation, and paired seed differences.  Seeds measure algorithmic
variability; they do not turn one dataset into three biological replicates.
Dataset, donor, and cell-type effects therefore remain descriptive unless a later
multi-dataset analysis supplies an appropriate inferential unit.

The pilot advances to full Tabula Sapiens only if all expected rows are present,
all embeddings and metrics are finite, no reducer silently changes input rows or
normalization, and memory headroom is adequate.  If an approximate internal graph
is used, advancement additionally waits for the planned exact-recall audit.  The
10x tier advances only after the full/PCA50 path demonstrates that preprocessing
and sampled metrics do not construct quadratic full-corpus arrays.

Expected artifacts
------------------

Large local artifacts live under ``artifacts/single-cell/`` and remain ignored by
Git.  Each prepared tier contains:

* ``pca50.npy``;
* optional ``log_hvg2000.npy``;
* ``metadata.parquet`` with row identity, labels, sampling stratum, and QC
  covariates;
* ``genes.parquet`` with stable feature IDs, symbols, HVG ranks, and PCA loadings
  or a reference to them; and
* ``manifest.json`` with URLs, Census release, collection and dataset IDs,
  source hashes/sizes, shapes, filters, sampling allocations, preprocessing
  parameters, seeds, stage timings, cache status, package resolution, and
  artifact hashes.

The selected raw-count checkpoint is deliberately *not* one of these published
outputs.  It lives in a separate checkpoint directory beside, rather than
inside, the final prepared tier.  The final ``manifest.json`` is the commit
marker and is written only after ``pca50.npy``, metadata, genes, and the optional
direct-HVG arm have been serialized and hashed.  Consumers must require and
validate that manifest rather than treating a directory's mere existence as a
completed tier.  If HVG selection or a later stage fails, the validated raw
checkpoint remains available for a retry without writing a final manifest.

Each experiment output contains layouts or checkpoints under ``artifacts/`` and
compact ``results.csv``, ``results.json``, raw/aggregate metric tables, figures,
logs, and a copied input manifest under ``docs/results/single-cell-*``.  A future
result page must retain the words ``planned, not run`` until all claimed cells
have completed successfully.

Pinned preparation environment
------------------------------

Preparation is isolated from RAPIDS because the benchmark's RAPIDS 26.06
environment uses Python 3.14, while ``cellxgene-census==1.17.0`` declares Python
3.10--3.12 support.  Version 1.17 is pinned because the Census release page names
the 1.17.x line as compatible with LTS ``2025-11-08``.  Scanpy 1.12.2 is pinned to
freeze preprocessing behavior, ``scikit-misc`` 0.5.2 supplies the Seurat-v3 LOESS
implementation.  The initial harness computes its metrics directly and does not
install scIB Metrics (and therefore does not pull JAX into the preparation
environment); scIB remains a reference implementation for planned LISI-type
follow-ons.  Each manifest captures the selected preparation-package versions;
retain the environment's complete ``uv pip freeze`` output in the run log when
full transitive resolution is required.  ``requirements/single-cell-prep.txt``
is not a complete lock file.

Restartable raw-count preparation
---------------------------------

The Census pilot opts into a selected-cell raw-count checkpoint with
``--raw-cache-dir``; preparation does not create or consult a raw cache by
default.  The cache root contains an identity-keyed directory named
``tabula-sapiens-<request-hash-prefix>/``, with a content-addressed
``selected_raw.<file-sha256>.h5ad`` generation and ``checkpoint.json`` inside
it.  The manifest is committed last and names the sole live generation.  The
request identity covers the pinned Census release, organism, collection and
dataset title, raw-layer and primary-data
selection, sampling and exported-metadata fields, seed, requested cell count,
and the detected-in-three-cells gene-filter setting.  Checkpoint validation also
binds resolved-dataset provenance, the selected ``soma_joinid`` sequence, stable
feature IDs, raw matrix shape, QC/sampling records, and the required observation
columns.  The checkpoint payload and identity metadata carry SHA-256 digests.  A
cache hit is accepted only after those values, the exact cell/feature order, and
the finite, nonnegative, integer-count invariants validate against the current
request.

Reuse is conservative.  A matching valid checkpoint skips Census X
materialization and continues from the validated, gene-filtered raw counts to
HVG selection.  A cache miss creates the identity-keyed checkpoint; a changed
request selects a different identity directory instead of overwriting an
unrelated cohort.  A committed checkpoint whose file, semantic, provenance, or
identity hashes do not validate is a hard error rather than an invitation to use
stale cells silently.  A directory without ``checkpoint.json`` has no commit
marker and is treated as an uncommitted cache miss that may be safely rebuilt.
``--refresh-raw-cache`` requires ``--raw-cache-dir`` and explicitly rebuilds the
current request's checkpoint under its lock.  A refresh interrupted before the
manifest commit leaves the previous committed generation valid; unreferenced
temporary or generation files are ignored and removed by the next successful
checkpoint operation.  A committed generation that later fails file, semantic,
provenance, or identity validation remains a hard error and must be refreshed
explicitly.
Final preparation artifacts are never read back as an implicit raw cache.

One preparation process is the intended execution model.  On this 64-core host,
the installed TileDB-SOMA defaults already provide 64 compute threads, 64 I/O
threads, 64 concurrent S3 operations, and a 32-worker SOMA executor.
``get_anndata`` partitions a sufficiently large X read across that executor;
wrapping the command in outer multiprocessing would multiply remote reads,
buffers, and thread pools without creating a new independent data partition.
Outer multiprocessing is therefore not recommended.  Later Scanpy and
randomized-PCA stages also use their own Numba/OpenMP/BLAS pools, so concurrency
changes, if investigated, must be an explicit measured systems ablation rather
than an undocumented run-book tweak.

Proposed commands (not yet executed)
------------------------------------

The following commands are the prospective run book.  They are documented before
execution and **have not been run**.  The preparation CLI is expected to reject a
changed dataset schema or missing donor/label columns rather than silently choose
fallback fields.

Create the Python 3.12 preparation environment:

.. code-block:: bash

   uv venv --python /home/igor/miniforge3/bin/python /tmp/drb-scrna-prep
   uv pip install --python /tmp/drb-scrna-prep/bin/python \
       -r requirements/single-cell-prep.txt

Prepare the deterministic Tabula Sapiens pilot and full cohort:

.. code-block:: bash

   /tmp/drb-scrna-prep/bin/python experiments/prepare_single_cell.py tabula-sapiens \
       --census-version 2025-11-08 \
       --collection-id e5f58829-1a66-40b5-a624-9046778e74f5 \
       --dataset-title "Tabula Sapiens - All Cells" \
       --max-cells 100000 \
       --sample-seed 20260715 \
       --stratify-by donor_id,cell_type \
       --min-cells-per-gene 3 \
       --n-hvg 2000 \
       --hvg-batch-fields donor_id,assay \
       --hvg-span 0.5 \
       --target-sum 10000 \
       --scale --scale-clip 10 \
       --n-pcs 50 \
       --write-log-hvg \
       --max-dense-bytes 34359738368 \
       --raw-cache-dir artifacts/single-cell/.raw-cache \
       --output artifacts/single-cell/tabula-pilot-100k

   /tmp/drb-scrna-prep/bin/python experiments/prepare_single_cell.py tabula-sapiens \
       --census-version 2025-11-08 \
       --collection-id e5f58829-1a66-40b5-a624-9046778e74f5 \
       --dataset-title "Tabula Sapiens - All Cells" \
       --n-hvg 2000 --hvg-span 0.5 --n-pcs 50 \
       --max-dense-bytes 17179869184 \
       --output artifacts/single-cell/tabula-full

The first pilot invocation creates its identity-keyed subdirectory under
``artifacts/single-cell/.raw-cache``; the same command later validates and reuses
it.  Keep that cache root outside
``artifacts/single-cell/tabula-pilot-100k``.  To deliberately discard and rebuild
the matching checkpoint, append ``--refresh-raw-cache`` while retaining
``--raw-cache-dir``.  Refresh is an operator decision after inspecting the
validation failure, not an automatic response to one.  The full-cohort command
above intentionally has no raw-cache option until the pilot establishes the
storage cost of retaining a second full raw matrix.

Prepare the 10x scale tier from the official filtered matrix:

.. code-block:: bash

   /tmp/drb-scrna-prep/bin/python experiments/prepare_single_cell.py tenx-brain-1m3 \
       --matrix-url https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5 \
       --n-hvg 2000 --hvg-span 0.5 --n-pcs 50 \
       --max-dense-bytes 17179869184 \
       --output artifacts/single-cell/tenx-brain-1m3

Run the paired PCA50 pilot in the existing ``rapids-26.06`` environment:

.. code-block:: bash

   conda run -n rapids-26.06 python experiments/compare_single_cell_reducers.py \
       --data artifacts/single-cell/tabula-pilot-100k/pca50.npy \
       --metadata artifacts/single-cell/tabula-pilot-100k/metadata.parquet \
       --manifest artifacts/single-cell/tabula-pilot-100k/manifest.json \
       --dataset-name tabula-sapiens-pilot-100k \
       --input-representation pca50 \
       --methods pca autoencoder dire umap \
       --target-dimensions 2 8 --seeds 0 1 2 --neighbors 30 \
       --cell-type-column cell_type --donor-column donor_id \
       --technical-columns total_counts n_genes_by_counts pct_counts_mt \
       --ae-amp \
       --dire-repo /home/igor/devel/dire-rapids \
       --dire-revision 0014f39818d0b8f967e0b84f06f904d7bc5db3a6 \
       --validate-input --dry-run \
       --output-dir artifacts/single-cell/tabula-pilot-100k/benchmark-pca50

After inspecting the validation report, remove **both** ``--dry-run`` and
``--validate-input`` to launch reducers: either option intentionally returns
before fitting.  The harness does not normalize prepared values;
``--validate-input`` checks artifact identity, shape, row alignment, and manifest
expectations without allocating a reducer.  ``--ae-amp`` selects the previously
validated mixed-precision AE training path.  The DiRe checkout and revision are
both explicit so the worker can reject source drift.

Run the direct-HVG sensitivity only after the PCA50 pilot passes:

.. code-block:: bash

   conda run -n rapids-26.06 python experiments/compare_single_cell_reducers.py \
       --data artifacts/single-cell/tabula-pilot-100k/log_hvg2000.npy \
       --metadata artifacts/single-cell/tabula-pilot-100k/metadata.parquet \
       --manifest artifacts/single-cell/tabula-pilot-100k/manifest.json \
       --dataset-name tabula-sapiens-pilot-100k \
       --input-representation log_hvg2000 \
       --methods pca autoencoder dire umap \
       --target-dimensions 2 8 --seeds 0 1 2 --neighbors 30 \
       --cell-type-column cell_type --donor-column donor_id \
       --technical-columns total_counts n_genes_by_counts pct_counts_mt \
       --ae-amp \
       --dire-repo /home/igor/devel/dire-rapids \
       --dire-revision 0014f39818d0b8f967e0b84f06f904d7bc5db3a6 \
       --validate-input --dry-run \
       --output-dir artifacts/single-cell/tabula-pilot-100k/benchmark-log-hvg2000

Full-cohort commands use the same options after replacing the three input paths,
``--dataset-name``, and ``--output-dir`` with the corresponding ``tabula-full``
or ``tenx-brain-1m3`` values.  The 10x run omits label and donor options unless
the prepared manifest supplies independently justified fields.  Both runs remain
gated on the pilot checks and require a separate explicit launch.

Sources
-------

* `CELLxGENE Census releases`_
* `CELLxGENE Python API`_
* `Tabula Sapiens collection`_
* `Tabula Sapiens open-data registry`_
* `10x 1.3M brain-cell page`_
* Scanpy `highly-variable genes`_, `total-count normalization`_, and `sparse PCA`_
* `scIB Metrics documentation`_

.. _CELLxGENE Census releases: https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_data_release_info.html
.. _CELLxGENE Python API: https://chanzuckerberg.github.io/cellxgene-census/python-api.html
.. _Tabula Sapiens collection: https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5
.. _Tabula Sapiens open-data registry: https://registry.opendata.aws/tabula-sapiens/
.. _10x 1.3M brain-cell page: https://www.10xgenomics.com/datasets/1-3-million-brain-cells-from-e-18-mice-2-standard-1-3-0
.. _highly-variable genes: https://scanpy.readthedocs.io/en/stable/api/scanpy.pp.highly_variable_genes.html
.. _total-count normalization: https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.normalize_total.html
.. _sparse PCA: https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html
.. _scIB Metrics documentation: https://scib-metrics.readthedocs.io/
