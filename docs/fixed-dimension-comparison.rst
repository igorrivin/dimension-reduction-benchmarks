Fixed-Target Reducer Comparison
===============================

This experiment asks a different question from intrinsic-dimension estimation: once the
target width is fixed, how does a neural autoencoder embedding compare with DiRe and
RAPIDS cuML UMAP?

Protocol
--------

The comparison is transductive because DiRe currently exposes ``fit`` and
``fit_transform`` but no out-of-sample ``transform`` or ``inverse_transform``. Every
method therefore embeds the same complete matrix at the same target dimension. Labels
are never passed to a reducer.

* Data: 1,000 observations per dataset, three paired seeds (0, 1, 2).
* Synthetic data: rank-``d`` linear subspace, Swiss roll, sphere, and flat torus,
  orthogonally embedded in 32 dimensions with small isotropic noise where applicable.
* Real data: stratified 1,000-row subsets of scikit-learn digits.
* Preprocessing: one shared mean-center and global RMS scale; no per-feature rescaling.
* Graph settings: 15 neighbours for DiRe and UMAP; Euclidean metric throughout.
* Autoencoder: ``D-128-64-d-64-128-D``, linear bottleneck, SiLU, AdamW, validation-based
  epoch selection followed by a full-data refit, under a 200-epoch cap.
* DiRe: PCA initialization, 128 layout iterations, ``normalize=False`` because shared
  preprocessing has already been applied, with ``knn_backend="pytorch"`` pinned.
* UMAP: brute-force kNN, serial epochs, ``min_dist=0.1`` and ``spread=1``.
* Hardware: device name ``NVIDIA GH200 480GB`` with 97,871 MiB CUDA-visible memory,
  driver 590.48.01, compute capability 9.0.
* Software: RAPIDS 26.06 and DiRe-RAPIDS 0.3.1 at exact revision
  ``f7d57367af8e10f6013e64026e2dbb0af6781947``. Raw artifacts also record the
  DiRe implementation SHA-256 and benchmark-harness SHA-256.

The headline metrics are shared by all embeddings:

``trust`` and ``continuity``
   Two-sided local-neighbourhood fidelity; higher is better.

``overlap``
   Exact fraction of original 15-nearest neighbours retained; higher is better.

``rho``
   Spearman correlation of 50,000 paired ambient-Euclidean/embedded distances; higher is
   better. This is not a manifold-geodesic metric.

``stress``
   Scale-aligned normalized ambient-Euclidean distance stress; lower is better.

``geo rho`` and ``geo stress``
   Synthetic-only intrinsic-distance fidelity: Swiss-roll arclength, spherical
   great-circle distance, and periodic flat-torus distance. These distinguish correct
   manifold unfolding from preservation of ambient chords.

``decode``
   Normalized MSE of the same train-fold-whitened kNN probe ``embedding -> X``; lower is
   better. This is a shared transductive local-decodability diagnostic, not held-out
   generalization or a general reconstruction score. Native AE/PCA/UMAP inverse errors
   remain method-specific and are not ranked.

``kNN``
   Cross-validated digit-label accuracy in the completed transductive embedding; higher
   is better. Labels are used only after embedding.

Target dimension 2
------------------

The values below are paired-seed means; time is the median end-to-end fit/transform time
in seconds. ``--`` means the dataset is unlabeled.

.. list-table::
   :header-rows: 1
   :widths: 12 13 8 8 9 8 8 8 8

   * - Dataset
     - Method
     - Trust
     - Overlap
     - Rho
     - Stress
     - Decode
     - kNN
     - Time
   * - Linear
     - PCA
     - 1.000
     - 0.988
     - 1.000
     - 0.002
     - 0.011
     - --
     - 0.001
   * - Linear
     - Autoencoder
     - 1.000
     - 0.940
     - 0.992
     - 0.064
     - 0.011
     - --
     - 3.894
   * - Linear
     - DiRe
     - 0.993
     - 0.761
     - 0.851
     - 0.265
     - 0.026
     - --
     - 0.220
   * - Linear
     - UMAP
     - 0.996
     - 0.749
     - 0.859
     - 0.248
     - 0.017
     - --
     - 0.075
   * - Swiss roll
     - PCA
     - 0.937
     - 0.497
     - 0.827
     - 0.238
     - 0.114
     - --
     - 0.001
   * - Swiss roll
     - Autoencoder
     - 0.980
     - 0.631
     - 0.537
     - 0.493
     - 0.055
     - --
     - 3.874
   * - Swiss roll
     - DiRe
     - 0.984
     - 0.716
     - 0.716
     - 0.303
     - 0.045
     - --
     - 0.218
   * - Swiss roll
     - UMAP
     - 0.998
     - 0.776
     - 0.312
     - 0.487
     - 0.013
     - --
     - 0.071
   * - Sphere
     - PCA
     - 0.848
     - 0.480
     - 0.774
     - 0.260
     - 0.360
     - --
     - 0.001
   * - Sphere
     - Autoencoder
     - 0.998
     - 0.836
     - 0.691
     - 0.377
     - 0.009
     - --
     - 3.863
   * - Sphere
     - DiRe
     - 0.931
     - 0.650
     - 0.724
     - 0.288
     - 0.142
     - --
     - 0.222
   * - Sphere
     - UMAP
     - 0.997
     - 0.753
     - 0.684
     - 0.325
     - 0.015
     - --
     - 0.092
   * - Torus
     - PCA
     - 0.830
     - 0.409
     - 0.629
     - 0.323
     - 0.391
     - --
     - 0.001
   * - Torus
     - Autoencoder
     - 0.984
     - 0.617
     - 0.382
     - 0.545
     - 0.051
     - --
     - 3.815
   * - Torus
     - DiRe
     - 0.928
     - 0.612
     - 0.597
     - 0.332
     - 0.169
     - --
     - 0.216
   * - Torus
     - UMAP
     - 0.997
     - 0.740
     - 0.552
     - 0.361
     - 0.018
     - --
     - 0.086
   * - Digits
     - PCA
     - 0.828
     - 0.210
     - 0.587
     - 0.367
     - 0.629
     - 0.632
     - 0.001
   * - Digits
     - Autoencoder
     - 0.968
     - 0.430
     - 0.484
     - 0.442
     - 0.365
     - 0.897
     - 3.653
   * - Digits
     - DiRe
     - 0.967
     - 0.536
     - 0.393
     - 0.398
     - 0.354
     - 0.949
     - 0.229
   * - Digits
     - UMAP
     - 0.984
     - 0.590
     - 0.411
     - 0.448
     - 0.297
     - 0.967
     - 0.092

Continuity tells a complementary story to trustworthiness/overlap. UMAP leads continuity
on Swiss roll and digits; the autoencoder leads on the sphere; and PCA leads on the torus,
with DiRe strongest there among the nonlinear methods:

.. list-table:: Continuity at target dimension 2
   :header-rows: 1
   :widths: 18 12 16 12 12

   * - Dataset
     - PCA
     - Autoencoder
     - DiRe
     - UMAP
   * - Linear
     - 1.000
     - 1.000
     - 0.996
     - 0.990
   * - Swiss roll
     - 0.989
     - 0.984
     - 0.992
     - 0.993
   * - Sphere
     - 0.992
     - 0.996
     - 0.989
     - 0.979
   * - Torus
     - 0.985
     - 0.967
     - 0.982
     - 0.968
   * - Digits
     - 0.937
     - 0.958
     - 0.972
     - 0.975

For the synthetic manifolds, intrinsic/geodesic distance metrics correct the misleading
ambient-chord view of a successful unfolding. The Swiss-roll result is the sharpest
example: UMAP has only ``ambient rho=0.312`` but ``geo rho=0.948``.

.. list-table:: Intrinsic/geodesic distance fidelity at target dimension 2
   :header-rows: 1
   :widths: 18 16 12 14

   * - Dataset
     - Method
     - Geo rho
     - Geo stress
   * - Linear
     - PCA
     - 1.000
     - 0.007
   * - Linear
     - Autoencoder
     - 0.992
     - 0.065
   * - Linear
     - DiRe
     - 0.851
     - 0.265
   * - Linear
     - UMAP
     - 0.859
     - 0.248
   * - Swiss roll
     - PCA
     - 0.384
     - 0.570
   * - Swiss roll
     - Autoencoder
     - 0.196
     - 0.670
   * - Swiss roll
     - DiRe
     - 0.539
     - 0.525
   * - Swiss roll
     - UMAP
     - 0.948
     - 0.221
   * - Sphere
     - PCA
     - 0.774
     - 0.273
   * - Sphere
     - Autoencoder
     - 0.691
     - 0.394
   * - Sphere
     - DiRe
     - 0.724
     - 0.303
   * - Sphere
     - UMAP
     - 0.683
     - 0.343
   * - Torus
     - PCA
     - 0.616
     - 0.333
   * - Torus
     - Autoencoder
     - 0.380
     - 0.555
   * - Torus
     - DiRe
     - 0.588
     - 0.344
   * - Torus
     - UMAP
     - 0.543
     - 0.376

Target dimension 8
------------------

The higher-width check separates general representation quality from the special case of
2-D visualization.

.. list-table::
   :header-rows: 1
   :widths: 12 13 8 8 9 8 8 8 8

   * - Dataset
     - Method
     - Trust
     - Overlap
     - Rho
     - Stress
     - Decode
     - kNN
     - Time
   * - Linear
     - PCA
     - 1.000
     - 0.999
     - 1.000
     - 0.000
     - 0.163
     - --
     - 0.001
   * - Linear
     - Autoencoder
     - 0.998
     - 0.800
     - 0.960
     - 0.071
     - 0.168
     - --
     - 3.949
   * - Linear
     - DiRe
     - 0.948
     - 0.465
     - 0.484
     - 0.244
     - 0.251
     - --
     - 0.220
   * - Linear
     - UMAP
     - 0.954
     - 0.456
     - 0.558
     - 0.221
     - 0.271
     - --
     - 0.132
   * - Digits
     - PCA
     - 0.991
     - 0.664
     - 0.922
     - 0.107
     - 0.268
     - 0.936
     - 0.001
   * - Digits
     - Autoencoder
     - 0.996
     - 0.733
     - 0.836
     - 0.165
     - 0.257
     - 0.952
     - 3.885
   * - Digits
     - DiRe
     - 0.978
     - 0.607
     - 0.296
     - 0.297
     - 0.322
     - 0.954
     - 0.229
   * - Digits
     - UMAP
     - 0.988
     - 0.630
     - 0.358
     - 0.382
     - 0.286
     - 0.969
     - 0.132

.. list-table:: Continuity at target dimension 8
   :header-rows: 1
   :widths: 18 12 16 12 12

   * - Dataset
     - PCA
     - Autoencoder
     - DiRe
     - UMAP
   * - Linear
     - 1.000
     - 0.998
     - 0.962
     - 0.947
   * - Digits
     - 0.995
     - 0.996
     - 0.977
     - 0.977

Interpretation
--------------

There is no universal winner because the objectives differ.

* **Linear subspace:** PCA is the correct control and is essentially exact. The
  autoencoder is close, while graph-layout methods intentionally give up global metric
  fidelity.
* **Swiss roll:** UMAP is the clear manifold winner: it leads trustworthiness, overlap,
  and intrinsic/geodesic fidelity (``geo rho=0.948``). Its low ambient rho is the expected
  consequence of correctly separating adjacent sheets. PCA best preserves ambient
  chords; DiRe is the strongest nonlinear method on ambient rho/stress and slightly
  easier to decode locally than the autoencoder.
* **Sphere:** the autoencoder is strongest on trustworthiness, overlap, and local
  decodability; UMAP nearly matches its trustworthiness. PCA best preserves global
  great-circle/ambient distance order, with DiRe the strongest nonlinear method on those
  global metrics. A sphere cannot be represented in one flat 2-D chart without a cut.
* **Torus:** UMAP leads trustworthiness, overlap, and local decodability; DiRe has better
  continuity than UMAP. PCA best preserves global ambient and intrinsic distances, with
  DiRe again strongest among the nonlinear methods. The autoencoder's high
  trustworthiness but weak global fidelity is consistent with the topology obstruction.
* **Digits at d=2:** UMAP leads trustworthiness, neighbour overlap, local decodability,
  and label accuracy. DiRe is close on labels and has better ambient stress than UMAP and
  the autoencoder, while PCA has the best ambient stress overall.
* **Digits at d=8:** the autoencoder leads trustworthiness, overlap, and local
  decodability; PCA leads ambient distance fidelity; UMAP still gives the best label
  accuracy. DiRe and the autoencoder have almost identical label accuracy here.
* **Runtime at n=1,000:** the autoencoder takes about 3.7--3.9 seconds, DiRe about
  0.22--0.23 seconds, and UMAP about 0.07--0.13 seconds on this GH200. These are
  three-run medians; the first run is cold rather than explicitly warmed up.

Architecture sensitivity
------------------------

Repeating the ``d=2`` autoencoder rows with a smaller ``D-64-32-d-32-64-D`` network did
not change the qualitative conclusions. It was slightly weaker on digits
(``trust=0.960``, ``overlap=0.404``, ``kNN=0.885`` versus 0.968, 0.430, and 0.897 for the
primary network), while the sphere, Swiss-roll, and torus patterns remained similar.

Most primary autoencoder fits selected epochs near the 200-epoch cap. A 400-epoch spot
check left the digit result unchanged (early stopping still selected epoch 182). On Swiss
roll it selected epoch 359 and improved native reconstruction, but did not improve the
embedding-quality conclusion: trustworthiness rose only marginally from 0.975 to 0.979
for that seed while continuity and ambient stress worsened. Reconstruction-loss tuning is
therefore not a substitute for optimizing manifold fidelity.

Limitations
-----------

* These are 1,000-row quality runs, not scaling benchmarks.
* The comparison is transductive; DiRe needs a native ``transform`` before a fully shared
  held-out generalization table is possible.
* The common decoder is a whitened local-interpolation probe on a split made *after* all
  reducers saw the complete matrix. It is not an out-of-sample reconstruction benchmark.
* The autoencoder result depends on architecture and optimization. The sensitivity run
  and 400-epoch spot check reduce but do not remove that dependence. Autoencoder runtime
  includes validation-based epoch selection and a second full-data refit, whereas DiRe
  and UMAP each run one ``fit_transform``.
* cuML UMAP's native inverse is hybrid CPU/GPU and is not part of fit timing. It is skipped
  at ``d>=8`` because cuML warns that quality may be poor or the call may fail, while its
  CPU Delaunay stage also becomes prohibitively expensive.
* The first CUDA call includes initialization overhead. Reported time is the paired-seed
  median, which suppresses that cold-start outlier.

Reproduce
---------

The exact raw artifacts are tracked with the report:

* :download:`d=2 raw JSON <results/fixed-dimension-gh200-d2/results.json>` and
  :download:`summary CSV <results/fixed-dimension-gh200-d2/summary.csv>`
* :download:`d=8 raw JSON <results/fixed-dimension-gh200-d8/results.json>` and
  :download:`summary CSV <results/fixed-dimension-gh200-d8/summary.csv>`
* :download:`small-autoencoder sensitivity JSON
  <results/fixed-dimension-gh200-ae-small/results.json>`
* :download:`400-epoch spot-check JSON
  <results/fixed-dimension-gh200-ae-400/results.json>`

Prepare the exact DiRe source without modifying the sibling checkout:

.. code-block:: bash

   git -C ../dire-rapids archive --format=tar \
     --output=/tmp/dire-rapids-f7d5736.tar \
     f7d57367af8e10f6013e64026e2dbb0af6781947
   mkdir -p /tmp/dire-rapids-f7d5736
   tar -xf /tmp/dire-rapids-f7d5736.tar -C /tmp/dire-rapids-f7d5736

Then run the harness, which writes raw JSON, per-run CSV, and paired-seed summary CSV:

.. code-block:: bash

   conda run -n rapids-26.06 python experiments/compare_fixed_dimension.py \
     --datasets linear,swiss-roll,sphere,torus,digits \
     --methods pca,autoencoder,dire,umap --target-dimension 2 \
     --seeds 0,1,2 --n-samples 1000 --ambient-dimension 32 \
     --device cuda --dire-knn-backend pytorch \
     --dire-revision f7d57367af8e10f6013e64026e2dbb0af6781947 \
     --dire-repo /tmp/dire-rapids-f7d5736 --require-all \
     --output-dir docs/results/fixed-dimension-gh200-d2

For ``d=8``, change ``--target-dimension`` and select ``--datasets linear,digits``.
