Large-corpus autoencoder, DiRe, and UMAP comparison
===================================================

The small fixed-dimension study is useful for checking metric definitions, but it does
not exercise the regime for which GPU reducers are built.  This page repeats the
comparison on the two real embedding corpora already used by the DiRe projects:

* **arXiv:** 723,457 mean-pooled BGE-small paper embeddings with 384 input
  dimensions, reduced independently to **2 and 8 dimensions**.
* **Wikipedia:** the first 1,000,000 English Cohere Wikipedia paragraph embeddings
  with 1,024 input dimensions, also reduced independently to **2 and 8 dimensions**.

The 2-D arXiv and 8-D Wikipedia cells match the earlier DiRe experiments; the crossed
8-D arXiv and 2-D Wikipedia cells keep the corpus fixed while changing target width.

The short answer is that there is still no universal winner.  UMAP separates arXiv
categories and Wikipedia articles best at both widths.  The autoencoder leads most
arXiv geometry metrics at 8-D and most Wikipedia local-geometry metrics at 8-D.  DiRe
has the lowest Wikipedia stress at both widths and the lowest arXiv beta1 DTW at both
widths.  The topology result is not monotone in target width: extra coordinates greatly
improve some diagnostics while worsening others.

These are fixed-target, transductive comparisons.  The autoencoder is trained from
scratch for every corpus, target width, and seed.  Its encoder is therefore not treated
as a universal pretrained reducer or recycled between experiments.

Protocol
--------

All methods receive the same row-L2-normalized ``float32`` matrix.  Euclidean distance
on these rows is equivalent to a monotone transform of cosine distance.  Each method is
run with reducer/training seeds 0, 1, and 2 and ``n_neighbors=16`` where applicable.

The autoencoder is a symmetric SiLU MLP with a linear bottleneck and output:

* arXiv: ``384-256-128-32-d-32-128-256-384``
* Wikipedia: ``1024-512-256-64-d-64-256-512-1024``

Here ``d`` is 2 or 8.  Every corpus, width, and seed receives a fresh validation run
and full-data refit; weights are never transferred between cells.

AdamW minimizes MSE.  Five percent of the rows select the epoch count, after which a
freshly initialized model is refit on the complete matrix for that many epochs.  The
reported autoencoder time includes input staging, validation-based epoch selection,
the fresh full-data refit, encoding, and batched native reconstruction.  DiRe and UMAP
times cover ``fit_transform``.  The timing columns are therefore systems context, not a
claim that all three methods perform identical work.

The primary comparison tables use FP32 model computation.  A separate BF16-autocast
sensitivity below keeps the primary scientific results unchanged.

DiRe 0.3.1 is pinned to commit
``f7d57367af8e10f6013e64026e2dbb0af6781947`` and uses the strict cuVS k-NN backend.
The resolved index is IVF-PQ for arXiv and IVF-Flat for Wikipedia.  UMAP is cuML 26.06
with scalable automatic graph construction, spectral initialization, ``min_dist=0.1``,
and ``spread=1``.  Native UMAP inverse transformation is disabled: its CPU Delaunay
stage is not a meaningful operation at 723K--1M rows.

Shared metrics
~~~~~~~~~~~~~~

Quadratic metrics are explicitly sample-conditional rather than described as
full-corpus trustworthiness:

* trustworthiness, continuity, and exact neighbor overlap use a fixed stratified
  5,000-row arXiv sample or a fixed uniform 3,000-row Wikipedia sample (there are more
  Wikipedia article URLs than sample slots, so stratification is not possible);
* local decodability is a whitened k-NN regression diagnostic on 10,000 arXiv or 5,000
  Wikipedia rows; it is not native reconstruction or an out-of-sample decoder;
* distance Spearman correlation and scale-aligned stress use the same 200,000 random
  full-corpus pairs for every method and seed;
* arXiv category accuracy uses a fixed split after excluding categories represented by
  fewer than five sampled papers;
* Wikipedia article retrieval queries 3,000 fixed anchor paragraphs against the **full
  one-million-row low-dimensional layout**.  Empty and full-corpus-singleton URLs are
  excluded, leaving 2,895 eligible anchors.

For topology, every method and both target widths are evaluated on identical point
indices within a corpus.  Each sampled cloud is rescaled to unit diameter before
computing 50-step Betti curves and fast-DTW errors.  The arXiv result uses three
stratified 4,000-paper samples, matching the paper protocol.  Wikipedia uses three
uniform 3,000-paragraph samples.  Topology summaries below pool three reducer seeds by
three evaluation samples; the nine values are descriptive and are not nine independent
experimental replicates.

Betti-curve DTW is a sampled descriptive proxy, not a proof of homology preservation.
The implementation computes a finite, discretized curve for each cloud over that
cloud's own longest finite-death range, then compares beta-value sequences with
fast-DTW.  Thus the sample indices and 50-step resolution are held fixed across target
widths, and the reference axes are identical, but each reduced cloud has its own
numeric filtration axis.  Unit-diameter scaling removes coordinate scale, but the
result remains sensitive to the point sample, curve discretization, and DTW alignment.
The H0 island-ness columns are likewise diagnostics with no single fidelity direction.

arXiv: 723,457 x 384
--------------------

2-D result
~~~~~~~~~~

Means over the three reducer/training seeds are shown below.  Higher is better for
trustworthiness, continuity, overlap, correlation, and category accuracy; lower is
better for stress and decoder NMSE.

.. list-table:: arXiv 2-D geometry and context metrics
   :header-rows: 1
   :widths: 13 10 9 9 9 9 9 9 10

   * - Method
     - Fit (s)
     - Trust
     - Continuity
     - Overlap
     - Pair rho
     - Stress
     - Decode NMSE
     - Category acc.
   * - Autoencoder
     - 24.32
     - **0.877**
     - **0.918**
     - 0.099
     - **0.588**
     - 0.523
     - 0.647
     - 0.363
   * - DiRe
     - **15.33**
     - 0.792
     - 0.860
     - 0.116
     - 0.313
     - 0.426
     - 0.721
     - 0.377
   * - UMAP
     - 30.03
     - 0.858
     - 0.902
     - **0.147**
     - 0.489
     - **0.418**
     - **0.643**
     - **0.482**

Reducer-seed variability is modest for most metrics.  The largest visible variations
are the autoencoder's stress (sample standard deviation 0.077) and UMAP's pair
correlation and category accuracy (0.054 and 0.027).  The autoencoder selected epochs
199, 200, and 199;
its native reconstruction NMSE was 0.597--0.601.

The sampled Betti-curve diagnostics give a different and much clearer ranking:

.. list-table:: arXiv 2-D sampled topology diagnostics, mean over 3 reducer x 3 evaluation seeds
   :header-rows: 1
   :widths: 16 14 14 14 16 12

   * - Method
     - DTW beta0
     - DTW beta1
     - Longest H0
     - Top-5 / median H0
     - H0 Gini
   * - Autoencoder
     - 4,841
     - 439
     - 0.135
     - 31.4
     - 0.493
   * - DiRe
     - **3,322**
     - **176**
     - 0.091
     - 15.6
     - 0.382
   * - UMAP
     - 10,414
     - 941
     - 0.267
     - 72.4
     - 0.420
   * - 384-D reference
     - --
     - --
     - 0.682
     - 2.33
     - 0.115

DiRe has the lowest two Betti-curve DTW errors, and the autoencoder is consistently between
DiRe and UMAP on those direct measures.  DiRe also has the lowest three island-ness
diagnostics.  Those H0 columns are descriptive rather than fidelity rankings: a lower
top-5/median or Gini means less concentration in a few bars, while the reference longest
bar is much larger than every reduced layout's.  The autoencoder's sampled DTW errors
are higher than DiRe's but much lower than UMAP's.

8-D cross-over
~~~~~~~~~~~~~~

Increasing the arXiv target width changes the geometry ranking much more than the
category ranking.  The autoencoder now leads trustworthiness, continuity, overlap,
pair correlation, stress, and local decodability; UMAP remains best on both ordinary
and balanced category accuracy.

.. list-table:: arXiv 8-D geometry and context metrics
   :header-rows: 1
   :widths: 13 10 9 9 9 9 9 9 10

   * - Method
     - Fit (s)
     - Trust
     - Continuity
     - Overlap
     - Pair rho
     - Stress
     - Decode NMSE
     - Category acc.
   * - Autoencoder
     - 23.42
     - **0.969**
     - **0.988**
     - **0.321**
     - **0.801**
     - **0.254**
     - **0.466**
     - 0.509
   * - DiRe
     - **15.08**
     - 0.905
     - 0.897
     - 0.187
     - 0.355
     - 0.335
     - 0.534
     - 0.531
   * - UMAP
     - 30.66
     - 0.937
     - 0.954
     - 0.247
     - 0.590
     - 0.357
     - 0.499
     - **0.609**

Balanced category accuracy follows the same ordering: 0.250 for the autoencoder,
0.254 for DiRe, and 0.329 for UMAP.  The autoencoder selected 200, 198, and 200 epochs;
its native reconstruction NMSE was 0.424--0.425.  All displayed 8-D geometry means are
over the same three reducer seeds and the same evaluation samples as the 2-D result.

.. list-table:: arXiv 8-D sampled topology diagnostics, mean over 3 reducer x 3 evaluation seeds
   :header-rows: 1
   :widths: 16 14 14 14 16 12

   * - Method
     - DTW beta0
     - DTW beta1
     - Longest H0
     - Top-5 / median H0
     - H0 Gini
   * - Autoencoder
     - **1,583**
     - 3,045
     - 0.315
     - 3.67
     - 0.154
   * - DiRe
     - 2,685
     - **177**
     - 0.416
     - 17.9
     - 0.396
   * - UMAP
     - 6,469
     - 499
     - 0.470
     - 42.9
     - 0.319
   * - 384-D reference
     - --
     - --
     - 0.682
     - 2.33
     - 0.115

The cross-over resolves the earlier corpus/width confounding, but it does not yield a
single topology ranking.  At 8-D the autoencoder's beta0 DTW falls from 4,841 to 1,583,
while its beta1 DTW rises from 439 to 3,045.  DiRe's beta1 DTW is essentially unchanged
(176 at 2-D and 177 at 8-D), and remains the best.  UMAP improves on both DTW measures
with the wider target but remains behind the best method for each.

Wikipedia: 1,000,000 x 1,024
----------------------------

8-D result
~~~~~~~~~~

.. list-table:: Wikipedia 8-D geometry and article-retrieval metrics
   :header-rows: 1
   :widths: 13 9 8 8 8 8 8 9 9 9

   * - Method
     - Fit (s)
     - Trust
     - Continuity
     - Overlap
     - Pair rho
     - Stress
     - Decode NMSE
     - Article hit
     - Article recall
   * - Autoencoder
     - 81.85
     - **0.960**
     - **0.944**
     - **0.337**
     - **0.459**
     - 0.362
     - **0.791**
     - 0.283
     - 0.079
   * - DiRe
     - **63.93**
     - 0.908
     - 0.894
     - 0.229
     - 0.378
     - **0.268**
     - 0.844
     - 0.570
     - 0.345
   * - UMAP
     - 144.05
     - 0.917
     - 0.885
     - 0.235
     - 0.389
     - 0.438
     - 0.841
     - **0.754**
     - **0.530**

Article-neighbor purity, omitted from the table for width, is 0.071 for the autoencoder,
0.306 for DiRe, and 0.410 for UMAP.  The apparent tension between the autoencoder's high
sample-conditional overlap and low article retrieval is real but the scopes differ: the
former finds neighbors inside a 3,000-row evaluation cloud, while the latter queries the
complete one-million-row layout for a semantic grouping not used by any reducer.

The winning method for every Wikipedia metric is unchanged across all three reducer
seeds, and variability is generally small.  The notable exception is autoencoder stress
(sample standard deviation 0.018); article-hit sample standard deviations range from
0.001 for DiRe to 0.006 for the autoencoder.  The autoencoder selected epoch 200 on every
run and produced native reconstruction NMSE 0.690--0.691.

.. list-table:: Wikipedia 8-D sampled topology diagnostics, mean over 3 reducer x 3 evaluation seeds
   :header-rows: 1
   :widths: 16 14 14 14 16 12

   * - Method
     - DTW beta0
     - DTW beta1
     - Longest H0
     - Top-5 / median H0
     - H0 Gini
   * - Autoencoder
     - **1,591**
     - **7,572**
     - 0.191
     - 2.79
     - 0.145
   * - DiRe
     - 3,335
     - 9,117
     - 0.338
     - 6.37
     - 0.435
   * - UMAP
     - 2,986
     - 9,198
     - 0.391
     - 23.07
     - 0.442
   * - 1,024-D reference
     - --
     - --
     - 0.785
     - 1.11
     - 0.040

At target width 8, the autoencoder has the lowest two sampled Betti-curve DTW errors and
the lowest three island-ness diagnostics.  It is therefore the strongest result on this
sampled Betti diagnostic as well as the local-geometry winner.  The H0 columns still
should not be read as simple distance-to-reference scores.

2-D cross-over
~~~~~~~~~~~~~~

At width 2, the autoencoder retains the best trustworthiness and a narrowly best
neighbor overlap, while UMAP has the best continuity and pair correlation.  DiRe has
the lowest stress by a wide margin.  UMAP remains the clear article-grouping winner.

.. list-table:: Wikipedia 2-D geometry and article-retrieval metrics
   :header-rows: 1
   :widths: 13 9 8 8 8 8 8 9 9 9

   * - Method
     - Fit (s)
     - Trust
     - Continuity
     - Overlap
     - Pair rho
     - Stress
     - Decode NMSE
     - Article hit
     - Article recall
   * - Autoencoder
     - 81.60
     - **0.887**
     - 0.827
     - **0.183**
     - 0.230
     - 0.747
     - 0.9070
     - 0.050
     - 0.006
   * - DiRe
     - **62.14**
     - 0.752
     - 0.823
     - 0.118
     - 0.305
     - **0.428**
     - 0.959
     - 0.288
     - 0.141
   * - UMAP
     - 144.01
     - 0.847
     - **0.832**
     - 0.182
     - **0.337**
     - 0.475
     - **0.9070**
     - **0.498**
     - **0.236**

The two displayed decoder NMSE values round alike: UMAP is lower by only 0.000034
(0.906980 versus 0.907014), which is negligible relative to their across-seed standard
deviations.  Article-neighbor purity is 0.006 for the autoencoder, 0.136 for DiRe, and
0.201 for UMAP.  The autoencoder selected 200, 200, and 199 epochs and had native
reconstruction NMSE 0.816--0.824.  Reducing Wikipedia from 8-D to 2-D particularly
damages autoencoder article retrieval and global stress; it does not overturn UMAP's
grouping lead or DiRe's stress lead.

.. list-table:: Wikipedia 2-D sampled topology diagnostics, mean over 3 reducer x 3 evaluation seeds
   :header-rows: 1
   :widths: 16 14 14 14 16 12

   * - Method
     - DTW beta0
     - DTW beta1
     - Longest H0
     - Top-5 / median H0
     - H0 Gini
   * - Autoencoder
     - 4,822
     - **8,974**
     - 0.087
     - 74.8
     - 0.702
   * - DiRe
     - **2,107**
     - 9,355
     - 0.057
     - 6.23
     - 0.361
   * - UMAP
     - 3,566
     - 9,207
     - 0.128
     - 23.7
     - 0.485
   * - 1,024-D reference
     - --
     - --
     - 0.785
     - 1.11
     - 0.040

The 2-D Wikipedia topology ranking is split: DiRe has the lowest beta0 DTW, while the
autoencoder has the lowest beta1 DTW.  At 8-D the autoencoder improves sharply on both
and leads both.  DiRe's beta0 moves in the opposite direction (2,107 at 2-D versus
3,335 at 8-D), while its beta1 improves slightly.  UMAP's beta1 is nearly unchanged.
Together with the arXiv cross-over, this shows that target width matters but does not
alone determine the topology ranking; the corpus and the homology degree matter too.

Training-budget sensitivity
---------------------------

Because the autoencoders in the original arXiv 2-D and Wikipedia 8-D cells selected the
200-epoch boundary, seed 0 in those two cells was repeated with a 400-epoch cap and
patience 30.  No 400-epoch run was made for either crossed cell.  These are sensitivity
runs, not geometry-tuned replacements for the three-seed headline protocol.

.. list-table:: Effect of longer reconstruction training, seed 0
   :header-rows: 1
   :widths: 17 9 10 9 9 9 9 11 11

   * - Corpus
     - Epoch
     - Recon. NMSE
     - Trust
     - Continuity
     - Overlap
     - Stress
     - DTW beta0
     - DTW beta1
   * - arXiv 2-D primary
     - 199
     - 0.597
     - 0.882
     - 0.914
     - 0.101
     - 0.562
     - 4,544
     - 516
   * - arXiv 2-D longer
     - 398
     - **0.575**
     - **0.894**
     - 0.909
     - **0.114**
     - 0.695
     - 5,771
     - 772
   * - Wikipedia 8-D primary
     - 200
     - 0.691
     - 0.960
     - **0.943**
     - 0.334
     - **0.382**
     - 1,599
     - **8,146**
   * - Wikipedia 8-D longer
     - 400
     - **0.674**
     - **0.963**
     - 0.940
     - **0.346**
     - 0.397
     - **1,545**
     - 9,709

Longer MSE training improves reconstruction, trustworthiness, overlap, and local
decodability on both corpora.  It worsens continuity and global distance fidelity, and
its sampled Betti-curve effect is mixed or negative.  On arXiv both DTW errors get worse;
on Wikipedia beta0 improves slightly while beta1 gets substantially worse.
Reconstruction convergence is therefore not a universal stopping rule for geometric
fidelity.

BF16 mixed-precision sensitivity
--------------------------------

Every autoencoder cell was repeated with standard CUDA BF16 autocast: both corpora,
target dimensions 2 and 8, and seeds 0--2.  Model computation during training,
epoch-selection validation, and final encoding used BF16 autocast, while parameters,
gradients, AdamW state, the explicit MSE calculation, and reconstruction accumulations
remained FP32.  The inputs, samples, seeds, architectures, and optimizer settings match
the primary FP32 cells.  Speedup below is FP32 time divided by BF16 time, so a value
below one is a slowdown.

.. list-table:: Paired FP32 and standard-BF16 autoencoder sensitivity
   :header-rows: 1
   :widths: 13 4 11 11 8 11 11 8 12

   * - Corpus
     - ``d``
     - Train FP32 (s)
     - Train BF16 (s)
     - Train speedup
     - Total FP32 (s)
     - Total BF16 (s)
     - Total speedup
     - NMSE ratio
   * - arXiv
     - 2
     - 23.49
     - 25.49
     - 0.920x
     - 24.32
     - 26.28
     - 0.925x
     - 1.00193
   * - arXiv
     - 8
     - 22.64
     - 25.46
     - 0.889x
     - 23.42
     - 26.25
     - 0.892x
     - 1.00131
   * - Wikipedia
     - 2
     - 78.74
     - 39.43
     - 1.997x
     - 81.60
     - 42.28
     - 1.931x
     - 1.00096
   * - Wikipedia
     - 8
     - 78.96
     - 41.53
     - 1.901x
     - 81.85
     - 44.34
     - 1.846x
     - 1.00023

Training times are arithmetic means, training speedups are paired geometric means, total
speedups are paired arithmetic means, and NMSE ratios are arithmetic means of the three
paired ratios.  Both the standard and FP32-evaluation-isolation BF16 arms pass all
prespecified practical checks on native reconstruction and shared geometry or label
metrics in all four cells.  These three-seed margins are descriptive rather than formal
equivalence tests.  The largest paired BF16/FP32 native-NMSE ratio in the standard arm
is only 1.00243.  The full-fit FP32 runs and later BF16 runs were not alternated
contemporaneously, so their timing ratios should also be read as descriptive.

Sampled topology is more variable.  The pooled FP32-to-BF16 ``(beta0, beta1)`` DTW
means are ``(4841, 439) -> (5660, 537)`` for arXiv 2-D, ``(1583, 3045) ->
(1659, 3068)`` for arXiv 8-D, ``(4822, 8974) -> (4820, 8986)`` for Wikipedia 2-D,
and ``(1591, 7572) -> (1505, 8121)`` for Wikipedia 8-D.  A clean isolation rerun used
BF16 only for training and FP32 for validation, encoding, reconstruction, shared-metric
evaluation, and topology input generation.  Shared-metric quality remains essentially
unchanged, while the pooled topology mean moves closer to FP32 in only four of eight
corpus/width/Betti comparisons.  This descriptive isolation does not support final-
coordinate quantization as the main explanation for those topology differences.

The timing result is also model-size dependent.  With TF32 matrix multiplication
disabled, a fixed-order 100-step microprofile on a synthetic standard-normal resident
batch gives total-step FP32/BF16 speedups of 0.859 and 0.842 for the arXiv 2-D and 8-D
models, versus 2.407 and 2.428 for Wikipedia.  Each precision had its own warm-up.
Wikipedia's forward-plus-backward work alone reaches about 2.65x.  The roughly
272K-parameter arXiv network is too small to amortize autocast overhead at this batch
shape; the roughly 1.35M-parameter Wikipedia network is compute-heavy enough to benefit.
FP32 AdamW and non-training fit work then limit the Wikipedia end-to-end gain to about
1.85--1.93x, rather than the hypothesized 3--4x.

Interpretation
--------------

The useful conclusion is objective-specific:

* **DiRe** has the lowest Wikipedia stress at both widths, the lowest arXiv beta1 DTW
  at both widths, and the lowest Wikipedia 2-D beta0 DTW.
* **UMAP** is strongest when visibly or semantically coherent local groups are the
  goal: it leads arXiv category prediction and Wikipedia same-article retrieval at
  both widths.
* **The autoencoder** leads most arXiv geometry metrics at 8-D and most Wikipedia
  local-geometry metrics at 8-D.  It leads both Wikipedia 8-D topology DTW measures,
  and offers a reusable encoder *within each fitted corpus*.

The crossed design sharpens rather than simplifies the conclusion.  A wider target
usually helps autoencoder reconstruction and sampled geometry, but topology is not
monotone: arXiv autoencoder beta0 improves while beta1 becomes much worse, and DiRe's
Wikipedia beta0 is better at 2-D than 8-D.  Method agreement therefore depends on what
is being preserved, not just on the number of output coordinates.

The autoencoder's training cost is expected and is not evidence against it.  On this
GH200 it is in fact comparable to the graph reducers at these settings, but that timing
depends on architecture, batch size, epoch rule, and hardware.  The more important cost
is methodological: it must be fitted and validated again when the corpus, scale, target
width, or reconstruction model changes.

Artifacts and provenance
------------------------

The tracked compact artifacts contain every reducer row and every topology metric; the
multi-gigabyte prepared matrices and layouts remain under ignored ``artifacts/``.

* arXiv 2-D: :download:`reducer JSON <results/large-arxiv-d2/results.json>`,
  :download:`reducer CSV <results/large-arxiv-d2/results.csv>`,
  :download:`topology rows <results/large-arxiv-d2/topology-raw.csv>`,
  :download:`topology aggregate <results/large-arxiv-d2/topology-aggregate.csv>`, and
  :download:`full curves/provenance <results/large-arxiv-d2/topology-raw.json>`
* arXiv 8-D: :download:`reducer JSON <results/large-arxiv-d8/results.json>`,
  :download:`reducer CSV <results/large-arxiv-d8/results.csv>`,
  :download:`topology rows <results/large-arxiv-d8/topology-raw.csv>`,
  :download:`topology aggregate <results/large-arxiv-d8/topology-aggregate.csv>`, and
  :download:`full curves/provenance <results/large-arxiv-d8/topology-raw.json>`
* :download:`arXiv input manifest <results/large-arxiv-d2/input-manifest.json>` and the
  :download:`crossed-cell copy <results/large-arxiv-d8/input-manifest.json>`
* :download:`arXiv 2-D 400-epoch sensitivity <results/large-arxiv-d2/ae400-results.json>`
  plus sensitivity topology :download:`rows <results/large-arxiv-d2/ae400-topology-raw.csv>`
  and :download:`full provenance <results/large-arxiv-d2/ae400-topology-raw.json>`
* Wikipedia 2-D: :download:`reducer JSON <results/large-wikipedia-d2/results.json>`,
  :download:`reducer CSV <results/large-wikipedia-d2/results.csv>`,
  :download:`topology rows <results/large-wikipedia-d2/topology-raw.csv>`,
  :download:`topology aggregate <results/large-wikipedia-d2/topology-aggregate.csv>`, and
  :download:`full curves/provenance <results/large-wikipedia-d2/topology-raw.json>`
* Wikipedia 8-D: :download:`reducer JSON <results/large-wikipedia-d8/results.json>`,
  :download:`reducer CSV <results/large-wikipedia-d8/results.csv>`,
  :download:`topology rows <results/large-wikipedia-d8/topology-raw.csv>`,
  :download:`topology aggregate <results/large-wikipedia-d8/topology-aggregate.csv>`, and
  :download:`full curves/provenance <results/large-wikipedia-d8/topology-raw.json>`
* :download:`Wikipedia 8-D 400-epoch sensitivity <results/large-wikipedia-d8/ae400-results.json>`
  plus sensitivity topology
  :download:`rows <results/large-wikipedia-d8/ae400-topology-raw.csv>` and
  :download:`full provenance <results/large-wikipedia-d8/ae400-topology-raw.json>`
* :download:`Wikipedia input manifest <results/large-wikipedia-d8/input-manifest.json>`
  and the :download:`crossed-cell copy <results/large-wikipedia-d2/input-manifest.json>`
* BF16 analysis: :download:`paired summary <results/large-ae-precision/summary.json>`,
  :download:`paired rows <results/large-ae-precision/paired-results.csv>`, and
  :download:`training-step microprofile <results/large-ae-precision/microprofile.json>`
* Standard BF16 reducer results:
  :download:`arXiv JSON <results/large-ae-precision/arxiv-bf16-results.json>`,
  :download:`arXiv CSV <results/large-ae-precision/arxiv-bf16-results.csv>`,
  :download:`Wikipedia JSON <results/large-ae-precision/wikipedia-bf16-results.json>`,
  and :download:`Wikipedia CSV <results/large-ae-precision/wikipedia-bf16-results.csv>`
* BF16-training/FP32-evaluation isolation results:
  :download:`arXiv JSON <results/large-ae-precision/arxiv-bf16-train-fp32-eval-results.json>`,
  :download:`arXiv CSV <results/large-ae-precision/arxiv-bf16-train-fp32-eval-results.csv>`,
  :download:`Wikipedia JSON <results/large-ae-precision/wikipedia-bf16-train-fp32-eval-results.json>`,
  and :download:`Wikipedia CSV <results/large-ae-precision/wikipedia-bf16-train-fp32-eval-results.csv>`
* Standard-BF16 topology aggregates:
  :download:`arXiv d=2 <results/large-ae-precision/arxiv-d2-bf16-topology-aggregate.csv>`,
  :download:`arXiv d=8 <results/large-ae-precision/arxiv-d8-bf16-topology-aggregate.csv>`,
  :download:`Wikipedia d=2 <results/large-ae-precision/wikipedia-d2-bf16-topology-aggregate.csv>`,
  and :download:`Wikipedia d=8 <results/large-ae-precision/wikipedia-d8-bf16-topology-aggregate.csv>`
* BF16-training/FP32-evaluation topology aggregates:
  :download:`arXiv d=2 <results/large-ae-precision/arxiv-d2-bf16-train-fp32-eval-topology-aggregate.csv>`,
  :download:`arXiv d=8 <results/large-ae-precision/arxiv-d8-bf16-train-fp32-eval-topology-aggregate.csv>`,
  :download:`Wikipedia d=2 <results/large-ae-precision/wikipedia-d2-bf16-train-fp32-eval-topology-aggregate.csv>`,
  and :download:`Wikipedia d=8 <results/large-ae-precision/wikipedia-d8-bf16-train-fp32-eval-topology-aggregate.csv>`
* The analysis and profiling programs are
  :download:`analyze_ae_precision.py <../experiments/analyze_ae_precision.py>` and
  :download:`profile_ae_precision.py <../experiments/profile_ae_precision.py>`.

The Wikipedia cache is pinned to Hugging Face revision
``ade45fb52bd549f5e8c065636fe4160a43c2af36``; all ten embedding and metadata shard
hashes are recorded in its manifest.  The local arXiv input hashes are
``637e0726...cc0be80`` for ``embeddings.npy`` and ``25f62936...38e283`` for
``meta.parquet``.  The normalized prepared-matrix content hashes are
``c15c11bb...434d0`` for arXiv and ``f5d56282...dc316`` for Wikipedia.  The reducer
harness hash embedded in every final result is
``b3b75a2c...26fdc``; its exact
:download:`archived harness <results/large-benchmark-harness-b3b75a2.py>` is tracked.
The topology evaluator hash is ``d249aed7...79c8``; its exact
:download:`archived evaluator <results/large-topology-harness-d249aed.py>` is also
tracked.  For archived DiRe source, the
factory/base-module hash is ``8097f834...b495d0d34``, the CuVS-module hash is
``2a36eee6...33fb354``, and the Betti implementation hash is
``553cc327...774479ad9``.  A Git archive has no ``.git`` directory, so result rows record
``dire_revision: null`` alongside the expected commit; the exported files were
separately verified against that commit.

The main runs use Python 3.14, NumPy 2.4.6, PyTorch 2.11.0+cu130, CuPy 14.1.1, and
cuML 26.06.00 on an NVIDIA GH200 480GB (96,768 MiB recorded through the CUDA device
property; NVML reports 97,871 MiB).  Topology evaluation
uses ``ripser 0.6.14`` and ``fastdtw 0.3.4`` from the RAPIDS 26.04 environment because
those two optional packages are not installed in the 26.06 environment.

Reproduction
------------

Cache the exact Wikipedia prefix into a **fresh** directory while retaining article
identifiers.  (The published cache manifest requested ``main`` and recorded the resolved
SHA; it validates in place, but changing its requested-revision field during resume is
correctly rejected.)

.. code-block:: bash

   conda run -n rapids-26.04 env HF_HOME=$PWD/artifacts/data/huggingface-pinned \
     python experiments/cache_wikipedia_embeddings.py \
     --dataset CohereLabs/wikipedia-2023-11-embed-multilingual-v3 \
     --revision ade45fb52bd549f5e8c065636fe4160a43c2af36 \
     --config en --n-rows 1000000 --shard-size 100000 \
     --cache-dir artifacts/data/cohere-wikipedia-english-1m-pinned

Export the pinned DiRe source without touching a dirty sibling worktree:

.. code-block:: bash

   mkdir -p /tmp/dire-rapids-origin-f7d5736
   git -C ../dire-rapids archive \
     f7d57367af8e10f6013e64026e2dbb0af6781947 \
     | tar -x -C /tmp/dire-rapids-origin-f7d5736

Resolve the RAPIDS environment prefix in the outer shell.  This avoids expanding a base
environment's ``$CONDA_PREFIX`` before ``conda run``.  The commands use fresh output
directories, so they refit rather than reusing successful checkpoints.  They invoke the
exact tracked harness that generated the published rows.

.. code-block:: bash

   RAPIDS_2606="$(conda run -n rapids-26.06 python -c \
     'import sys; print(sys.prefix)')"

Run arXiv:

.. code-block:: bash

   conda run -n rapids-26.06 env \
     LD_PRELOAD="$RAPIDS_2606/lib/libnvJitLink.so.13" \
     OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
     python docs/results/large-benchmark-harness-b3b75a2.py \
     --dataset-name arxiv \
     --data ../dire-rapids-arxiv/data/embeddings.npy \
     --labels ../dire-rapids-arxiv/data/meta.parquet \
     --label-column primary_category --label-policy category \
     --sizes 723457 --target-dimensions 2 8 \
     --methods autoencoder dire umap --seeds 0 1 2 --neighbors 16 \
     --dire-repo /tmp/dire-rapids-origin-f7d5736 \
     --dire-revision f7d57367af8e10f6013e64026e2dbb0af6781947 \
     --ae-epochs 200 --ae-patience 15 --ae-batch-size 32768 \
     --device cuda --metric-sample 5000 --decoder-sample 10000 \
     --distance-pairs 200000 --metric-jobs 1 \
     --timeout 3600 --output-dir artifacts/repro-large-arxiv-d2-d8 --require-all

Run Wikipedia:

.. code-block:: bash

   conda run -n rapids-26.06 env \
     LD_PRELOAD="$RAPIDS_2606/lib/libnvJitLink.so.13" \
     OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
     python docs/results/large-benchmark-harness-b3b75a2.py \
     --dataset-name wikipedia \
     --data artifacts/data/cohere-wikipedia-english-1m-pinned \
     --shard-glob 'emb_*.npy' \
     --labels artifacts/data/cohere-wikipedia-english-1m-pinned \
     --label-column url --label-shard-glob 'meta_*.parquet' --label-policy group \
     --sizes 1000000 --target-dimensions 2 8 \
     --methods autoencoder dire umap --seeds 0 1 2 --neighbors 16 \
     --dire-repo /tmp/dire-rapids-origin-f7d5736 \
     --dire-revision f7d57367af8e10f6013e64026e2dbb0af6781947 \
     --ae-epochs 200 --ae-patience 15 --ae-batch-size 32768 \
     --device cuda --metric-sample 3000 --decoder-sample 5000 \
     --distance-pairs 200000 --metric-jobs 1 \
     --timeout 7200 --output-dir artifacts/repro-large-wikipedia-d2-d8 --require-all

The BF16 arms use the current live driver, which records training and evaluation
precision separately.  ``match-training`` reproduces the standard arm; ``fp32`` isolates
mixed-precision training from validation and final coordinate generation.  These loops
run both policies for arXiv and Wikipedia:

.. code-block:: bash

   for EVAL_PRECISION in match-training fp32; do
     conda run -n rapids-26.06 env \
       LD_PRELOAD="$RAPIDS_2606/lib/libnvJitLink.so.13" \
       OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
       python experiments/compare_large_embeddings.py \
       --dataset-name arxiv \
       --data ../dire-rapids-arxiv/data/embeddings.npy \
       --labels ../dire-rapids-arxiv/data/meta.parquet \
       --label-column primary_category --label-policy category \
       --sizes 723457 --target-dimensions 2 8 \
       --methods autoencoder --seeds 0 1 2 --neighbors 16 \
       --ae-epochs 200 --ae-patience 15 --ae-batch-size 32768 --ae-amp \
       --ae-evaluation-precision "$EVAL_PRECISION" \
       --device cuda --metric-sample 5000 --decoder-sample 10000 \
       --distance-pairs 200000 --metric-jobs 1 --timeout 3600 \
       --output-dir "artifacts/repro-large-arxiv-ae-bf16-$EVAL_PRECISION" \
       --require-all
   done

   for EVAL_PRECISION in match-training fp32; do
     conda run -n rapids-26.06 env \
       LD_PRELOAD="$RAPIDS_2606/lib/libnvJitLink.so.13" \
       OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
       python experiments/compare_large_embeddings.py \
       --dataset-name wikipedia \
       --data artifacts/data/cohere-wikipedia-english-1m-pinned \
       --shard-glob 'emb_*.npy' \
       --labels artifacts/data/cohere-wikipedia-english-1m-pinned \
       --label-column url --label-shard-glob 'meta_*.parquet' --label-policy group \
       --sizes 1000000 --target-dimensions 2 8 \
       --methods autoencoder --seeds 0 1 2 --neighbors 16 \
       --ae-epochs 200 --ae-patience 15 --ae-batch-size 32768 --ae-amp \
       --ae-evaluation-precision "$EVAL_PRECISION" \
       --device cuda --metric-sample 3000 --decoder-sample 5000 \
       --distance-pairs 200000 --metric-jobs 1 --timeout 7200 \
       --output-dir "artifacts/repro-large-wikipedia-ae-bf16-$EVAL_PRECISION" \
       --require-all
   done

Recreate the paired quality/timing summary and the isolated training-step profile with:

.. code-block:: bash

   python experiments/analyze_ae_precision.py
   python experiments/profile_ae_precision.py \
     --warmup-steps 20 --measured-steps 100 \
     --output artifacts/large-ae-precision/microprofile.json

Run the paper-compatible arXiv topology diagnostic once per target width.  The loop
keeps the corpus sample policy, sample size, seeds, and curve resolution fixed:

.. code-block:: bash

   for D in 2 8; do
     conda run -n rapids-26.04 python docs/results/large-topology-harness-d249aed.py \
       --input artifacts/repro-large-arxiv-d2-d8/n723457/prepared-l2.npy \
       --layout autoencoder-s0=artifacts/repro-large-arxiv-d2-d8/n723457/autoencoder-d${D}-s0-layout.npy \
       --layout autoencoder-s1=artifacts/repro-large-arxiv-d2-d8/n723457/autoencoder-d${D}-s1-layout.npy \
       --layout autoencoder-s2=artifacts/repro-large-arxiv-d2-d8/n723457/autoencoder-d${D}-s2-layout.npy \
       --layout dire-s0=artifacts/repro-large-arxiv-d2-d8/n723457/dire-d${D}-s0-layout.npy \
       --layout dire-s1=artifacts/repro-large-arxiv-d2-d8/n723457/dire-d${D}-s1-layout.npy \
       --layout dire-s2=artifacts/repro-large-arxiv-d2-d8/n723457/dire-d${D}-s2-layout.npy \
       --layout umap-s0=artifacts/repro-large-arxiv-d2-d8/n723457/umap-d${D}-s0-layout.npy \
       --layout umap-s1=artifacts/repro-large-arxiv-d2-d8/n723457/umap-d${D}-s1-layout.npy \
       --layout umap-s2=artifacts/repro-large-arxiv-d2-d8/n723457/umap-d${D}-s2-layout.npy \
       --labels artifacts/repro-large-arxiv-d2-d8/n723457/labels.npy \
       --labels-alignment prepared --sample-size 4000 --seeds 0 1 2 --n-steps 50 \
       --dire-repo /tmp/dire-rapids-origin-f7d5736 \
       --expected-betti-sha256 553cc327cb0397c3fc1b058109f720d0db7b1fbe02d064924b07e5f774479ad9 \
       --output-dir artifacts/repro-large-arxiv-d2-d8/topology-d${D}
   done

Run both Wikipedia topology diagnostics with the same uniform samples:

.. code-block:: bash

   for D in 2 8; do
     conda run -n rapids-26.04 python docs/results/large-topology-harness-d249aed.py \
       --input artifacts/repro-large-wikipedia-d2-d8/n1000000/prepared-l2.npy \
       --layout autoencoder-s0=artifacts/repro-large-wikipedia-d2-d8/n1000000/autoencoder-d${D}-s0-layout.npy \
       --layout autoencoder-s1=artifacts/repro-large-wikipedia-d2-d8/n1000000/autoencoder-d${D}-s1-layout.npy \
       --layout autoencoder-s2=artifacts/repro-large-wikipedia-d2-d8/n1000000/autoencoder-d${D}-s2-layout.npy \
       --layout dire-s0=artifacts/repro-large-wikipedia-d2-d8/n1000000/dire-d${D}-s0-layout.npy \
       --layout dire-s1=artifacts/repro-large-wikipedia-d2-d8/n1000000/dire-d${D}-s1-layout.npy \
       --layout dire-s2=artifacts/repro-large-wikipedia-d2-d8/n1000000/dire-d${D}-s2-layout.npy \
       --layout umap-s0=artifacts/repro-large-wikipedia-d2-d8/n1000000/umap-d${D}-s0-layout.npy \
       --layout umap-s1=artifacts/repro-large-wikipedia-d2-d8/n1000000/umap-d${D}-s1-layout.npy \
       --layout umap-s2=artifacts/repro-large-wikipedia-d2-d8/n1000000/umap-d${D}-s2-layout.npy \
       --sample-size 3000 --seeds 0 1 2 --n-steps 50 \
       --dire-repo /tmp/dire-rapids-origin-f7d5736 \
       --expected-betti-sha256 553cc327cb0397c3fc1b058109f720d0db7b1fbe02d064924b07e5f774479ad9 \
       --output-dir artifacts/repro-large-wikipedia-d2-d8/topology-d${D}
   done

The 400-epoch checks apply only to arXiv 2-D and Wikipedia 8-D.  They repeat the
corresponding single-width reducer command with ``--methods autoencoder --seeds 0
--ae-epochs 400 --ae-patience 30`` and a fresh output directory.

The topology evaluator writes exact deterministic sample indices and hashes, so a run
can be rerun with the same declared samples and audited.  It does not incrementally
resume a partially computed set of curves.
