# Structure-aware AE and 32/64-D prepass pilot

Results recorded on 2026-07-13. This note covers two related questions:

1. Can a bounded, label-free structure term improve a reconstruction autoencoder's
   two-dimensional geometry?
2. Can a fast 384-to-32/64 prepass make DiRe or UMAP faster without losing too much
   final-layout quality?

The short answer is mixed. Sparse normalized stress robustly improved the easy linear
control, but neither auxiliary loss reliably improved the digits clusters, and the
Swiss-roll trade-off changed across seed splits. On the 100k-row arXiv pilot, every
compressed input made the downstream graph fit faster, but an AE prepass was much too
expensive end to end. PCA was approximately break-even for one DiRe fit, slower for
one UMAP fit, and modestly useful when reused by both; it was a stronger practical
control than the AE in this experiment.

Machine-readable evidence and harnesses:

- Structure screen, seeds 0--2: [`results.json`](structure-screen/results.json),
  [`results.csv`](structure-screen/results.csv), and
  [`summary.csv`](structure-screen/summary.csv).
- Locked structure confirmation, seeds 3--5:
  [`results.json`](structure-confirmation/results.json),
  [`results.csv`](structure-confirmation/results.csv), and
  [`summary.csv`](structure-confirmation/summary.csv).
- arXiv prepass pilot: [`results.json`](prepass-arxiv-100k/results.json)
  and [`results.csv`](prepass-arxiv-100k/results.csv).
- Harnesses:
  [`compare_structure_aware_autoencoder.py`](../../../experiments/compare_structure_aware_autoencoder.py)
  and [`compare_prepass_reducers.py`](../../../experiments/compare_prepass_reducers.py).

## 1. Sparse structure-aware reconstruction loss

### Protocol

The paired pilot used 1,000 rows, a two-dimensional bottleneck, 200 fixed epochs, and
identical initial weights and reconstruction-batch order within each dataset/seed.
Digits has 64 input features; the noisy linear and Swiss-roll controls were embedded in
32 dimensions. The MSE-only arm was compared with:

- **rank**: a scale-normalized InfoNCE loss on
  `(anchor, positive, near-negative, uniform-negative)` tuples, weighted by 0.03 or
  0.3;
- **stress**: scale-normalized squared distance stress, weighted by 0.1.

Both auxiliaries used only prepared input features. A deterministic 256-row reservoir
defined 15-neighbor positives, 3,840 ranking tuples, and 2,048 stress pairs. An
auxiliary minibatch was applied every fourth reconstruction update. Labels were never
used for graph construction or training; digits labels were passed only to the
post-fit evaluator.

The tables report arithmetic means over three paired seeds. `Comp` metrics are
computed on the 744 rows outside the structure reservoir, an explicit check against
merely memorizing those 256 rows. `Overlap` is 15-neighbor set overlap. `Intrinsic`
distance correlation/stress use the known latent geometry and therefore exist only for
the synthetic controls. Label kNN accuracy exists only for digits. Lower NMSE and
stress are better; higher other values are better. Fit time includes the roughly
0.50-second structure preparation for auxiliary arms.

### Screen: seeds 0--2

| Dataset | Loss | Fit (s) | NMSE | Comp overlap | Comp intrinsic rho | Comp intrinsic stress | Comp label kNN |
|---|---|---:|---:|---:|---:|---:|---:|
| digits | MSE | 1.897 | 0.3171 | 0.4922 | -- | -- | 0.8889 |
| digits | rank 0.03 | 2.749 | 0.3255 | 0.4894 | -- | -- | 0.8781 |
| digits | rank 0.3 | 2.737 | 0.4303 | 0.3926 | -- | -- | 0.8239 |
| digits | stress 0.1 | 2.820 | 0.3209 | 0.4923 | -- | -- | 0.8907 |
| linear | MSE | 1.897 | 0.001505 | 0.9379 | 0.9888 | 0.0753 | -- |
| linear | rank 0.03 | 2.763 | 0.001600 | 0.9420 | 0.9869 | 0.0782 | -- |
| linear | rank 0.3 | 2.745 | 0.006157 | 0.9013 | 0.9723 | 0.1058 | -- |
| linear | stress 0.1 | 2.738 | 0.001504 | 0.9723 | 0.9995 | 0.0152 | -- |
| Swiss roll | MSE | 1.906 | 0.04578 | 0.5833 | 0.2195 | 0.6719 | -- |
| Swiss roll | rank 0.03 | 2.771 | 0.04426 | 0.6168 | 0.1903 | 0.6606 | -- |
| Swiss roll | rank 0.3 | 2.772 | 0.04984 | 0.6162 | 0.2553 | 0.6253 | -- |
| Swiss roll | stress 0.1 | 2.759 | 0.04331 | 0.6151 | 0.1694 | 0.6592 | -- |

The screen selected rank 0.3 because it was the ranking arm that improved the
Swiss-roll intrinsic correlation, and retained stress 0.1 as the other candidate.
Rank 0.03 had good Swiss-roll overlap and NMSE but moved intrinsic correlation in the
wrong direction. It was not carried into the locked split.

### Locked confirmation: seeds 3--5

| Dataset | Loss | Fit (s) | NMSE | Comp overlap | Comp intrinsic rho | Comp intrinsic stress | Comp label kNN |
|---|---|---:|---:|---:|---:|---:|---:|
| digits | MSE | 2.010 | 0.3138 | 0.4952 | -- | -- | 0.9037 |
| digits | rank 0.3 | 2.875 | 0.4276 | 0.4000 | -- | -- | 0.8342 |
| digits | stress 0.1 | 2.907 | 0.3144 | 0.4881 | -- | -- | 0.9037 |
| linear | MSE | 1.983 | 0.001556 | 0.9186 | 0.9868 | 0.0831 | -- |
| linear | rank 0.3 | 2.884 | 0.006780 | 0.9134 | 0.9845 | 0.0807 | -- |
| linear | stress 0.1 | 2.884 | 0.001555 | 0.9691 | 0.9993 | 0.0175 | -- |
| Swiss roll | MSE | 2.013 | 0.03859 | 0.6420 | 0.1722 | 0.6690 | -- |
| Swiss roll | rank 0.3 | 2.875 | 0.05543 | 0.5700 | 0.2618 | 0.6182 | -- |
| Swiss roll | stress 0.1 | 2.863 | 0.03824 | 0.6554 | 0.2115 | 0.6335 | -- |

Normalized stress is a useful positive control: on linear data it preserved NMSE and
raised complement overlap by 3.44 and 5.05 percentage points in the screen and locked
split, while reducing intrinsic stress from 0.0753 to 0.0152 and from 0.0831 to
0.0175. The effect is not merely on reservoir rows.

The harder cases do not support a general win. On digits, stress was essentially tied
in the screen and lost 0.71 overlap points in confirmation; label kNN accuracy was
unchanged in confirmation. Rank 0.3 clearly hurt digits. On Swiss roll, rank 0.3
improved intrinsic correlation/stress in both splits, but confirmation overlap fell
7.20 points and NMSE rose from 0.03859 to 0.05543. Stress 0.1 raised overlap in both
splits and complement trustworthiness from 0.9714 to 0.9792 in the screen and from
0.9809 to 0.9822 in confirmation, but its intrinsic-correlation direction changed
between splits. Three seeds per split are not enough to treat the smaller changes as
stable effects.

The auxiliary computation was bounded, but it was not free at this tiny scale. Training
alone rose by roughly 18--21%; including structure preparation, total fit time rose by
about 43--49% (roughly 0.85--0.90 seconds). Only reservoir construction is fixed-size:
the auxiliary loss still fires every fourth optimizer update, so its training overhead
can scale with the update count and need not vanish as `N` grows.

## 2. arXiv 384-to-32/64 prepass before DiRe and UMAP

### Protocol

The harness selected the same deterministic 100,000 rows from the 723,457-by-384
L2-normalized arXiv matrix for every cell. It compared a direct 384-D input with PCA
raw scores, whitened PCA, and MSE-autoencoder codes at widths 32 and 64. Whitened
variants used `center -> regularized covariance whitening -> row L2 normalization`;
covariance whitening addresses the affine scale/basis non-identifiability caused by a
decoder that can absorb an invertible affine change of latent coordinates. Row-L2
normalization is a separate angular-metric heuristic, not a consequence of that
identifiability argument. PCA raw scores were kept as a control on both transformations.

Each prepass was cached and then passed to fresh DiRe and RAPIDS UMAP workers for an
8-D final layout. There were 42 successful cells: seven input representations, two
reducers, and seeds 0--2. DiRe used 16 neighbors, 128 layout iterations, an IVF-flat
cuVS index, and squared Euclidean distance. UMAP used 16 neighbors, Euclidean distance,
random initialization, and NN-descent. Every final metric below is evaluated against
the original selected 384-D input, not against the intermediate code.

The AE used FP32 MSE training, a 5% validation split, and a fresh full-data refit at the
selected epoch. Five of six runs selected 200 epochs and one selected 199, so the
reported 19--24 seconds include almost the full 200-epoch selection plus refit rather
than a deliberately cheap one-pass encoder.

### Intermediate fidelity

These are medians across seeds (PCA is identical apart from small timing variation).
The PCA reconstruction columns are the same before and after canonicalization because
reconstruction uses the raw scores.

| Prepass representation | Prepass + canonicalize (s) | Reconstruction NMSE | 16-NN overlap | Trustworthiness | Distance rho |
|---|---:|---:|---:|---:|---:|
| PCA raw 32 | 0.327 | 0.3433 | 0.5593 | 0.9943 | 0.9526 |
| PCA whitened + L2, 32 | 0.370 | 0.3433 | 0.3489 | 0.9558 | 0.3361 |
| AE whitened + L2, 32 | 23.777 | 0.3369 | 0.3414 | 0.9585 | 0.3095 |
| PCA raw 64 | 0.326 | 0.2410 | 0.6991 | 0.9984 | 0.9772 |
| PCA whitened + L2, 64 | 0.415 | 0.2410 | 0.3409 | 0.9450 | 0.2645 |
| AE whitened + L2, 64 | 19.755 | 0.2352 | 0.3304 | 0.9398 | 0.2588 |

The AE reconstructed slightly better than same-width PCA, but that did not translate
to a better metric representation. Raw PCA preserved much more original-space
neighborhood and distance structure. After applying the same whitening and row
normalization, PCA and AE were quite similar, suggesting that much of the locality
loss here comes from the chosen canonical metric, not uniquely from AE training.

### Final-layout quality

All values are medians over seeds 0--2. `Bal. acc.` is distance-weighted kNN balanced
accuracy on the fixed evaluation sample after excluding empty or singleton categories:
4,973 rows and 95 classes. Labels are evaluation-only. The ordinary accuracy and all
per-seed values remain in the JSON/CSV artifacts.

| DiRe input | Overlap | Trust | Distance rho | Stress | Bal. acc. |
|---|---:|---:|---:|---:|---:|
| direct 384 | 0.1687 | 0.8812 | 0.3071 | 0.4170 | 0.1869 |
| PCA raw 32 | 0.1876 | 0.9135 | 0.3134 | 0.3672 | 0.1739 |
| PCA raw 64 | 0.1862 | 0.9034 | 0.2506 | 0.3825 | 0.1966 |
| PCA whitened + L2, 32 | 0.1815 | 0.9033 | 0.3792 | 0.3274 | 0.2182 |
| PCA whitened + L2, 64 | 0.1823 | 0.9013 | 0.4119 | 0.3346 | 0.2608 |
| AE whitened + L2, 32 | 0.1859 | 0.9126 | 0.3603 | 0.3451 | 0.2149 |
| AE whitened + L2, 64 | 0.1803 | 0.9010 | 0.4336 | 0.3326 | 0.2418 |

| UMAP input | Overlap | Trust | Distance rho | Stress | Bal. acc. |
|---|---:|---:|---:|---:|---:|
| direct 384 | 0.2441 | 0.9372 | 0.6052 | 0.3428 | 0.2455 |
| PCA raw 32 | 0.2338 | 0.9415 | 0.5078 | 0.3229 | 0.1910 |
| PCA raw 64 | 0.2434 | 0.9419 | 0.5905 | 0.3269 | 0.2373 |
| PCA whitened + L2, 32 | 0.2197 | 0.9262 | 0.5449 | 0.3318 | 0.2298 |
| PCA whitened + L2, 64 | 0.2093 | 0.9154 | 0.5493 | 0.3406 | 0.2850 |
| AE whitened + L2, 32 | 0.2191 | 0.9308 | 0.5383 | 0.3248 | 0.2202 |
| AE whitened + L2, 64 | 0.2082 | 0.9146 | 0.5500 | 0.3391 | 0.2659 |

There is no single quality winner. Every compressed input improved DiRe median overlap
and trust over direct input. PCA raw 32 had the best DiRe locality; AE 64 had the best
DiRe distance correlation, but whitened PCA 64 was close and had higher balanced
category accuracy. For UMAP, direct input and PCA raw 64 retained the best locality;
the whitened/AE inputs lost overlap. Some compressed layouts reduced stress or raised
balanced category accuracy while worsening locality, so those are trade-offs rather
than dominance claims.

Matched AE-minus-whitened-PCA deltas show no broad AE advantage. The one repeatable AE
edge was DiRe-64 distance correlation: its mean paired gain was 0.0171 and was positive
in all three seeds. At the same width, however, PCA had slightly better mean category
scores and essentially tied local quality/stress. AE's reconstruction-NMSE advantage
over PCA was only 0.00675 at width 32 and 0.00592 at width 64 (positive in all three
paired seeds), and did not become a general downstream advantage. Direct UMAP remains
the best general-purpose UMAP choice; raw PCA 64 is the least disruptive compressed
alternative if its prepass can be amortized.

### Timing

`Graph` is only the downstream `fit_transform`. Timing uses the arithmetic mean of the
two steady runs, seeds 1 and 2; the seed-0 exception is explained below. `Shared core`
adds one mean prepass and canonicalization to the mean DiRe and UMAP graph fits,
assuming that the same cached representation is reused by both reducers. It excludes
interpreter/backend startup and reconstruction diagnostics. This derived
shared-pipeline view is the fair one for a prepass intended to serve both methods.

There was no one-shot PCA speedup. Steady direct DiRe took 3.023 seconds core end to
end, versus 3.044--3.152 seconds with the four PCA inputs; direct UMAP took 0.753
seconds, versus 0.899--1.015 seconds with PCA. The table below instead tests the
different, amortized case in which one cached prepass feeds both reducers.

| Shared input | Prepass + canonicalize (s) | DiRe graph (s) | UMAP graph (s) | Shared core (s) | Versus direct |
|---|---:|---:|---:|---:|---:|
| direct 384 | 0.000 | 3.023 | 0.753 | 3.776 | baseline |
| PCA raw 32 | 0.327 | 2.722 | 0.573 | 3.621 | -4.10% |
| PCA raw 64 | 0.326 | 2.719 | 0.601 | 3.646 | -3.45% |
| PCA whitened + L2, 32 | 0.373 | 2.723 | 0.577 | 3.673 | -2.72% |
| PCA whitened + L2, 64 | 0.417 | 2.735 | 0.598 | 3.750 | -0.69% |
| AE whitened + L2, 32 | 23.774 | 2.699 | 0.565 | 27.038 | +616.04% (7.16x) |
| AE whitened + L2, 64 | 19.556 | 2.753 | 0.592 | 22.902 | +506.51% (6.07x) |

The graph-stage hypothesis is correct at this scale: reduced inputs cut steady UMAP
fit by about 20--25% and DiRe fit by about 9--11%. PCA's approximately 0.33-second
fit almost amortized, leaving only a 1--4% derived shared-core improvement. That margin
is too small, on two steady sequential runs, to claim a robust speedup. The full AE
selection/refit cost overwhelmed its graph-stage saving.

One timing cell needs special treatment. Direct DiRe was the first scheduled DiRe fit,
and its seed-0 graph clock was 5.208 seconds, versus 3.008 and 3.039 seconds for seeds 1
and 2. That isolated first-cell excess is consistent with a cold order/JIT effect, but
the artifact does not establish the cause. In addition, the cached seed-0 artifacts
lack the newer outer worker-launch measurement, so their process-wall fallback is not
commensurable with seeds 1--2. Timing conclusions therefore use the 3.023-second
steady-run mean, not seed 0 or a three-seed mean; three-seed quality values remain
valid because they do not compare clocks.

The steady memory counters do not reveal a full-pipeline peak-memory win. DiRe's
incremental graph-stage GPU peak was 516 MiB for both direct and reduced inputs. UMAP
fell from 302 MiB direct to 218 MiB reduced, but the PCA and AE prepasses themselves
peaked at 318 and 536 MiB, respectively.

## 3. Yandex Deep1B 96-to-8 scaling context

A separate local-file benchmark tested a one-pass MSE AE and the direct graph methods
on public 96-D Yandex Deep1B vectors at 10M and 100M rows. The complete account is in
the [Deep1B scaling note](../yandex-deep1b-scaling/summary.md); its
[machine-readable results](../yandex-deep1b-scaling/results.csv) and
[provenance](../yandex-deep1b-scaling/provenance.json) define the configurations,
hashes, hardware, and timing/memory scopes. The full 1B source was not run because it
exceeded the remaining local disk; 100M is the tested ceiling. The relevant
measurements are:

| Method | Rows | Core clock (s) | Total/worker clock (s) | Process peak RSS (GB) | GPU counter (GB) |
|---|---:|---:|---:|---:|---:|
| AE GPU FP32 | 10M | 5.45 train + 3.19 encode | 12.16 AE total | 1.48 | 0.150 PyTorch allocated |
| AE GPU FP32 | 100M | 52.86 train + 31.71 encode | 88.18 AE total | 2.02 | 0.150 PyTorch allocated |
| UMAP, 16 clusters | 10M | 66.40 fit | 68.82 worker | 12.13 | 8.49 NVML process |
| UMAP, 16 clusters | 100M | 861.46 fit | 874.79 worker | 104.48 | 78.34 NVML process |
| DiRe | 10M | 158.14 fit | 161.46 worker | 19.82 | 29.63 NVML process |
| DiRe | 100M | OOM; no fit clock | 2,694.38 failed worker | 211.10 | 63.60 NVML process |

Each graph row is one fitted layout and one reducer seed, and the local-file page
cache may have been warm. The clocks are therefore descriptive single-machine
measurements rather than stable timing estimates.

The GPU counters in the table have different scopes and must not be compared as if
they were one memory measure: AE reports the PyTorch allocator, whereas graph workers
report sampled whole-process NVML memory. The bounded-block AE nevertheless showed the
expected operational scaling. From 10M to 100M, training and encoding clocks grew
9.71x and 9.94x, process RSS rose only from 1.48 to 2.02 GB, and PyTorch peak allocated
memory was exactly 150,232,064 bytes at both sizes. Its unavoidable FP32 8-D output
grew from 0.32 to 3.20 GB. This supports approximately linear work and bounded
application working state, not literally unlimited streaming: the one-pass run read
the local source roughly twice, and source I/O, output storage, and optimizer work
still grow with row count.

UMAP completed at 100M, but its measured state grew sharply. The incremental NVML
peak above baseline was 7.92 GB at 10M and 77.77 GB at 100M. A matched 100M control
raised the cluster count from 16 to 160, cutting the estimated partition input by
10x, yet fit/worker time was still 836.65/847.41 seconds, process RSS was 102.07 GB,
and total/incremental NVML peaks were 78.41/77.84 GB. That near-tie points to global
graph/layout state rather than partition input as the dominant measured peak.

The tested DiRe path did not complete at 100M. It ran for 2,694.38 seconds before a
CUDA OOM while trying to allocate a 102.4 GB (95.37 GiB) full negative-index tensor on
a 94.50 GiB GPU. The failed allocation was not resident, so the earlier sampled
63.60 GB NVML peak does not measure it. This result is scoped to the current
cuVS-backed vectorized-force implementation and tested configuration (`k=16`, negative
ratio 8, 128 iterations), not to every possible DiRe formulation.

The quality evidence prevents treating AE throughput as a complete victory. Exact
streamed retrieval in the 8-D AE layout gave strict ID recall of 0.0358 at 10 and
0.0677 at 100 for the first 1,000 queries against 10M rows; for the first 100 queries
against 100M rows it was 0.0210 and 0.0449. Tied original-space alternatives received
no credit, and the two sizes used different query prefixes and training devices, so
the apparent decline is not a controlled `N` effect. It is nevertheless direct
evidence that optimizing aggregate reconstruction loss does not by itself guarantee
preservation of exact neighbors.

The label-free topology diagnostic is descriptive rather than a definitive method
ranking. On
three shared 5,000-point samples from one 10M layout per method, AE had the lowest
beta-curve DTW values (`beta0=1627.7`, `beta1=1954.0`), compared with DiRe
(`4166.7`, `10940.0`) and UMAP (`8943.0`, `13528.0`), while finite-H0 summaries were
mixed. There were no labels, the samples overlapped, and each method contributed only
one fitted layout; the result shows that the layouts are topologically different, not
that AE is definitively best.

These scale results change the interpretation, but not the prime-time verdict for the
two hacks in this note. They establish a genuine advantage for bounded-block AE
execution at large `N`, while the strict-retrieval result strengthens the motivation
for a geometry-aware loss. They do not validate the sampled structure loss, which was
tested only at 1,000 rows, or an AE prepass: the Deep1B AE went directly from 96 to 8
dimensions, and reducing feature width would not remove UMAP's global graph/layout
state or DiRe's full negative-index tensor. Structure-aware AE and AE/PCA prepasses
therefore remain research options rather than defaults.

## Caveats and next experiment

- This is a 100k-row transductive pilot, not a large-`N` streaming test. Each prepass
  saw all selected rows, and the results do not establish out-of-sample behavior.
- Locality, trust/continuity, and category metrics are computed within one fixed
  5,000-row sample, rather than from full-100k neighborhoods. Distance correlation and
  stress use 200,000 pairs sampled across all 100,000 rows.
- DiRe's nominally common random initialization is an input-dependent random
  projection. Changing the prepass therefore changes both the input metric and the
  starting coordinates, so DiRe quality deltas cannot be attributed to the graph alone.
- RAPIDS UMAP used NN-descent, which the harness records as not exactly deterministic
  even with a fixed random seed. Three reducer seeds describe this run; they are not a
  precision estimate of a deterministic algorithm.
- Process timing has only two commensurable steady runs. The small shared-PCA savings
  are descriptive and may be order or run-to-run variation.
- The structure study used only three screen and three locked seeds, and the datasets
  are small. Its MSE baseline is a paired fixed-epoch baseline, not the repository's
  authoritative early-stopping-plus-refit protocol.
- The structure losses supervise reservoir-conditional distances, not exact full-data
  trustworthiness or hard kNN overlap. Hard metrics are not directly differentiable.

The practical scale-up should be narrow: compare direct input, raw PCA 32, and
whitened-PCA 64 on 300k/full arXiv to see whether graph-stage savings grow into a real
one-shot speedup. If AE work continues, the more informative research experiment is a
metric-regularized AE that tries to match raw-PCA locality without relying on post-hoc
whitening. That would connect the two pilots and directly test whether learned
reconstruction plus sparse geometry offers anything beyond the inexpensive linear
control.
