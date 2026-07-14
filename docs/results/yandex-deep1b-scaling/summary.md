# Deep1B 96-to-8 scaling pilot

Results recorded on 2026-07-13. The machine-readable rows are in
[`results.csv`](results.csv), and hashes, sources, configurations, software, and
hardware are in [`provenance.json`](provenance.json).

## What the pilot says

The bounded-block local-file autoencoder (AE) behaved as expected: its application
memory stayed roughly bounded while work and output grew with the number of rows. A
one-pass run reads the local corpus roughly twice (once while training and once while
encoding); this is not network streaming. From 10M to 100M
rows, FP32 GPU training and encoding took 9.71x and 9.94x as long, respectively, while
process peak RSS rose only from 1.48 GB to 2.02 GB (decimal). The unavoidable FP32
8-D layout grew from 0.32 GB to 3.20 GB. More directly, PyTorch peak allocated memory
was exactly 150,232,064 bytes at both scales, with neither the corpus nor output
persistently mapped. This supports *streamable, approximately linear scaling*; it does
not mean literally unbounded scaling because source I/O, output storage, and repeated
training passes still grow with `N`.

RAPIDS UMAP is less stream-like but was not stopped by 100M rows. With 16 clusters,
clustered NN-descent completed at 10M and 100M: fit time grew from 66.40 s to
861.46 s (12.97x), process peak RSS from 12.13 GB to 104.48 GB, and incremental
GPU-process memory above the measured baseline from 7.92 GB to 77.77 GB. Because
the cluster count stayed fixed, the harness's overlap-aware estimated partition grew
from 1.25M to 12.5M rows; that comparison was not a fixed-size-partition test.

A matched 100M control raised the cluster count from 16 to 160, returning the
estimated partition to 1.25M rows and reducing its estimated input from 4.8 GB to
0.48 GB. The run still reached 77.84 GB incremental GPU-process memory, essentially
the same as 77.77 GB with 16 clusters (+0.09%); process RSS was 102.07 GB rather than
104.48 GB (-2.30%). Fit and worker clocks were 836.65 s and 847.41 s, versus 861.46 s
and 874.79 s, but with one run per arm those small changes cannot be distinguished
from run-to-run variation. The nearly identical sampled GPU peaks despite a 10x
smaller estimated partition are the stronger result:
global graph/layout state, rather than partition input alone, dominated the measured
peak. Thus the distinction here is bounded AE working state versus graph/layout state
that grows with `N`, not “AE works and UMAP cannot.”

At 10M rows, the measured core clocks were 5.45 s training plus 3.19 s encoding for
the FP32 GPU AE, 66.40 s fit for UMAP, and 158.14 s fit for DiRe. The corresponding
AE total and graph-worker totals were 12.16 s, 68.82 s, and 161.46 s. These are not
identical timing scopes; the definitions below are part of the result.

The same DiRe configuration did **not** complete at 100M. It spent 2,694.38 s
(44.91 min) in the worker and then raised a recorded CUDA OOM during fit while trying
to allocate its 102.4 GB (95.37 GiB) full negative-index tensor on a 94.50 GiB GPU.
This was an attempted fit, not a preflight rejection or timeout; it published no
layout, and the source remained unchanged. Peak process RSS was 211.10 GB and sampled
NVML PID memory was 63.60 GB total / 63.03 GB above baseline. That sampled peak
occurred earlier; the failed 102.4 GB allocation was never resident, and the backend
returned no `fit_seconds`. This failure applies to the current cuVS-backed
vectorized-force path and configuration
(`k=16`, negative ratio 8, 128 iterations), not to every possible DiRe implementation.

No BF16 speedup was observed for this small MLP on this GH200 run. Relative to FP32,
BF16 training was 15.7% slower (6.30 s versus 5.45 s) and total time was 7.7% slower. Its
held-out quality was effectively unchanged: NMSE moved from 0.45293 to 0.45339 and
reconstruction cosine from 0.75198 to 0.75168. PyTorch peak allocated memory fell
from 150.2 MB to 122.5 MB, but process RSS was higher. These were single sequential
runs, with BF16 immediately before FP32, so cache/order effects are not controlled.
This result does not establish that BF16 is intrinsically slower.

The exact low-dimensional retrieval diagnostic is sobering. On the first 1,000 public
queries against the 10M layout, strict ID recall was 0.0358 at 10 and 0.0677 at 100.
On the first 100 queries against the 100M layout it was 0.0210 and 0.0449. The 100- and
1,000-query 10M results differ because they are different query prefixes. Moreover,
10M used a CPU-trained model while 100M used a GPU-trained model, and the base size,
official ground truth, and query count also changed. The apparent recall drop is
therefore not a controlled `N`-only comparison. These numbers test exact neighbor
identity after compression, not reconstruction error, and no official distance-tie
credit was applied.

The label-free topology sample is descriptive rather than a quality ground truth.
Across three shared 5,000-point evaluation samples, AE had the smallest beta-curve
DTW values (`beta0=1627.7`, `beta1=1954.0`), compared with DiRe
(`4166.7`, `10940.0`) and UMAP (`8943.0`, `13528.0`). The H0 diagnostics are mixed:
AE is closest to the reference for top-five/median and Gini, while UMAP is closest for
the longest finite H0 bar. This is evidence that the layouts are not topologically
interchangeable, not a definitive method ranking.

The matched 100M UMAP partition control was effectively tied on the same sampled
diagnostic. Mean beta-curve DTWs were `10334.0/13167.7` with 16 clusters and
`10312.7/13193.7` with 160 clusters. The paired c160-minus-c16 differences across
the three common samples averaged -21.3 (-0.206%) for beta0 and +26.0 (+0.197%) for
beta1, with mixed signs by seed. H0 descriptors slightly favored c16, but both layouts
remained far from the reference. Given one non-deterministic NN-descent fit per arm,
these tiny differences do not establish a partition-count quality effect.

## Scaling measurements

All sizes below are decimal GB. `--` means that the harness did not record that scope.

| Method / run | Rows | Core fit or training (s) | AE encode (s) | AE total / graph worker total (s) | App peak RSS (GB) | GPU process / PyTorch allocated peak (GB) | Held-out NMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| AE CPU FP32 | 1M | 1.56 train | 0.60 | 3.22 AE total | 0.78 | -- | 0.81702 |
| AE CPU FP32 | 10M | 16.95 train | 4.55 | 22.63 AE total | 0.78 | -- | 0.45293 |
| AE GPU FP32 | 10M | 5.45 train | 3.19 | 12.16 AE total | 1.48 | 0.15 allocator | 0.45293 |
| AE GPU BF16 | 10M | 6.30 train | 3.22 | 13.09 AE total | 1.61 | 0.12 allocator | 0.45339 |
| AE GPU FP32 | 100M | 52.86 train | 31.71 | 88.18 AE total | 2.02 | 0.15 allocator | 0.39255 |
| UMAP GPU, 16 clusters | 10M | 66.40 fit | -- | 68.82 worker | 12.13 | 8.49 process | -- |
| UMAP GPU, 16 clusters | 100M | 861.46 fit | -- | 874.79 worker | 104.48 | 78.34 process | -- |
| UMAP GPU, 160 clusters | 100M | 836.65 fit | -- | 847.41 worker | 102.07 | 78.41 process | -- |
| DiRe GPU | 10M | 158.14 fit | -- | 161.46 worker | 19.82 | 29.63 process | -- |
| DiRe GPU | 100M | CUDA OOM in fit | -- | 2694.38 failed worker | 211.10 | 63.60 process | -- |

The 1M and 10M one-pass AE runs do not have an equal optimizer-update budget: they
made 115 and 1,221 updates, while the 100M run made 12,208. The improving AE NMSE with
`N` therefore combines more data with more optimization and is not a controlled
sample-size-only comparison.

## Exact retrieval

The search is an exact streamed search in the 8-D layout with float64 squared
Euclidean distances. It scans each base layout once and retains bounded per-query
top-k state; it does not materialize the full query-by-base matrix.

| Base layout | Query prefix | Recall@10 | Recall@100 |
|---|---:|---:|---:|
| AE CPU FP32, 10M | 100 | 0.0300 | 0.0595 |
| AE CPU FP32, 10M | 1,000 | 0.0358 | 0.0677 |
| AE GPU FP32, 100M | 100 | 0.0210 | 0.0449 |

Recall is the mean per-query set intersection between retrieved IDs and the first `k`
official original-space ground-truth IDs, divided by `k`. Unlike the official Big ANN
protocol, alternative IDs tied in original-space distance are **not** credited.

## Timing and memory semantics

- **AE training** is the complete streamed training loop: source-block reads, local
  shuffling and batching, host-to-device copies, forward/backward passes, and optimizer
  updates. **AE encoding** is the complete base-layout source read, hash, forward pass,
  and layout write. **AE total** also includes preflight and
  validation, held-out evaluation, source validation and hashing, and query encoding;
  downloading and final result/pointer publication are excluded.
- **Graph fit** is the backend `fit_transform` call, which produces a fitted layout.
  **Graph worker total** also includes backend import (where separately recorded),
  bounded input inspection, layout validation, and layout writing. Parent-process
  launch/wait overhead is not the reported worker total.
- **Application RSS** is the process high-water resident set. Graph RSS can include
  touched pages from its read-only input mmap; the AE's `pread`-driven OS page cache is
  outside AE process RSS. Graph **GPU-process memory** is all PID allocations reported
  by NVML and sampled at 0.1 s; the table shows total peak, while the prose also removes
  the recorded baseline. AE **PyTorch allocator memory** is PyTorch's peak
  allocated/reserved counter, not total GPU-process memory. On Grace Hopper these
  scopes can overlap and must not be added or treated as a single total-memory measure.
- Retrieval RSS is `ru_maxrss` from a fresh CPU process. The 67.1 MB value is a process
  high-water mark, while its explicit bounded-workspace estimate excludes process
  baseline and OS page cache.

## Topology protocol

One already-fitted 10M-row layout per method (AE seed 1729; UMAP and DiRe seed 0) was
uniformly reduced to a common 20,000-row prepared set. The 100M UMAP control separately
prepared the c16 and c160 layouts on another common 20,000-row sample; both reducer
fits used seed 0 and differed in reducer configuration only by `knn_n_clusters`.
For each prepared set, three shared, uniform 5,000-row subsamples (seeds 17, 29, and
43) were rescaled to unit diameter. The evaluator then computed 50-step Betti curves,
fast-DTW distances to the 96-D reference curves, and finite-H0 persistence summaries.
Lower DTW means closer sampled curves. The H0 summaries have no universal “higher is
better” direction.

There are no class labels in either evaluation. The topology seeds are evaluation
samples from one fitted layout per arm, not independent reducer fits. Because each
draws 5,000 points from the same prepared 20,000, pairwise sample overlap is about
24–26%; reported standard deviations are descriptive, not independent confidence
estimates. The results are sensitive to sampling, 50-step discretization, per-cloud
filtration axes, and DTW alignment; they are not proofs of homology preservation.

## Caveats and next step

- Data download time is excluded, and several experiments ran consecutively, so OS
  page cache may have been warm. The host had about 526 GiB of RAM, and neither the
  application buffer budget nor RSS isolates or clears the system page cache. Treat
  these as single-machine local-file throughput observations, not cold-storage or
  network-streaming benchmarks.
- Every AE result is one pass. Every graph row is one fitted layout and one reducer
  seed. RAPIDS NN-descent is documented by the harness as non-deterministic even with
  `random_state` set.
- UMAP used `knn_n_clusters=16` in the 10M-to-100M scaling comparison. The additional
  100M `knn_n_clusters=160` control restored the 10M arm's estimated partition size
  but did not materially lower sampled GPU-process memory. Both 100M measurements are
  single runs of non-deterministic NN-descent; their small timing and RSS differences
  should not be interpreted as a stable cluster-count effect.
- DiRe's 100M OOM is an observed result for the tested implementation and parameters;
  it is not a claim that every partitioned or reformulated DiRe must fail at 100M.
- The full 1B base file is 384,000,000,008 bytes before its 32 GB 8-D output. At bundle
  creation the filesystem had only about 218.4 GB free, so the 1B source did not fit
  locally. The tested ceiling is therefore 100M, not 1B.
- The practical next experiment is the same one-pass AE protocol on 1B with at least
  about 450 GB of dedicated free storage (more for safe atomic output and other runs),
  followed by a fixed-update arm if training-quality comparisons are desired.
