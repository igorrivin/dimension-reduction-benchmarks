# Predeclared Deep1B autoencoder update-budget protocol

Status: revised after pre-run audit and re-frozen before the production discovery run on
2026-07-14. No production outcomes were generated under the superseded draft.

## Question and scope

Find the earliest optimizer-update budget that is practically no worse than a demonstrably
longer trajectory for the fixed Deep1B 96-to-8 MSE autoencoder. This protocol does not test the
stability of the intrinsic-dimension elbow and does not claim a corpus-independent training
budget.

The 65,536-row training pool is primary. The 1,048,576-row pool is a separately reported
pool-by-budget robustness check. Pools are never pooled statistically. A universal operational
recommendation, if both confirm, is the larger of their two confirmed budgets.

## Invariants

- Source universe, model, optimizer, batch size, preprocessing, precision, split seeds, and pool
  order are identical to the completed training-pool-size experiment. Discovery and confirmation
  use the same deterministic *local sampling rules* for geometry, but their global row identities
  differ because discovery samples the tune cohort and confirmation samples the legacy cohort.
- Architecture: symmetric 96-256-128-32-8 autoencoder, trained with elementwise MSE and AdamW.
- Batch size: 8,192. Precision: deterministic CUDA FP32 with TF32 disabled.
- A checkpoint is an uninterrupted prefix of the same initialized trajectory as its long-budget
  reference. Model-only resumption is not permitted because it would lose AdamW and batch-order
  state.
- Initialization seed is the experimental unit. Geometry rows and sampled pairs are not
  independent replicates.

## Metrics and signed degradation

For candidate checkpoint `B` and reference checkpoint `R` from the same seed and pool, signed
degradation is positive when `B` is worse. Relative margins are computed against `R`.

| Metric | Signed degradation | Non-inferiority margin |
|---|---|---:|
| Reconstruction NMSE | `B / R - 1` | 0.010 relative |
| Reconstruction cosine | `R - B` | 0.005 absolute |
| Trustworthiness | `R - B` | 0.002 absolute |
| Continuity | `R - B` | 0.002 absolute |
| Neighbor overlap at 16 | `R - B` | 0.005 absolute |
| Pair-distance Spearman | `R - B` | 0.010 absolute |
| Normalized stress | `B - R` | 0.005 absolute |
| Local-decoder NMSE | `B / R - 1` | 0.010 relative |

These one-sided harm margins define practical non-inferiority, not symmetric equivalence.

## Phase 1: tune-only discovery

- Seeds: 1729, 1730, and 1731.
- Pools: 65,536 and 1,048,576 rows, analyzed separately.
- Coarse update grid: 0, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1,024,
  1,204, 1,808, 2,408, 3,612, and 4,816.
- Cohort: the fixed disjoint tuning cohort only. Reconstruction uses all 65,536 tuning rows;
  geometry uses fixed deterministic 5,000-row rank, 10,000-row decoder, and 200,000-pair
  subsets. The legacy cohort must not be materialized or reported by discovery.
- Reference: the configured terminal checkpoint, initially 4,816 updates. Update 1,204 remains a
  replay anchor, not the reference by definition.

At a coarse checkpoint, every metric passes when its mean signed degradation is strictly below
its margin and no individual seed exceeds twice that margin. Update zero is descriptive and is
excluded from candidate selection. A checkpoint is a sustained coarse pass only when it and the
next two chronologically later coarse checkpoints all pass; consequently, the final two coarse
points cannot themselves be candidates. The earliest sustained coarse pass defines the upper
boundary. Its lower boundary is the largest earlier coarse checkpoint that fails; if none exists,
the lower boundary is update zero. Points strictly inside that bracket are then filled every 16
updates when the upper boundary is at most 256, and every 32 updates otherwise. Pass/fail status
is recomputed on the combined ordered grid. The locked discovery candidate is the earliest
positive checkpoint for which it and the next two chronologically later points on that combined
grid pass. This definition remains unchanged if earlier isolated passes are followed by failures.

The post-1,204 points determine whether “plateau” is defensible. For each adjacent terminal pair
and seed, compute the absolute relative change `abs(later / earlier - 1)` for reconstruction NMSE
and local-decoder NMSE, and the absolute paired difference for every other metric. Average those
absolute paired changes across the three seeds. The terminal region is called flat only when, for
both 2,408-to-3,612 and 3,612-to-4,816, mean absolute reconstruction-NMSE change is at most 0.25%
relative and every other metric's mean absolute change is within its full tabled margin (including
the 1% relative local-decoder margin). This rule treats improvement and worsening symmetrically.
If either pool fails, discovery is extended once to 7,224 and 9,632 updates and the analogous
4,816-to-7,224 and 7,224-to-9,632 intervals are checked. On extension, 9,632 becomes the sole
reference: every signed degradation, pass/fail status, bracket, fine-grid result, and candidate is
discarded and recomputed against 9,632. If the extended terminal check still fails, the study
reports that no plateau was observed through 9,632 updates and uses “practical sufficiency
relative to the tested terminal budget” language only.

Before interpreting discovery, the uninterrupted trajectories at update 1,204 must reproduce
the earlier split, pool, initialization, model-state, and tune-reconstruction identities. Exact
model-state replay is the firewall-compatible anchor; prior geometry used the legacy cohort and
is deliberately not recomputed during tune-only discovery.

## Phase 2: frozen-candidate confirmation

- Fresh seeds: 1732 through 1743 inclusive.
- The pool-specific candidate, fine-grid step, terminal reference, metrics, and margins are frozen
  before confirmation begins.
- Evaluate `max(0, candidate-minus-one-step)`, candidate, candidate-plus-one-step, update 1,204,
  and the
  terminal reference. The formal comparison is only the locked candidate against the terminal
  reference; the neighboring points and update 1,204 are descriptive.
- Cohort: the fixed legacy evaluation cohort and its previously pinned geometry subsets. This
  cohort was used in earlier work, so it is a fresh-seed confirmation on a legacy cohort rather
  than a pristine final test.

For each metric, compute paired signed degradation across the 12 seeds. The candidate passes a
metric only when the upper one-sided 95% paired-t confidence bound lies strictly below the frozen
margin. All metrics must pass. This intersection-union non-inferiority gate needs no additional
multiplicity correction. If any metric fails, report that the proposed saving was not confirmed;
do not adapt the candidate after inspecting confirmation results.

## Optional locked-candidate retrieval

Retrieval is not a stopping metric. After the primary 65,536-pool candidate is locked, seed 1732
is encoded at the candidate and terminal reference over the identical complete 10M base and the
first 1,000 public queries. One exact top-100 scan per checkpoint supplies strict Recall@10 and
Recall@100. With one seed this remains descriptive.
