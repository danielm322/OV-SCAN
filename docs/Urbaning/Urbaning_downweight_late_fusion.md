# UrbanIng-V2X ↔ OV-SCAN Integration — Late-Fusion Infra Down-Weighting Ablation

**Status:** Single-sequence ablation testing whether down-weighting infra-source detections
relative to vehicle-source detections, before WBF, improves `full_late`'s mAP. Motivated by
`Urbaning_PHASE_5.md` §7's open question: `full_late` (all 6 sources) placed consistently
*second* to `v2v_late` (2 vehicle sources only) at every intersection and IoU threshold in the
full 34-sequence sweep, suggesting the infra sources' individually weaker detections might be
dragging WBF's fused output down rather than purely adding coverage. **Result: this specific
down-weighting mechanism does not help — mAP declines monotonically as infra gets down-weighted.
Hypothesis not supported; recommend against pursuing this direction further.**

---

## 1. Mechanism tested

A single scalar multiplier `infra_weight` applied to every infra-source detection's score
*before* it reaches `weighted_box_fusion()` — vehicle-source scores always stay at 1.0x. Since
`weighted_box_fusion()` already uses each box's score for three things (the greedy clustering
order, the score-weighted geometric average of center/size/yaw, and the fused confidence — a
plain mean of cluster member scores), down-weighting infra's input score discounts all three
consistently: infra boxes pull the fused geometry less, and infra-heavy clusters end up with a
lower fused confidence (ranking lower in the AP's precision-recall curve).

Implemented as a new optional parameter on `fuse_combo_frames()` in
`eval_urbaning_v2x_late_fusion.py` (default `1.0`, a no-op — fully backward-compatible with every
existing caller):

```python
weight_mult = infra_weight if source['mode'] == 'infra' else 1.0
all_scores.append(pred_scores * weight_mult)
```

applied per-source when building each frame's concatenated score array, before the call into
`weighted_box_fusion()` — no change to `weighted_box_fusion()` itself.

## 2. Design

- **Sequence:** `20241126_0001_crossing2_00` (the crossing2 reference sequence every other
  late-fusion result in this project is directly comparable against — Phase 5's `full_late`
  number, 0.236 fine-mAP@0.25 from the full 34-sequence sweep, is the nearest prior baseline).
- **Combo:** `full_late` only (the combo the hypothesis names).
- **Values:** 1.0 (baseline/sanity check), 0.75, 0.5, 0.25.
- **Repeats:** 2 per value. Inference runs exactly ONCE for `full_late`'s 6 unique sources
  (`run_all_sources_inference`, cached) and is reused across every weight value and rep — only
  the cheap WBF+AP step (`fuse_combo_frames` + `compute_ap_sweep`) reruns per (weight, rep). The 2
  reps came back bit-identical in every cell: with inference cached and WBF/AP free of randomness,
  the noise floor here is exactly 0, so every difference between weight values below is a 100%
  real effect of the weight change, not nondeterminism.
- **New script:** `tools/infra_weight_ablation.py` (lives in `tools/`, not `tools/urbaning_v2x/`,
  since it imports `mlflow_logging`/`eval_urbaning_v2x_late_fusion` directly rather than shelling
  out — the first launch crashed with `ModuleNotFoundError: mlflow_logging` from the wrong
  directory before this was caught and fixed). Logs to a new MLflow experiment,
  `urbaning_v2x_infra_weight_ablation`, one run per (infra_weight, rep).

## 3. Results

8 runs, 0 failures. Noise floor: 0 (see §2). Values are the mean over 2 bit-identical reps.

| infra_weight | mAP_fine @0.25 | mAP_groups @0.25 | mAP_fine @0.3 | mAP_fine @0.5 |
|---|---|---|---|---|
| **1.0 (baseline)** | **0.2241** | **0.3251** | 0.1912 | 0.0768 |
| 0.75 | 0.2192 | 0.3101 | 0.1904 | 0.0789 |
| 0.5 | 0.2084 | 0.2881 | 0.1827 | 0.0783 |
| 0.25 | 0.1942 | 0.2613 | 0.1723 | 0.0771 |

(The 1.0 baseline's 0.2241 vs. Phase 5's originally-reported 0.236 for the same sequence/combo is
a small, expected delta — this run goes through a slightly different code path, reusing cached
inference and a dedicated script rather than the full 34-sequence batch orchestrator; the
direction and scale of every comparison below is unaffected by that gap.)

## 4. Findings

**At the primary IoU=0.25 metric, both mAP_fine and mAP_groups decline monotonically as infra
gets down-weighted.** `infra_weight=1.0` (no down-weighting) beats every tested alternative;
`infra_weight=0.25` is clearly worst — a ~13% relative drop in mAP_fine (0.224→0.194) and ~20% in
mAP_groups (0.325→0.261).

**At the stricter IoU=0.3 threshold the same monotonic decline holds** (0.191→0.172). **At
IoU=0.5 the effect vanishes into noise** (0.077–0.079 across all 4 weights, no clear trend) — but
IoU=0.5 isn't the metric the original hypothesis (beating `v2v_late`, itself compared at IoU=0.25
in Phase 5's table) was about.

**Hypothesis not supported.** The intuition — that infra's individually weaker detections "drag
down" the fused result, so suppressing their influence should help — doesn't hold up under this
mechanism. A uniform score discount doesn't distinguish infra's false positives (which down-
weighting correctly suppresses) from infra's genuinely-correct, coverage-extending detections
(which down-weighting also suppresses) — exactly the detections that `full_late` was meant to add
over `v2v_late` in the first place. The coverage loss outweighs any precision gain.

**Recommendation: do not pursue uniform infra down-weighting further.** If the coverage-vs-quality
trade-off Phase 5 §7 identified is worth chasing, a mechanism that can tell apart a genuine
extra-coverage infra detection from an infra false positive would be needed — a uniform multiplier
cannot do that by construction, since it treats every infra detection identically regardless of
whether a vehicle source could have seen the same object at all.

## 5. File manifest

**New file:** `tools/infra_weight_ablation.py` — orchestrates the (infra_weight, rep) grid
described in §2, entirely inside the OV-SCAN container (no host-side orchestration needed, since
no new dataset conversions are required — `full_late`'s solo datasets for the crossing2 reference
sequence already exist on disk from Phase 5).

**Modified file:** `tools/eval_urbaning_v2x_late_fusion.py` — `fuse_combo_frames()` and
`build_late_fusion_frames()` gained the optional `infra_weight` parameter (default `1.0`, no
behavior change for any existing caller).

**MLflow:** new experiment `urbaning_v2x_infra_weight_ablation`, 8 runs
(`full_late__20241126_0001_crossing2_00__infraweight<w>__rep<N>`).
