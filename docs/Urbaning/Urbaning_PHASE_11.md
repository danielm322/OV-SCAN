# Attempting to fix margin collapse and van-hallucination volume — Phase 11 (negative result)

**Status:** Built and trained two independent, targeted fixes for Phase 10's three open problems
(`van` p90-score regression, fine-grained `truck` AP drop, persistent `van` hallucination rate) —
a same-bucket pseudo-labeled hard-negative alignment loss, and a per-class cap on the fixed
200-proposal budget. **Neither lever, alone or combined, fixes any of the three targeted symptoms,
and both partially undo Phase 10's `matched_car%`/`van`/`car` AP gains. Recommendation: do not
adopt any Phase 11 checkpoint — Phase 10's `checkpoint_epoch_8.pth` remains the project default.**

---

## 1. Context

Phase 10 gave `dense_head.alignment_head` its first real gradient on UrbanIng-V2X and fixed the
Phase 8 §2 CLIP car/van-mislabeling mechanism (`matched_car%` 11.2%→6.1%), but left three open
problems:

1. `van` p90 score regressed (0.506→0.155) — Phase 9's unusually well-calibrated `van` confidence
   was lost.
2. Fine-grained `truck` AP dropped (0.616→0.267), though group-level `heavy_vehicle` AP stayed flat
   (0.480→0.477) — a reallocation within the heavy_vehicle bucket, not necessarily a real loss.
3. `van` hallucination rate stayed at ~83-86% unmatched, unchanged since Phase 7 — every phase has
   only dampened absolute volume, never the rate.

Code-tracing (`ov_scan_head.py`) attributed these to two independent, previously undiagnosed
mechanisms:

- **Mechanism A (margin collapse):** the AP-ranking score is the heatmap confidence rescaled by
  `sigmoid(ov_margin / 0.3)`, where `ov_margin` is the top1-vs-runner-up gap among CLIP logits
  reachable from a proposal's bucket. `loss()`'s alignment branch trains `CosineEmbeddingLoss` with
  **only positive pairs** — nothing pushes a query's embedding away from other classes sharing its
  bucket (`car`/`van`; `truck`/`bus`/`trailer`/`other_vehicle`). Plausible cause of both #1 (smaller
  margins → scores collapse toward 0.5) and #2 (noisier argmax within the heavy_vehicle bucket).
- **Mechanism B (uncapped fixed budget):** `predict()` selects a global top-200 via one flat argsort
  over all class channels, with no per-channel cap and no pre-classification score floor — the
  `car` channel (which also covers `van`) can flood the budget with low-confidence noise that still
  gets a full CLIP forward pass and label. Named but never implemented since Phase 8 §1.

## 2. Design

### Lever 1 — same-bucket pseudo-labeled cross-entropy (targets #1, #2)

The plan's original design called for a per-class memory bank of real crop embeddings to sample
hard negatives keyed by fine-grained class. **Pivoted during implementation**: `gt_grounding_sam`
entries carry no fine-grained native-class field, and the only class signal available in `loss()`
is the coarse 10-way bucket id (shared by `car`/`van`) — implementing the memory-bank design would
have required regenerating Phase 10's entire crop-embedding dataset just to add a class-name field.

Instead, for each matched positive, derive a pseudo fine-label for the REAL crop embedding via
CLIP's own pretrained zero-shot classification (restricted to the GT's bucket vocab via
`heatmap_to_vocab_mask`), then train the model's predicted embedding against that pseudo-label with
ordinary softmax cross-entropy, restricted to the same bucket. New `align_neg_weight` loss weight
(0.0 default = exact Phase 10 behavior; swept at 0.5 here).

### Lever 2 — per-class proposal cap (targets #3)

New opt-in `DENSE_HEAD.MAX_PROPOSALS_PER_CLASS: {'car': 100}`, applied in `predict()` right before
the global top-200 argsort: for each configured channel, zero out all but the top-`cap` entries of
that channel's post-NMS heatmap slice.

**Bug found and fixed during smoke-testing:** the first implementation thresholded on the k-th
largest *value* (`channel < kth_largest`), which large flat/background BEV regions defeat via tied
activations — up to 118 survivors observed with `cap=100`. Fixed by selecting by *index*
(`torch.topk().indices` + `scatter_`), guaranteeing exactly `k` survivors regardless of ties.

## 3. Training

All three variants continued from Phase 9's checkpoint with the same `--train_alignment_only`,
8-epoch, `--max_ckpt_save_num 3` recipe as Phase 10:

- (b) `align_negloss` — Lever 1 only (`align_neg_weight: 0.5`, no cap).
- (c) `align_capprop` — Lever 2 only (`MAX_PROPOSALS_PER_CLASS: {'car': 100}`, no neg loss).
- (d) `align_combined` — both together.

Each run's `train.py` process exits nonzero *after* training and checkpointing finish — a
pre-existing, config-independent bug in `nuscenes_ov_dataset.py`'s `evaluation()` (constructs
`dataroot/version/version`, doubling the version segment) that also affected Phase 10 and is
unrelated to this phase's changes; harmless since this project's own eval tooling is used instead
(§5).

**Checkpoint-diff discipline:** all three checkpoints diffed against Phase 9's base show exactly 29
changed parameter tensors (28 + `global_step`), all under `dense_head.alignment_head.*` — identical
pattern to Phase 10, confirming no leakage into the frozen backbone/heatmap/prediction-head.

## 4. Evaluation

Both of this project's standard tools, full 1400-frame held-out set, compared against Phase 9 (a)
and Phase 10:

### 4.1 Per-class AP @0.25

| Metric | Phase 9 (a) | Phase 10 align | (b) negloss | (c) capprop | (d) combined |
|---|---|---|---|---|---|
| `van` AP | 0.330 | **0.463** | 0.365 | 0.427 | 0.356 |
| `car` AP | 0.397 | **0.669** | 0.595 | 0.651 | 0.598 |
| `truck` AP | 0.616 | 0.267 | 0.267 | 0.275 | 0.274 |
| `cyclist` AP | 0.717 | 0.719 | 0.695 | 0.713 | 0.691 |
| mAP (fine classes) | 0.263 | **0.270** | 0.248 | 0.264 | 0.248 |
| mAP (groups) | 0.576 | 0.579 | 0.579 | 0.579 | 0.578 |
| `heavy_vehicle` AP | 0.480 | 0.477 | 0.480 | 0.485 | 0.480 |
| `vehicle` AP | 0.782 | 0.798 | 0.796 | 0.797 | 0.794 |

### 4.2 Cross-class `van` diagnostic (decisive metric)

Run in two passes — once under each variant's own inference config, so the proposal-cap config
also applies to the Phase 9/10 baseline checkpoints in that pass (a clean ceteris-paribus read of
the cap's effect on its own, independent of which checkpoint's weights are loaded):

| Variant | total `van` preds | unmatched% | matched_car% | matched_van% | p90 score |
|---|---|---|---|---|---|
| Phase 9 (no-cap cfg) | 104,009 | 83.8% | 11.2% | 3.9% | 0.506 |
| Phase 10 (no-cap cfg) | 51,506 | 86.3% | 6.2% | 5.9% | 0.153 |
| (b) negloss | 61,002 | 85.7% | **8.7%** | 4.7% | 0.180 |
| Phase 9 (cap=100 cfg) | 99,634 | 83.1% | 11.7% | 4.1% | 0.527 |
| Phase 10 (cap=100 cfg) | 49,792 | 85.9% | 6.3% | 6.1% | 0.178 |
| (c) capprop | 53,918 | 86.2% | 6.7% | 5.5% | 0.163 |
| (d) combined | 65,140 | 86.4% | **8.0%** | 4.5% | 0.143 |

## 5. Why each lever failed

**Lever 1 (negloss) made things worse, not better.** The pseudo-label used as the cross-entropy
target is CLIP's own zero-shot classification of the *real* crop embedding — but that's exactly the
classification step Phase 8 §2 showed is frequently wrong on this car/van boundary (68.5% of
high-confidence `van` predictions were real cars, pre-Phase-10). Training against a pseudo-label
that is itself wrong a meaningful fraction of the time plausibly pushes the model in the wrong
direction on ambiguous cases, which is consistent with `matched_car%` moving the *wrong* way
(6.2%→8.7%) and `van`/`car` AP both dropping. The margin-collapse hypothesis (Mechanism A) was not
disproven, but this particular fix for it — supervision from a noisy pseudo-label — is not viable
as implemented.

**Lever 2 (capprop) barely changes anything, because the cap rarely binds at 100.** Applying
`cap=100` to the *existing* Phase 9/10 checkpoints (the clean ceteris-paribus rows above) only cut
`van` prediction volume ~3-4% (104,009→99,634; 51,506→49,792) and left `unmatched%` within noise.
Average car+van GT is ~24/frame, but the `car` channel's total (car+van) proposal count per frame is
typically well under 100 even before any cap — the 100 cap was set conservatively (per the
mid-plan clarification) specifically to stay "well above real object density," which means it
mostly doesn't engage. The hallucination volume Phase 7 identified isn't concentrated in a small
number of frames blowing past 100; it's spread more evenly below that line, so a hard count cap at
this threshold doesn't touch most of it. A much tighter cap might bind more often, but there is no
evidence here that volume reduction would translate into *rate* reduction (`unmatched%`) rather than
just removing a mix of both hallucinated and genuine low-confidence detections proportionally.

**Combined (d) inherits negloss's harm** without capprop's near-null effect compensating for it —
worst `van`/`car` AP of the three variants, matching the independent-lever ablation design's intent
(isolate contributions) by showing the combination is dominated by the worse lever.

## 6. Conclusion

**Do not adopt any Phase 11 checkpoint.** Phase 10's
`output/.../ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align/v2i_vehicle1_finetune_align_v1/ckpt/checkpoint_epoch_8.pth`
remains the project default — it is strictly better than every Phase 11 variant on `van` AP, `car`
AP, and fine-class mAP, and no Phase 11 variant meaningfully improved `truck` AP, `unmatched%`, or
p90 score over it.

**What this rules out:** a same-bucket hard-negative term supervised by CLIP's own zero-shot
pseudo-labels is not a viable fix for the margin-collapse hypothesis — it needs a *trustworthy*
negative signal, which this pseudo-label is not. A per-class proposal cap at `car: 100` is not tight
enough to meaningfully move hallucination volume, let alone rate.

**Possible future directions, not attempted here:**
- A genuine negative signal for Lever 1 would need ground-truth-grade labels, not CLIP's own
  zero-shot guess on the same ambiguous boundary — e.g. real grounding-SAM detections per class, or
  human/dataset-native fine labels threaded through `gt_grounding_sam` (the latter was the original
  plan's memory-bank design, blocked by missing per-box metadata, not by concept).
- A direct CLIP logit-scale/temperature recalibration on top of Phase 10's checkpoint (noted as
  open in Phase 10 §6) targets the p90-score symptom without touching the classification boundary at
  all — untried in either phase.
- Lever 2 would need a much tighter cap (well below 100) swept empirically, and should be evaluated
  for whether it trades volume for rate at all, or just scales both down proportionally.

## 7. File manifest

**Modified:**
- `pcdet/models/dense_heads/ov_scan_head.py` — added `MAX_PROPOSALS_PER_CLASS` (Lever 2, `__init__`
  + `predict()`) and `align_neg_weight`/same-bucket pseudo-labeled cross-entropy term (Lever 1,
  `__init__` + `loss()`). Both opt-in and no-op by default; every prior config (including Phase
  9/10's) is unaffected.

**New configs:**
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align_negloss.yaml` —
  variant (b).
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align_capprop.yaml` —
  variant (c).
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align_combined.yaml` —
  variant (d).

**New checkpoints (not adopted):**
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align_negloss/v2i_vehicle1_finetune_align_negloss_v1/ckpt/checkpoint_epoch_8.pth`
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align_capprop/v2i_vehicle1_finetune_align_capprop_v1/ckpt/checkpoint_epoch_8.pth`
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align_combined/v2i_vehicle1_finetune_align_combined_v1/ckpt/checkpoint_epoch_8.pth`

**Result snapshots** (repo root inside the container, `/OV-SCAN/OV-SCAN/`):
`phase11_results_negloss.json`, `phase11_results_capprop.json`, `phase11_results_combined.json`.

**Recommended default (unchanged from Phase 10):**
`output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align/v2i_vehicle1_finetune_align_v1/ckpt/checkpoint_epoch_8.pth`.
