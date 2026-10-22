# Fixing the `van` false-positive problem: `hm_weight` tuning and a score-threshold fix — Phase 8

**Status:** Two attempts to fix the `van` proposal-hallucination problem Phase 7 §5.1 diagnosed.
**Halving `hm_weight` (§1) was ruled out** — no real change to the over-firing behavior, and a
net-negative trade against group-level mAP. **A non-training deployment score threshold (§2) is a
genuine, free win for every class except `van`** — but investigating *why* `van` alone resists
thresholding surfaced a second, distinct mechanism (confident CLIP mislabeling of real cars)
coexisting with the first (low-confidence hallucination), neither of which the threshold or
`hm_weight` can fix on its own.

---

## Context

Phase 7 built the first fine-tuning pipeline for OV-SCAN on UrbanIng-V2X and ran an 8-epoch
early-fusion (`v2i_vehicle1`) fine-tune, improving group-level mAP substantially (+38% to +97%
across IoU thresholds) but causing `van` AP to drop (0.443→0.240) as its prediction count exploded
17x. Phase 7 §5.1 investigated this directly (cross-class IoU matching, not left as a guess) and
found the frozen-CLIP-boundary-shift hypothesis doesn't hold for the *aggregate* volume: 87% of the
new `van` predictions are pure hallucination (unmatched to any GT, median score ~0.0001), not
mislabeled real cars. The actual mechanism: fine-tuning made the heatmap's "car" bucket (the
channel both `car` and `van` predictions are CLIP-classified from) far less selective — its total
proposal volume grew ~4.7x — so a fixed 200-proposal-per-frame budget gets flooded with
low-information, non-object locations that the frozen CLIP step still classifies, apparently
defaulting to "van" on ambiguous query features.

Phase 7's doc named two candidate fixes for this; the user chose to test the cheaper,
more-directly-targeted one first: **lower `hm_weight` (currently 1.0, unchanged from the original
un-fine-tuned recipe) to reduce how hard the heatmap loss pushes on the dominant `car` class**,
while checking it doesn't give up the group-level/pedestrian/cyclist gains Phase 7 achieved.

## 1. Testing the `hm_weight` fix (2026-10-15)

Phase 7 §5.1's recommended next step was to test whether a smaller `hm_weight` reins in the
car-channel proposal-selectivity loss without giving up the group-level gains. Tested directly: new
config `ov_scan_lidar_urbaning_v2i_vehicle1_finetune_hmw0p5.yaml`, identical to the baseline
fine-tune config in every field except `hm_weight: 0.5` (half of the baseline's `1.0`), trained for
the same 8 epochs on the same crossing1+crossing2 split, evaluated on the identical 1400-frame
held-out crossing3 set with both tools from Phase 7 §5/§5.1.

| | Zero-shot | Fine-tuned `hm_weight=1.0` | Fine-tuned `hm_weight=0.5` |
|---|---|---|---|
| mAP_groups @0.25 | 0.422 | 0.584 | 0.569 |
| mAP_groups @0.5 | 0.223 | 0.440 | 0.430 |
| mAP_fine @0.25 | 0.250 | 0.245 | 0.243 |
| mAP_fine @0.5 | 0.141 | 0.190 | 0.192 |
| `van` AP @0.25 | 0.443 | 0.240 | **0.213** |
| `van` total predictions (1400 frames) | 8,566 | 145,819 | 112,805 |
| `van` unmatched (hallucination) % | 36.5% | 87.0% | **85.5%** |
| `van` median prediction score | 0.044 | 0.0001 | 0.0002 |
| `car` AP @0.25 | 0.571 | 0.369 | 0.421 |

**Halving `hm_weight` does not fix the problem, and is a net-negative trade.** Prediction volume
dropped ~23% (145,819→112,805) but the unmatched/hallucination fraction barely moved (87.0%→85.5%)
and the median confidence is still essentially zero (0.0001→0.0002) — the proposal-selectivity
problem identified in Phase 7 §5.1 is still fully present, just slightly smaller in absolute volume.
Worse, group-level mAP *regressed* slightly at every threshold (mAP_groups @0.25: 0.584→0.569;
@0.5: 0.440→0.430) — exactly the gains §5.1 wanted to protect. `van` AP itself also got slightly
worse, not better (0.240→0.213), since AP is sensitive to the *relative* rank of the small amount
of genuine signal against the (still enormous) noise floor, not just the noise floor's absolute
size. The one clear win is `car` AP, which improved (0.369→0.421) — consistent with a less
dominant heatmap loss leaving more "room" for the already-strong `car` signal, but this doesn't make
up for the lack of improvement everywhere else.

**Conclusion: `hm_weight` is not the right lever, at least not a simple halving of it.** The
mechanism from Phase 7 §5.1 (heatmap loss pushing hardest on the class with the most GT instances)
is real, but scaling its weight down just slows convergence toward the same qualitative over-firing
behavior rather than changing it — by epoch 8, the heatmap has still learned to fire broadly across
the car channel; it's not primarily a *loss-weight-magnitude* problem. This points toward the
*other* named candidate (a lower peak LR with more epochs, addressing training-dynamics/overfitting
rather than the static loss weighting — tested in Phase 9) as the more promising remaining lever, or
a different kind of fix entirely — e.g. a deployment-side score threshold (Phase 7 §5.1 already
noted ~0.3 filters out ~92% of the noise), or an architectural cap on how many proposals one heatmap
channel can contribute per frame — rather than continuing to tune `hm_weight` itself.

## 2. Non-training score-threshold exploration (2026-10-15)

§1 above and Phase 7 §5.1 both treat the fine-tuned checkpoint's prediction set as fixed and look
for either a root-cause explanation or a retraining fix. A third, much cheaper option needs no
retraining at all: filter predictions by confidence at inference/deployment time. One fact makes
this worth checking directly rather than assuming: `OVScanHead.decode_bbox()`'s own score filter
reads `DENSE_HEAD.POST_PROCESSING.SCORE_THRESH`, which every config in this project sets to **0.0**
(not the different, unused-by-this-path `0.1` in the top-level `MODEL.POST_PROCESSING` block) — so
every frame always returns all `NUM_PROPOSALS=200` raw proposals regardless of confidence, and
every AP number in Phase 7 §5/§5.1 and §1 above is computed over the *full* PR curve, which is
threshold-invariant by construction. AP can never show what choosing an actual deployment cutoff
would do; only precision/recall *at* a specific threshold can. Built
`tools/sweep_score_threshold.py` to measure exactly that, on the `hm_weight=1.0` checkpoint (the
best group-level performer), swept over `[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]` at IoU≥0.25,
full 1400-frame held-out set:

| score ≥ | car P / R | truck P / R | cyclist P / R | pedestrian P / R | **van P / R** | vehicle (group) P / R |
|---|---|---|---|---|---|---|
| 0.0 | 0.429 / 0.378 | 0.178 / 0.816 | 0.112 / 0.792 | 0.041 / 0.573 | 0.031 / 0.891 | 0.165 / 0.840 |
| 0.1 | 0.967 / 0.321 | 0.853 / 0.323 | 0.692 / 0.678 | 0.550 / 0.373 | 0.262 / 0.733 | 0.950 / 0.667 |
| 0.2 | 0.978 / 0.304 | 0.898 / 0.263 | 0.744 / 0.659 | 0.664 / 0.335 | 0.271 / 0.675 | 0.958 / 0.612 |
| 0.3 | 0.984 / 0.285 | 0.933 / 0.231 | 0.784 / 0.645 | 0.747 / 0.302 | 0.273 / 0.622 | 0.962 / 0.566 |
| 0.5 | 0.993 / 0.211 | 0.947 / 0.179 | 0.831 / 0.602 | 0.851 / 0.243 | 0.266 / 0.399 | 0.970 / 0.394 |
| 0.7 | 0.997 / 0.080 | 1.000 / 0.066 | 0.899 / 0.495 | 0.907 / 0.170 | 0.295 / 0.183 | 0.968 / 0.156 |

**A score threshold is a clear, free win for every class except `van`.** `car`/`truck`/`cyclist`/
`pedestrian` precision all climb cleanly and substantially as the threshold rises (`car`:
0.429→0.997; `pedestrian`: 0.041→0.907), confirming that for those classes the bulk of the
low-confidence volume really is disposable noise, exactly the mechanism Phase 7 §5.1 described. A
threshold around 0.2–0.3 looks like a good general-purpose operating point: strong precision across
the board (car ≥0.98, truck ≥0.90, cyclist ≥0.74, pedestrian ≥0.66, vehicle-group ≥0.96) while
keeping recall reasonable (car ~0.29–0.30, cyclist ~0.65, vehicle-group ~0.57–0.61).

**But `van` precision never rises above ~0.27–0.30, at *any* threshold up to 0.7** — flat, while
every other class's precision keeps climbing. That is the signature of a different failure mode
than the one driving the bulk volume: thresholding removes low-confidence noise, but something
else is keeping `van` wrong even among its most confident predictions. Checked directly: reran
`investigate_cross_class_fps.py` (extended with a `--min_score` filter) restricted to `van`
predictions with score ≥0.5, on the same checkpoint:

| | all `van` preds (unfiltered) | `van` preds with score ≥0.5 |
|---|---|---|
| total | 145,819 | 7,594 |
| `unmatched` (hallucination) | 87.0% | **0.8%** |
| `matched_car` (right place, wrong label) | 8.9% | **68.5%** |
| `matched_van` (correct) | 3.1% | 26.6% |
| median score | 0.0001 | 0.67 |

**This reopens, and now actually confirms, the original Phase 7 §5.1 "frozen CLIP boundary shift"
hypothesis — just for a different population than first assumed.** Phase 7 §5.1 disproved it as the
explanation for the *aggregate* volume (dominated by near-zero-confidence hallucination). But
restricted to confident, well-localized detections, the picture flips entirely: 68.5% of
high-confidence `van` predictions are real, correctly-localized `car`s that CLIP confidently labels
"van" instead — almost no hallucination at all (0.8%) in this slice. Two genuinely distinct
mechanisms coexist in the same checkpoint's `van` output: a low-confidence proposal-selectivity
flood (Phase 7 §5.1, fixable in principle by reining in heatmap over-firing) and a separate,
high-confidence classification-boundary error on real cars (not touched by either Phase 7 §5.1's
heatmap story or §1 above's `hm_weight` experiment, and not removable by any score threshold since
the threshold doesn't distinguish "confidently right" from "confidently wrong").

**Practical recommendation:** adopt a deployment score threshold around 0.2–0.3 — it is a free,
no-retraining precision win for every class/group *except* `van`, including the practically more
relevant `vehicle` group (car+van combined), which already reaches ≥0.96 precision at this
threshold specifically *because* group-level scoring absorbs the car/van confusion rather than
being hurt by it. For any use case that needs the fine-grained `car`/`van` distinction specifically
(not just "is this a vehicle"), neither a score threshold nor the `hm_weight` change (§1) fixes
it — that would require addressing the CLIP alignment step's car/van decision boundary directly
(e.g. prompt/vocabulary changes, or training the alignment head after all), a different kind of fix
than anything tried in this phase.

## 3. Conclusion and recommendation

The `van` regression from Phase 7 turned out to have **two distinct causes**: a low-confidence
proposal-selectivity flood (Phase 7 §5.1) and a separate high-confidence classification error on
real cars (§2 above) — not one problem with one fix. §1 ruled out `hm_weight` reduction as a fix for
the first cause (it doesn't help, and costs a small amount of group-level mAP). §2 found that a
deployment score threshold (~0.2–0.3) is a genuine free win for every class/group except `van`
specifically, including the more practically relevant `vehicle` group — but cannot fix `van` itself,
since its high-confidence errors are confidently-wrong labels, not noise.

**Recommended default at the time of this phase:** deploy the `hm_weight=1.0` fine-tuned checkpoint
(Phase 7 §4/§5) with a score threshold of ~0.2–0.3 (§2) — the single highest-value, zero-cost change
available at this point, a straightforward config change (`DENSE_HEAD.POST_PROCESSING.SCORE_THRESH`)
rather than a code or training change. **Superseded by Phase 9**, which found a better base
checkpoint (lower LR + more epochs) that this same threshold recommendation still applies to.

**Natural next steps**, not attempted in this phase:
- If the fine-grained `car`/`van` distinction specifically matters for a use case: address the CLIP
  alignment step's car/van decision boundary directly (§2) — e.g. prompt/vocabulary tuning, or
  reconsidering `align_weight: 0.0`'s "never train the alignment head" decision from Phase 7's
  original scope — rather than more heatmap/loss-weight tuning, which §1/§2 showed doesn't touch
  this specific error mode. **Still not attempted as of Phase 9.**
- The remaining named candidate from Phase 7 §5.1/§1 above for the *low-confidence* flood
  specifically: a lower peak LR with more epochs, to test whether the heatmap over-firing is a
  training-dynamics/convergence effect rather than a static loss-weighting one. **Tested in Phase
  9** (along with a freeze-backbone alternative) — lower LR adopted as the new default.

## 4. File manifest

**New files:**
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_hmw0p5.yaml` — §1's
  `hm_weight=0.5` variant of Phase 7's fine-tune config (ruled out — see §1)
- `tools/sweep_score_threshold.py` — §2's precision/recall-at-threshold sweep tool (distinct from
  AP, which is threshold-invariant by construction)

**Modified files:**
- `tools/investigate_cross_class_fps.py` (originally created in Phase 7 §5.1) — added a
  `--min_score` filter, used in §2 to isolate the high-confidence `van` slice

**Checkpoints:** `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_hmw0p5/
v2i_vehicle1_finetune_hmw0p5/ckpt/checkpoint_epoch_8.pth` — ruled out, kept for reference.
