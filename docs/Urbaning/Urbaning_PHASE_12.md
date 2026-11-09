# Architectural-modification ablation study — Phase 12

**Status:** Tested two independent architectural levers neither training-regime phase (7-11)
could reach — reallocating the shared 10-channel detection heatmap into a 7-channel (A1) and
9-channel (A2) taxonomy with a dedicated `car`/`van` split, and a structural per-class
proposal-budget floor at inference time (B1 uniform / B2 floor+remainder / B3 inverse-GT-frequency)
— plus a retraining-free audit of how much of the residual gap is even addressable without
widening the model's frozen ±54m detection range (Lever C). **All three levers are negative
results. A1 cuts `van` `unmatched%` from 86.4% to 64.9% (the first lever in the whole project to
move this number), but at the cost of a NEW, larger failure mode: real cars are now mislabeled
`van` by the heatmap itself 19.5% of the time (vs. 6.1% under Phase 10's CLIP-level version of the
same confusion) — splitting the channel exposed that the heatmap regressor, under this fine-tune
budget, discriminates `car`/`van` shape less reliably than frozen CLIP's visual prior did. A2
underperforms A1 on every axis. Lever B (MIN_PROPOSALS_PER_CLASS) delivers no clean win on any
architecture: its recommended default (B2) is close to flat vs. no-floor, and its more aggressive
policies (B1/B3) merely convert "hallucination in empty space" into "confident mislabeling of a
real car" — a worse failure mode, not a better number. Lever C's audit finds ~70% of ALL missed
GT boxes, on every checkpoint tested, already lie outside the model's ±54m range — a structural
ceiling no head-level lever can touch.** Recommendation: do not adopt A1, A2, or any Lever B
policy. Phase 10's `checkpoint_epoch_8.pth` remains the project default. A future full,
unfrozen-backbone retrain with a widened detection range is the one lever in this project's entire
history with a quantified, substantial addressable upside (§5.3) — scoped but not attempted here.

---

## 1. Context

Phases 7-11 exhausted every training-regime lever (LR, backbone freeze/unfreeze, loss weights, a
dedicated alignment-head fine-tune, a hard-negative loss term, a post-hoc proposal cap) without
fixing the project's most persistent failure: `van` predictions are ~83-86% unmatched
hallucinations (unchanged since Phase 7), and real cars get confidently mislabeled `van` by a
frozen-CLIP disambiguation step operating on proposals from a **single shared heatmap channel**
(Phase 8 §2: 68.5% of high-confidence `van` predictions, pre-Phase-10, were real cars). Phase 11's
doc explicitly concluded this needs a different kind of fix than anything tried so far.

Code-tracing `ov_scan_head.py` plus an audit of UrbanIng-V2X's actual label distribution converged
on a structural explanation: the 10-channel classification heatmap, inherited unchanged from
nuScenes pretraining, is badly misallocated for this dataset. `car` (num_gt=28,614) and `van`
(num_gt=5,068, a 5.6:1 imbalance) share one heatmap channel, with CLIP disambiguating them
post-hoc from that channel's shared proposal pool. `truck`/`bus`/`trailer` (3 separate channels)
all route to the same 4-way vocabulary (redundant capacity). `construction_vehicle`/`traffic_cone`
(2 channels) are permanently zero-GT dead weight. No training-regime lever can fix a channel-count
and -allocation problem.

**Mid-plan correction, before any implementation started:** a review of whether UrbanIng-V2X's
`other_*` catch-all native categories (`other_pedestrian`, and the converter's general-purpose
`OtherVehicle`) deserve dedicated model capacity found they're genuinely ambiguous/low-GT
(`other_pedestrian`=263, `other_vehicle`=0 in the held-out split) and visually indistinguishable
from their sibling class for a LiDAR detector. Both designs below fold them into a sibling's
vocabulary (zero extra capacity cost) rather than giving them a dedicated channel or dropping them
entirely (today's status quo: `other_pedestrian` is structurally unscored, AP=0.000 in every
Phase 9-11 result). This turned out to matter a great deal, in an unanticipated way (§5.4).

This phase tests two independent, previously-untried architectural levers, plus a cheap scoping
check on a third, much more expensive lever (widened detection range) that is explicitly deferred.

## 2. Design

### Lever A — heatmap channel reallocation (A1 conservative / A2 full redesign)

**A1 (7 channels):** retires the 2 permanently-dead channels (`construction_vehicle`,
`traffic_cone`) and the semantically-unused `barrier` channel, using the freed capacity to split
`car`/`van` and `cyclist`/minor-two-wheelers into separate channels, while consolidating the
redundant 3-channel heavy-vehicle group down to 2 (`truck`, `heavy_other`).

| Channel | Vocab |
|---|---|
| `car` | `["car"]` |
| `van` | `["van"]` |
| `truck` | `["truck"]` |
| `heavy_other` | `["bus", "trailer", "other vehicle"]` |
| `cyclist` | `["cyclist"]` |
| `two_wheeler_minor` | `["motorcycle", "e-scooter"]` |
| `pedestrian` | `["pedestrian", "other pedestrian"]` |

**A2 (9 channels):** one channel per genuinely distinguishable canonical fine class — car, van,
truck, bus, trailer, cyclist, motorcycle, escooter, pedestrian — removing the bucket+CLIP
disambiguation mechanism almost entirely. `other_vehicle`/`other_pedestrian` still have no
dedicated channel (folded into `trailer`'s and `pedestrian`'s vocab respectively).

Both variants: `RETRAIN_HEAD: True` (reinitializes only the shape-mismatched final heatmap
conv/`class_encoding`/`prediction_head.heatmap`; everything else, including the entire
`alignment_head`, transfers from Phase 10's checkpoint), `FULL_RES_NMS_CLASSES: ['pedestrian']`,
`--freeze_backbone`, continued from Phase 10's `checkpoint_epoch_8.pth`. A1: 12 epochs. A2: 24
epochs (longer, to give the near-scratch low-GT channels more chance to converge).

**Required code changes** (generalize two previously-hardcoded nuScenes/Waymo special cases so
any taxonomy works, both opt-in/no-op by default):
- `ov_scan_head.py`: `FULL_RES_NMS_CLASSES` config list, resolved to indices by class name in
  `__init__`, replacing the old hardcoded index-8/9 (nuScenes) and index-1/2 (Waymo) branches in
  `predict()`.
- `nuscenes_utils.py`'s `map_name_from_general_to_detection`: added `vehicle.van`,
  `vehicle.heavy_other`, `vehicle.two_wheeler_minor`, `vehicle.cyclist`, `vehicle.bus`,
  `vehicle.escooter`, `other.ignore` — purely additive, harmless for real nuScenes data.
- New dataset variants built by `/tmp/build_taxonomy_variant.py` (container-side, not committed):
  relabels UrbanIng-V2X's converted nuScenes-style info pkls under the A1/A2 category mappings,
  reusing Phase 10's crop-embedding alignment pkl unchanged (crops are keyed by box
  geometry/token, independent of class label). Train/val split: 5400/1400 samples, 46.3% crop
  coverage — identical to Phase 10's established baseline, confirming correctness.

### Lever B — structural per-class proposal-budget floor (inference-only, no retraining)

Replaces `predict()`'s single flat `argsort` over all channels with a selection that reserves a
guaranteed floor per class before the competitive remainder is filled — unlike Phase 11's
ceiling-only `MAX_PROPOSALS_PER_CLASS`, this can actually guarantee another channel's share of the
fixed 200-proposal budget, not just stop one channel from flooding it.

**Code change** (`ov_scan_head.py`'s `predict()`, same index-based `topk`+`scatter_` pattern as
Phase 11's cap, for the same tie-breaking reason): new opt-in `MIN_PROPOSALS_PER_CLASS:
{class: floor_k}`. Each configured class's `torch.topk` claims its floor directly; claimed cells
are zeroed out of a cloned remainder heatmap; the existing global argsort fills the remaining
`200 - Σfloor_k` slots competitively. Also added a `query_is_floor` per-proposal diagnostic,
threaded through `get_bboxes`/`decode_bbox` into `proposal_is_floor`, and a new
`--log_proposal_source` flag in `eval_urbaning_v2x_per_class.py` that tallies floor-vs-remainder
provenance per class and prints it — confirming the mechanism engages exactly as designed (§4)
before trusting any AP number built on top of it, mirroring Phase 11's own tie-threshold
verification discipline.

**Three floor policies**, swept against all 4 checkpoints (Phase 9, Phase 10, A1, A2), each using
that checkpoint's own taxonomy:

- **B1 (uniform):** `floor_k = num_proposals // num_classes` for every channel (baseline: 20;
  A1: 28; A2: 22) — including permanently-dead channels, deliberately, as a noise/upper-bound
  reference.
- **B2 (floor + remainder, the plan's recommended default):** A1: `car:40, van:25, truck:15,
  heavy_other:10, cyclist:25, two_wheeler_minor:10, pedestrian:25` (sum 150, 50 competitive). A2:
  `car:40, van:25, truck:15, bus:8, trailer:7, cyclist:20, motorcycle:8, escooter:7,
  pedestrian:20` (sum 150). Baseline (10-channel, no dedicated `van`): floors summed across
  whichever fine channels its coarser taxonomy merges (`car:40, truck:25, bicycle:35,
  pedestrian:25`), for a comparable ceteris-paribus reference against the old checkpoints.
- **B3 (pure inverse-GT-frequency):** floors ∝ 1/GT_count on the real held-out val set (computed
  directly from each variant's own converted pkl, not estimated), normalized to sum 200,
  zero-GT channels excluded. Deliberately extreme — e.g. A2's `motorcycle` (332 GT) gets
  floor_k=102/200 — a stress test for where protecting rare classes starts starving `car`/
  `pedestrian` below what they need.

### Lever C — scoping audit only (no retrain)

UrbanIng-V2X's GT boxes are frequently outside the model's frozen ±54m `POINT_CLOUD_RANGE`
(inherited from nuScenes pretraining); Phase 3 showed naively widening this zero-shot collapses
mAP, and a real fix needs a full unfrozen retrain (30-50 epochs) — an order of magnitude more
expensive than anything in Phases 7-11, explicitly out of scope here. New script
`leverc_range_audit.py`: for every GT box in the val set (not just the ones already inside range,
unlike every other eval tool in this project), checks whether a same-canonical-class prediction
matched it, then splits every miss into out-of-range (no proposal mechanism could ever reach it
without widening the range) vs. in-range (a genuine model failure). Quantifies how much headroom
is even addressable by Lever A/B before paying for a future retrain.

## 3. Training

Both variants continued from Phase 10's `checkpoint_epoch_8.pth`, `--freeze_backbone`,
`--max_ckpt_save_num 3`: A1 12 epochs (loss converged ~0.276), A2 24 epochs (loss converged
~0.259-0.260). Both runs' `train.py` process exited nonzero *after* training, checkpointing, and
a successful partial eval pass all completed — the same pre-existing, config-independent
`repeat_eval_ckpt`/NuScenes-devkit crash documented in Phase 10/11, unrelated to this phase.

**Checkpoint-diff discipline:** both final checkpoints diffed against Phase 10's base show exactly
the expected pattern — 5 shape-mismatched tensors (`class_encoding`/`heatmap_head.1`/
`prediction_head.heatmap.1`, reinitialized for the new class count), 116 value-changed tensors all
under `dense_head.*` (28 `alignment_head` + 87 other dense-head tensors + `global_step`), **0**
frozen-backbone (`vfe`/`backbone_3d`/`map_to_bev_module`/`backbone_2d`) tensors touched. Identical
for A1 (10→7 classes) and A2 (10→9 classes).

## 4. Evaluation

All runs on the full 1400-frame held-out set, `tools/eval_urbaning_v2x_per_class.py` +
`investigate_cross_class_fps.py --class_name van`. Phase 9/Phase 10 no-floor numbers are reused
directly from their own docs (identical checkpoint, identical eval protocol); everything else
(14 AP-table runs + 11 van-diagnostic runs + 3 Lever-C runs) is new.

**`--log_proposal_source` sanity check** (run against A2/B2 on a 5-frame slice before trusting any
full-scale number): every configured class's floor was claimed exactly as configured
(`floor_share=1.000` for `van`/`truck`/`bus`/`trailer`/`cyclist`/`motorcycle` — meaning on that
slice, 100% of their surviving predictions came from the guaranteed floor, zero from the
competitive remainder; `car`/`pedestrian` showed `floor_share<1` — meaning they *also* won
additional slots beyond their floor). Mechanism confirmed working as designed.

### 4.1 Lever A alone (no floor) vs. Phase 9/10 baseline

| Metric | Phase 9 (a) | Phase 10 (adopted) | A1 (7ch) | A2 (9ch) |
|---|---|---|---|---|
| `van` AP @0.25 | 0.330 | 0.463 | 0.403 | 0.259 |
| `car` AP @0.25 | 0.397 | 0.669 | **0.729** | 0.718 |
| `truck` AP @0.25 | 0.616 | 0.267 | 0.478 | **0.635** |
| `cyclist` AP @0.25 | 0.717 | **0.719** | 0.721 | 0.691 |
| mAP (fine classes) | 0.263 | **0.270** | 0.259 | 0.257 |
| mAP (groups) | 0.576 | **0.579** | 0.542 | 0.554 |
| `heavy_vehicle` (group) AP | 0.480 | 0.477 | 0.360 | **0.455** |
| `vehicle` (group) AP | 0.782 | 0.798 | **0.799** | 0.798 |

**Cross-class `van` diagnostic (decisive metric)** — each checkpoint under its own architecture's
plain config, no floor:

| Variant | total `van` preds | unmatched% | matched_car% | matched_van% | p90 score |
|---|---|---|---|---|---|
| Phase 9 (a) lowlr | 104,073 | 83.8% | 11.2% | 3.9% | 0.506 |
| Phase 10 align | 51,478 | 86.4% | 6.1% | 5.9% | 0.155 |
| **A1** | 28,910 | **64.9%** | **19.5%** | 12.9% | 0.0009 |
| A2 | 34,338 | 83.9% | 6.4% | 7.9% | 0.0009 |

A1 is the first lever in the project's entire history to move `unmatched%` meaningfully below the
83-86% band stuck since Phase 7 (86.4%→64.9%). But `matched_car%` — the metric Phase 10 spent its
whole phase improving (11.2%→6.1%) — gets **three times worse** under A1 (6.1%→19.5%). Because A1
constrains each heatmap channel to an exclusive, single-item CLIP vocab
(`CONSTRAIN_TO_HEATMAP_BUCKET` + 1-item buckets), a `van`-labeled prediction under A1 is
*guaranteed* to have come from the dedicated `van` heatmap channel — there is no CLIP relabeling
left to blame. `matched_car%=19.5%` is therefore a direct, CLIP-free measurement of the heatmap
regressor itself misplacing a `van`-channel proposal on a real car's location. **Splitting the
channel exposed that the geometry-only signal (12 epochs, frozen backbone, from-scratch final
conv) discriminates `car`/`van` shape less reliably than frozen CLIP's visual crop embedding did**
— the opposite of the plan's working hypothesis. A2 stays much closer to Phase 10's behavior on
this metric (`matched_car%=6.4%`) but barely moves `unmatched%` (83.9%, within noise of baseline),
consistent with its capacity being spread thinner across 9 channels (several near-zero-GT) over
twice A1's training budget.

**Score calibration collapsed under both A1 and A2** (p90 ≈0.0009, vs. Phase 10's already-regressed
0.155 and Phase 9's 0.506) — the `van` channel's heatmap activation is near-uncalibrated after this
short, frozen-backbone retrain, on top of the existing `van`-vocab CLIP margin-rescaling. Neither
variant is deployable on raw score alone without further calibration work.

### 4.2 Lever B sweep (B1 / B2 / B3), all 4 checkpoints

**AP table** (van AP / car AP / mAP fine / mAP groups; no-floor baseline for Phase 9/10 reused
from §4.1):

| Config | van AP | car AP | mAP (fine) | mAP (groups) |
|---|---|---|---|---|
| Phase 9, no-floor | 0.330 | 0.397 | 0.263 | 0.576 |
| Phase 9, B1 | 0.306 | 0.336 | 0.257 | 0.552 |
| Phase 9, B2 | 0.330 | 0.390 | 0.263 | 0.575 |
| Phase 9, B3 | 0.142 | 0.112 | 0.213 | 0.443 |
| Phase 10, no-floor | 0.463 | 0.669 | 0.270 | 0.579 |
| Phase 10, B1 | 0.423 | 0.596 | 0.259 | 0.555 |
| Phase 10, B2 | 0.466 | 0.676 | 0.271 | 0.580 |
| Phase 10, B3 | 0.219 | 0.219 | 0.192 | 0.444 |
| A1, no-floor | 0.403 | 0.729 | 0.259 | 0.542 |
| A1, B1 | 0.409 | 0.707 | 0.258 | 0.539 |
| A1, B2 | 0.405 | 0.727 | 0.259 | 0.542 |
| A1, B3 | 0.355 | 0.095 | 0.182 | 0.394 |
| A2, no-floor | 0.259 | 0.718 | 0.257 | 0.554 |
| A2, B1 | 0.266 | 0.672 | 0.252 | 0.544 |
| A2, B2 | 0.265 | 0.714 | 0.258 | 0.556 |
| A2, B3 | 0.203 | 0.048 | 0.177 | 0.374 |

**B2 (the recommended default) is within noise of no-floor on every architecture** (van AP moves
at most +0.003 to +0.010; car AP moves ≤0.015). B1's uniform floor is mildly net-negative
everywhere (gives up car AP for no van AP gain). B3's extreme inverse-frequency floor is
**catastrophic for `car`** on every architecture that has a dedicated or shared car-containing
channel (0.397→0.112 Phase 9; 0.669→0.219 Phase 10; 0.729→0.095 A1; 0.718→0.048 A2) — starving
`car` of budget costs far more than any rare-class gain is worth.

**Van diagnostic under the floor policies** reveals *why* B2 looks flat and B1/B3 look "better" on
raw `unmatched%` but aren't real wins:

| Config | total van preds | unmatched% | matched_car% |
|---|---|---|---|
| Phase 9, B1 | 16,511 | 14.9% | **58.3%** |
| Phase 9, B3 | 5,101 | 1.2% | **68.0%** |
| Phase 10, B1 | 6,350 | 21.7% | **28.0%** |
| Phase 10, B3 | 1,688 | 1.8% | **21.6%** |
| A1, B3 | 14,000 | 51.5% | 24.0% |
| A2, B1 | 30,800 | 80.3% | 8.1% |

Under the old shared-channel taxonomy (Phase 9/10), shrinking the `car`-channel's total budget
(B1/B3) doesn't remove hallucination — it removes the *low-confidence* hallucination and leaves
behind a smaller pool of *high-confidence* proposals, most of which sit on a real, correctly
located car that CLIP still mislabels `van`. `unmatched%` crashing to 1-22% looks like a huge win
in isolation; `matched_car%` simultaneously exploding to 22-68% shows it's the same overconfidence
problem changing which failure mode it shows up as, not disappearing. This is precisely why the
plan tracks `unmatched%` and `matched_car%` together as the decisive pair, not either alone.

### 4.3 Lever C: range-coverage audit

| Checkpoint | Total GT | In-range matched | In-range missed | Out-of-range | Out-of-range share of ALL misses |
|---|---|---|---|---|---|
| Phase 10 | 81,666 | 28,891 | 15,922 | 36,853 | **69.8%** |
| A1 | 81,666 | 30,381 | 14,432 | 36,853 | **71.9%** |
| A2 | 81,666 | 29,228 | 15,585 | 36,853 | **70.3%** |

`total_gt` and `out_of_range` are identical across all three rows by construction (they depend
only on the dataset's raw GT boxes and the shared ±54m range, not on predictions) — a useful
internal sanity check that the audit is measuring what it claims to.

**~70% of every checkpoint's missed ground truth is already outside the model's frozen detection
range**, regardless of which Lever-A architecture is loaded. No head-level lever (A or B) can ever
touch this majority of the gap; only a future full retrain with a widened range can. Per-class
breakdown for `van` specifically: Phase 10 `out_of_range_share_of_all_misses`=60.6%, A1=70.0%,
A2=57.0% — consistent with A1 genuinely reducing *in-range* `van` misses (1,338 vs. Phase 10's
2,024) even though its overall `van` AP is lower (driven by the new car-mislabeling problem in
§4.1, not by missed detections). **`motorcycle` and `escooter` have zero in-range matches on
every single checkpoint tested** (Phase 10, A1, A2 alike) — these two classes are never correctly
detected at all, in or out of range, regardless of architecture; their channels are informative
negative results, not usable detectors. This quantifies that a future range-widening full retrain
has real, substantial, checkpoint-independent addressable upside — the single most promising
unexplored lever this project has identified, but out of scope for this phase.

### 4.4 An unplanned finding: CLIP's zero-shot vocab-pair sensitivity

A1/A2's fine-grained `pedestrian` AP collapses (0.311 under Phase 10 → 0.003 under A1, 0.006 under
A2) while the group-level `pedestrian_group` AP stays flat (0.298/0.289/0.294) — not a detection
regression, but a labeling artifact. Phase 9/10's `pedestrian` bucket vocab was a single-item list
(`["pedestrian"]`); the mid-plan correction's `other_pedestrian`-folding design (§1) made it
two-item (`["pedestrian", "other pedestrian"]`) under A1/A2. The predicted-label volume shows CLIP's
zero-shot preference, when given that exact two-string choice, overwhelmingly favors "other
pedestrian" regardless of the underlying crop — `other_pedestrian`'s prediction volume explodes to
76,651 (A1) / 90,869 (A2), almost entirely real pedestrians relabeled, not a genuine improvement
in recognizing the genuinely-rare `other_pedestrian` native category (true `other_pedestrian` AP
stays ≈0, now for a different reason: swamped by FPs instead of never predicted at all). The same
heatmap boxes, same detections — just a different string attached by CLIP's zero-shot argmax.
**Caveat for any future per-class vocab design in this project: adding a second, semantically-close
candidate string to an existing single-item CLIP bucket can flip the entire predicted-label
distribution, independent of visual content** — this was invisible in every prior phase because no
bucket vocab had ever grown past one item. Group-level metrics are unaffected and remain the
reliable number to read for any bucket with this kind of near-synonym pair.

### 4.5 Why Lever B's floor doesn't rescue genuinely under-converged channels

A1's merged `heavy_other` channel (vocab `["bus","trailer","other vehicle"]`, 1,317 combined GT)
has **zero** surviving predictions for any of its three fine labels, under both no-floor *and*
B2's 10-proposals/frame floor. The floor only guarantees that `topk`-selected cells get pushed
through the full query-decode-and-CLIP-classify pipeline; it can't force those cells to carry a
genuine local heatmap maximum. `predict()`'s NMS step (`heatmap = heatmap * (heatmap == local_max)`)
zeroes every non-peak cell *before* the floor's `topk` ever runs — if a channel's post-NMS heatmap
has fewer true peaks than its configured floor (or none at all, for a channel this undertrained in
only 12 epochs), the floor's extra slots are filled with exact-zero cells that the final
`score > 0` filter then discards. **Lever B fixes budget crowding (too many other channels'
proposals squeezing a class out of the top-200), not under-training (a channel having nothing
confident to contribute at all)** — a second, independent reason, alongside Phase 11's own finding
that a 100-cap "rarely binds," that no proposal-selection lever can be a durable substitute for
giving a channel enough real training signal to converge in the first place.

## 5. Conclusion

**Do not adopt A1, A2, or any Lever B floor policy.** Phase 10's
`output/.../ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align/v2i_vehicle1_finetune_align_v1/ckpt/checkpoint_epoch_8.pth`
remains the project default.

**Lever A (channel reallocation):** partially validates and partially refutes the plan's working
hypothesis. It genuinely reduces raw `van` hallucination volume and rate (`unmatched%`
86.4%→64.9% under A1) by giving `van` dedicated capacity — but doing so removes CLIP's
disambiguation step entirely for that channel, and the heatmap regressor alone (under this
fine-tune budget) turns out to confuse `car`/`van` shape *more* than CLIP's visual prior did
(`matched_car%` 6.1%→19.5%). A1 (7-channel) clearly outperforms A2 (9-channel) on every decisive
metric despite training for only half as many epochs (12 vs. 24) — spreading capacity across more
near-empty channels (A2's `bus` channel: 0 GT in this dataset, confirmed by direct inspection of
the converted pkl) is
a net loss, not a finer-grained improvement. Score calibration collapses under both (p90 van score
≈0.0009), ruling out either as deployable without a dedicated recalibration pass.

**Lever B (proposal floor):** does not fix anything. Its recommended default (B2) is statistically
flat vs. no-floor on every architecture. Its more aggressive policies (B1/B3) *look* like dramatic
wins on `unmatched%` alone, but §4.2's paired diagnostic shows this is the exact same
`car`→`van` overconfidence mechanism changing which number it shows up in (`matched_car%` exploding
to 22-68% under the old shared-channel taxonomy) rather than shrinking. §4.5 identifies a second,
independent reason the floor can't rescue the genuinely rare channels it was built to protect:
guaranteeing a proposal *slot* can't substitute for a channel having actually converged.

**Lever C (scoping audit):** the one unambiguous, actionable finding of this phase. ~70% of every
tested checkpoint's missed ground truth already lies outside the current ±54m detection range,
regardless of architecture — this majority of the project's remaining detection gap is
structurally unreachable by any head-level lever Phases 7-12 have tried or could try. A future
full, unfrozen-backbone retrain with a widened point-cloud range is the single most promising
untried lever in this project's history, with a now-quantified, checkpoint-independent addressable
upside — scoped here, not attempted, given its cost (an order of magnitude more than any phase so
far).

**What this rules out:** channel-count reallocation alone, without fixing the underlying
geometric car/van discrimination capability, is not a viable path to reducing `van` hallucination
below Phase 10's level. A structural proposal floor is not a substitute for either (a) more
training signal for genuinely rare classes or (b) removing the car/van overconfidence mechanism
itself — it can only redistribute an already-fixed error budget. Any future CLIP-vocab redesign in
this project must account for near-synonym pairs within one bucket silently dominating the
predicted-label distribution (§4.4), independent of detection quality.

**Possible future directions, not attempted here:**
- The full range-widening retrain scoped by Lever C §4.3 — now with concrete, checkpoint-verified
  evidence of its addressable upside, rather than the untested assumption Phase 3 left behind.
- A `car`/`van`-specific geometric feature (e.g. a dedicated small regression head keyed on
  length/width ratio, which differs systematically between the two classes even when LiDAR point
  density is similar) might recover A1's heatmap-level discrimination without reverting to CLIP's
  worse-calibrated post-hoc fix — untried here.
- Any future bucket vocab change should be smoke-tested for the §4.4 zero-shot-preference effect
  (compare raw predicted-label volume before/after adding a candidate string, on a handful of
  frames, before trusting fine-grained AP) before committing to a full evaluation pass.

## 6. File manifest

**Modified:**
- `pcdet/models/dense_heads/ov_scan_head.py` — `FULL_RES_NMS_CLASSES` (generalizes the hardcoded
  nuScenes/Waymo full-res-NMS special case), `MIN_PROPOSALS_PER_CLASS` (Lever B floor+remainder),
  `query_is_floor`/`proposal_is_floor` diagnostic threaded through `predict()` →`get_bboxes()` →
  `decode_bbox()`. All opt-in/no-op by default; every prior config (including Phase 9/10's) is
  unaffected.
- `pcdet/datasets/nuscenes/nuscenes_utils.py` — 7 new `map_name_from_general_to_detection` entries
  for A1/A2's new general-category strings; purely additive.
- `tools/eval_urbaning_v2x_per_class.py` — new `--log_proposal_source` diagnostic flag.

**New tool:**
- `tools/leverc_range_audit.py` — Lever C scoping audit (per-GT in-range-matched /
  in-range-missed / out-of-range split, no retraining).

**New configs:**
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_a1.yaml` /
  `_a2.yaml` — the two Lever-A taxonomies.
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_{baseline,a1,a2}_leverb_{b1,b2,b3}.yaml`
  — 9 eval-only Lever-B floor overlays (never used for training).

**New checkpoints (not adopted):**
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_a1/v2i_vehicle1_finetune_a1_v1/ckpt/checkpoint_epoch_12.pth`
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_a2/v2i_vehicle1_finetune_a2_v1/ckpt/checkpoint_epoch_24.pth`

**Result snapshots** (inside the container, not committed): `/tmp/phase12_leverb/ap_*.json` (14
AP tables), `/tmp/phase12_leverb/van_*.log` (11 van diagnostics), `/tmp/phase12_leverc/leverc_*.json`
(3 range audits).

**Recommended default (unchanged from Phase 10):**
`output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align/v2i_vehicle1_finetune_align_v1/ckpt/checkpoint_epoch_8.pth`.
