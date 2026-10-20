# Fine-tuning OV-SCAN on UrbanIng-V2X — Phase 7

**Status:** First fine-tuning experiment in this project's history (every prior phase used
`pretrained/ov_scan_lidar.pth` strictly zero-shot). Built the training infrastructure this project
never needed before — a real train/val split, a usable `train.py` invocation, several blocking
bug fixes — and ran an 8-epoch fine-tune of `v2i_vehicle1` on crossing1+crossing2, evaluated on
held-out crossing3. **Mixed, now well-understood result**: group-level mAP improved substantially
across every IoU threshold (+38% to +97%), fine-class mAP stayed roughly flat with a notable
localization-quality improvement at strict IoU, alongside a real regression in `van` precision
(explosion in false-positive count) that turned out to have two distinct causes (§5.1, §5.3), one
of which (§5.3) is fixable today with a free, no-retraining deployment score threshold.

---

## 1. Motivation

Two independent findings pointed at the same gap. Phase 3 §8: widening early fusion's shared
window (needed since `v2i_vehicle1` fuses infra sensors up to 135m away, outside the current ±54m
box) collapsed mAP 0.28→0.007 zero-shot, and concluded this needs retraining, not a config change.
Phase 6 §6: intermediate (BEV-feature) fusion failed even in the cleanly-correct case because the
frozen head was only ever exposed to single-source features during training — also a training
problem, not an inference-time design problem. This phase builds the training infrastructure
needed to eventually attack either, validated on the simpler, already-proven-architecture case
first: fine-tune `v2i_vehicle1` as-is (still ±54m, no widened range yet) and confirm it's at least
not harmful before attempting anything riskier.

**Scope decisions (confirmed with the user before implementing):** target early fusion
(`v2i_vehicle1`) first, not intermediate fusion. Loss scope: box-regression + heatmap only
(`align_weight: 0.0`) — the CLIP zero-shot classification path is never trained, only
localization. Full fine-tune (no backbone freeze) for this first attempt. Split: train on
crossing1 (17 sequences) + crossing2 (10 sequences) = 27 sequences; hold out crossing3 (7
sequences) entirely. Scale: a quick validation run (8 epochs), not the original 20-epoch recipe.

## 2. Building the training infrastructure (none of this existed before this phase)

Reading `tools/train.py`'s actual code path (not just its `--help`) surfaced five blocking issues,
none of which had ever been hit because `train.py` had never been run in this project:

1. **No `OPTIMIZATION` config block.** `train.py:68` reads `cfg.OPTIMIZATION.get('USE_AMP', ...)`
   unconditionally, before any CLI-arg branching — the very first thing that touches it. Fixed by
   copying the original `ov_scan_lidar.yaml`'s block (the only config in this project that had
   one) into a new, dedicated fine-tuning config (§3).
2. **No `DATA_AUGMENTOR` block in any urbaning dataset config.** `DatasetTemplate.__init__`
   accesses `self.dataset_cfg.DATA_AUGMENTOR` unconditionally whenever `training=True`. Fixed with
   an empty `{DISABLE_AUG_LIST: [], AUG_CONFIG_LIST: []}` block — safe for the existing zero-shot
   eval path too, since eval never sets `training=True`.
3. **No `gt_grounding_sam` field in the built info pkls.** `NuScenesOVDataset.__getitem__`
   requires this key whenever `self.training=True`; UrbanIng-V2X has no SAM-grounding data at all.
   Fixed by injecting `[None] * num_boxes` per record after building the pkls — a path the
   existing loss code already tolerates safely (degenerates to a zero-dummy loss term), doubly
   inert here since `align_weight=0.0` anyway.
4. **`OVScanHead.forward()`'s training branch unconditionally reads `batch_dict['cam_ids']`/
   `['raw_imgs']`.** UrbanIng-V2X has no `CAMERA_CONFIG`, so these keys are never populated.
   Confirmed by reading `loss()` in full that `images_dict` (built from these two values) is never
   actually referenced in the loss math — fixed with a `.get(key, <empty default>)` guard, a
   provably safe two-line change.
5. **`wandb` (0.18.0) fails to import against this container's `protobuf` (6.33.6)** — the
   protobuf version was deliberately upgraded earlier in this project for `mlflow`, and the
   resulting wandb/protobuf incompatibility was noted as a "pip-check warning" at the time but
   never actually exercised until this phase tried to use `train.py`'s wandb integration.
   Downgrading protobuf risks breaking mlflow (used throughout every other phase); fixed instead
   by making `wandb` fully optional in `train_utils_wandb.py` (try/except import, every call site
   guarded) — training works identically with or without it, just without wandb logging when it's
   unavailable.

A sixth issue surfaced empirically, not from reading code: **`NuScenesOVDataset.__init__` resolves
`self.root_path` as `DATA_PATH/VERSION`, and loads point clouds from `self.root_path /
info['lidar_path']`** — but `info['lidar_path']` is computed (in `nuscenes_utils.
fill_trainval_infos`) relative to the *unversioned* `DATA_PATH`, matching where
`convert_to_nuscenes.py` and `create_nuscenes_info`'s own `NuScenes(dataroot=...)` call need the
files to be *during the build step*. Every single-sequence dataset this project had used until now
happened to have the point clouds duplicated in both locations (confirmed directly — a historical
artifact from an earlier converter code version, not something anyone did deliberately), masking
this latent inconsistency. The new multi-sequence dataset, built cleanly, only had the one
(build-step-correct) copy — surfacing the bug as a `FileNotFoundError` on the first real training
run. Fixed by moving `samples/` under the version directory as a step after `create_nuscenes_info`
runs (a move, not a duplicate copy — nothing else needs the unversioned location afterward).

A **separate, unrelated, and ultimately harmless** bug was also hit: `pcdet`'s own automatic
post-training evaluation (`tools/test.py`'s `repeat_eval_ckpt`, triggered by `train.py` after every
run) calls `NuScenes(dataroot=str(self.root_path), version=VERSION)` at
`nuscenes_ov_dataset.py:487` — since `self.root_path` is *already* `DATA_PATH/VERSION`, this
doubles the version suffix and crashes with `Database version not found: .../v1.0-custom/
v1.0-custom`. This only affects `pcdet`'s own generic kitti-style recall eval, which this project
has never used for real evaluation (Phase 3 §9 already established that its `recall_dict` output
is consulted by nothing) — training and checkpoint-saving both complete successfully *before* this
runs, confirmed directly by inspecting the checkpoint directory. Not fixed, since nothing in this
project's actual evaluation path depends on it; noted here so a future run isn't alarmed by the
traceback at the very end of the log.

## 3. Real train/val split by intersection

`create_nuscenes_info` (`pcdet/datasets/nuscenes/nuscenes_dataset.py`) hardcoded `train_scenes =
[]` / `val_scenes = available_scene_names` for `version == 'v1.0-custom'`, with no way to override
it — every sample in every prior phase landed in val by construction. Added an optional
`train_scene_names` parameter: when given, partitions `available_scene_names` into a real
train/val split instead of the all-to-val default (backward-compatible — every existing call site
uses keyword args, and the official-nuScenes branches are untouched).

**New script, `tools/urbaning_v2x/build_v2i_vehicle1_multiseq_dataset.py`**, builds one combined
multi-sequence dataset spanning all 34 sequences with this real split baked in. Needed because
`convert_to_nuscenes.py`'s `write_tables()` always overwrites its whole output directory from one
converter instance's own in-memory state — there's no built-in multi-sequence append. Merging is
safe as plain list concatenation because every token is `uuid.uuid4().hex` (globally unique) and
every point-cloud filename is sequence-prefixed (confirmed in code) — no remapping needed.
Storage-efficient like `run_full_dataset_eval.py`: only one sequence's raw archive + converted
output is ever on disk at a time; the merged output (point-cloud files + concatenated JSON tables)
grows across the run and is the only thing that persists.

Validated incrementally before committing to the full 34-sequence run: tested on one train-
intersection sequence alone (correctly landed 200/200 samples in train, 0 in val), then added one
val-intersection sequence (correctly rebalanced to 200/200 train/val) — confirming both branches of
the split logic before trusting it at scale.

**Full build result:** 34/34 sequences, 0 failures, **6800 total samples** (34 × 200) split into
**5400 train** (27 sequences, crossing1+crossing2) / **1400 held-out val** (7 sequences, crossing3)
— exactly matching the planned split.

## 4. Training

```
WANDB_MODE=offline python train.py \
    --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune.yaml \
    --pretrained_model ../../pretrained/ov_scan_lidar.pth \
    --batch_size 1 --epochs 8 --extra_tag v2i_vehicle1_finetune_v1 --num_epochs_to_eval 0
```
`--pretrained_model` (not `--ckpt`): loads only `model_state` and starts a genuinely new
fine-tuning run from the frozen checkpoint's weights, rather than resuming a prior run's optimizer
state. New dedicated config,
`tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune.yaml` (and a matching
dataset config pointing at the merged dataset) — kept separate from the existing zero-shot eval
config so every prior phase's exact reproducibility is untouched.

**Smoke-tested first** (1 epoch) before committing to the full run — this is what caught the
wandb/protobuf and samples-path bugs above. Full run: all 8 epochs completed, checkpoints saved
every epoch. Training loss decreased steadily and consistently: **0.78 (epoch 1 avg) → 0.46 (epoch
3) → 0.26 (epoch 6) → 0.19 (epoch 8)** — real, healthy learning, not a stalled or diverging run.

## 5. Evaluation — held-out crossing3, 1400 frames

Both runs use the exact same eval methodology every other phase uses
(`eval_urbaning_v2x_per_class.py`, `class_ap`/`compute_ap_sweep`, IoU 0.25/0.3/0.5), on the *exact
same* 1400 held-out crossing3 samples (the merged dataset's own val split) — the zero-shot number
here is a fresh run on this precise held-out set, not reused from Phase 3's aggregate table, for a
rigorous apples-to-apples comparison differing only in which checkpoint is loaded.

| Metric | IoU | Zero-shot | Fine-tuned (epoch 8) | Δ |
|---|---|---|---|---|
| mAP (fine) | 0.25 | 0.250 | 0.245 | −2% |
| mAP (groups) | 0.25 | 0.422 | **0.584** | **+38%** |
| mAP (fine) | 0.3 | 0.236 | 0.239 | +1% |
| mAP (groups) | 0.3 | 0.397 | **0.569** | **+43%** |
| mAP (fine) | 0.5 | 0.141 | **0.190** | **+35%** |
| mAP (groups) | 0.5 | 0.223 | **0.440** | **+97%** |

**Group-level mAP improved substantially at every IoU threshold** (+38% to +97%) — the largest
single result of this phase. **Fine-class mAP stayed roughly flat at the lenient threshold but
improved meaningfully at stricter IoU** (+35% at 0.5), suggesting fine-tuning genuinely improved
box localization quality (tighter, better-fit boxes survive stricter IoU matching more often) even
where raw mAP@0.25 didn't move much.

**Per-class breakdown reveals a real mixed picture, not a clean win:**

| Class | Zero-shot AP (num_pred) | Fine-tuned AP (num_pred) |
|---|---|---|
| car | 0.571 (27,475) | 0.369 (25,255) |
| cyclist | 0.393 (10,157) | **0.683** (18,218) |
| pedestrian | 0.236 (143,981) | **0.376** (83,010) |
| truck | 0.608 (16,475) | 0.537 (5,593) |
| **van** | 0.443 (8,601) | **0.240 (145,706)** |
| motorcycle | 0.000 (4,844) | 0.000 (0) |
| trailer | 0.000 (10,075) | 0.000 (0) |

Pedestrian and cyclist (the hardest, most safety-critical classes) improved substantially. `car`
and `van` AP both dropped — most strikingly `van`, whose prediction count *exploded* from 8,601 to
145,706 (a 17x increase) while its AP roughly halved, meaning fine-tuning induced a large new
false-positive problem specifically for this class. `motorcycle` and `trailer` dropped to zero
predictions entirely (from zero-shot's own already-near-zero AP despite thousands of predictions)
— arguably a precision improvement (stopped firing nonsense predictions for these) but also a
recall loss (if any real instances existed).

**Investigated directly in §5.1 below** (not left as a guess) — the actual mechanism turned out to
be different from, and more specific than, the first hypothesis this section originally proposed.

### 5.1 Investigating the `van` explosion directly (2026-10-15)

The first hypothesis (written right after seeing the aggregate numbers, before investigating
further) was: the frozen CLIP alignment head, trained jointly with a specific feature
distribution, stayed frozen while fine-tuning shifted the *upstream* feature distribution feeding
into it — so real cars increasingly get mislabeled "van" by a decision boundary that never adapted.
This is a plausible-sounding story, but it's a guess, not a finding — so it was tested directly
rather than left standing.

**Method:** for every `van`-labeled prediction from both checkpoints, found its best-IoU match
against *every* GT box regardless of class (not just `van` GT), and classified each prediction as
`matched_van` (correct), `matched_car` (right place, wrong label — exactly what the hypothesis
predicts), `matched_other`, or `unmatched` (no GT anywhere near it — a genuine hallucination, not
a mislabel). Run on the full 1400-frame held-out crossing3 set for both checkpoints.

| | Zero-shot (8,566 van preds) | Fine-tuned (145,916 van preds) |
|---|---|---|
| `matched_van` (correct) | 36.5% | 3.1% |
| `matched_car` (right place, wrong label) | 14.1% | 8.9% |
| `matched_other` | 12.9% | 1.1% |
| **`unmatched` (hallucination)** | 36.5% | **87.0%** |
| median prediction score | 0.044 | **0.0001** |
| fraction with score > 0.3 | 2.3% | 7.9% |

**The hypothesis doesn't hold up.** `matched_car` (the category it specifically predicts) grew
only ~10.7x (1,210→12,922) — roughly in line with, if anything slightly *below*, the overall 17x
growth in van predictions. If the frozen-boundary/shifted-distribution story were the dominant
effect, `matched_car` should have grown disproportionately *faster* than the hallucination
category. It didn't — **87% of the new volume is pure hallucination**, boxes with no
spatial correspondence to any real object, at a median confidence score of essentially zero.

**What actually happened instead:** fine-tuning made the heatmap's "car" bucket (the one `van` is
classified from — `HEATMAP_BUCKET_TO_VOCAB_KEYS`/`NUSCENES_TO_OV_CLASSES` route every car-bucket
proposal to a CLIP choice between "car" and "van" specifically) far less selective — its total
proposal volume (car+van combined) grew ~4.7x (36,076→170,961 across the 1400 frames), consistent
with the heatmap loss pushing hardest on whichever class has the most GT instances in this
dataset (almost certainly `car`, at an intersection). With a fixed 200-proposal-per-frame budget
and a much more trigger-happy car-channel heatmap, far more of those 200 slots get filled by
low-information, non-object locations than before. The frozen CLIP step still runs on every one of
them regardless of whether they correspond to anything real, and for these noisy/ambiguous
query features, it apparently defaults to "van" over "car" more often than not. Since AP is
rank-based over the *entire* precision-recall curve with no score cutoff (the same `class_ap`
methodology every phase in this project uses), this flood of near-zero-confidence noise directly
dilutes `van`'s precision at every recall level, regardless of how unlikely a real deployment
would be to act on a 0.0001-confidence detection — a score threshold around 0.3 would already
filter out ~92% of it.

**Revised conclusion:** this is a real side effect of fine-tuning worth knowing about before
reusing this checkpoint, but it's a *proposal-selectivity* problem (the heatmap over-firing on the
car channel), not a *classification-boundary* problem (CLIP mislabeling real cars) — a
meaningfully different, more specific, and more actionable finding than the original guess. It
also means raw full-PR-curve AP somewhat overstates the practical severity for `van` specifically:
a deployment using any reasonable confidence threshold would see far less of this effect than the
mAP number alone suggests.

### 5.2 Testing the `hm_weight` fix (2026-10-15)

§5.1's recommended next step was to test whether a smaller `hm_weight` reins in the car-channel
proposal-selectivity loss without giving up the group-level gains. Tested directly: new config
`ov_scan_lidar_urbaning_v2i_vehicle1_finetune_hmw0p5.yaml`, identical to the baseline fine-tune
config in every field except `hm_weight: 0.5` (half of the baseline's `1.0`), trained for the same
8 epochs on the same crossing1+crossing2 split, evaluated on the identical 1400-frame held-out
crossing3 set with both tools from §5/§5.1.

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
problem identified in §5.1 is still fully present, just slightly smaller in absolute volume. Worse,
group-level mAP *regressed* slightly at every threshold (mAP_groups @0.25: 0.584→0.569; @0.5:
0.440→0.430) — exactly the gains §5.1 wanted to protect. `van` AP itself also got slightly worse,
not better (0.240→0.213), since AP is sensitive to the *relative* rank of the small amount of
genuine signal against the (still enormous) noise floor, not just the noise floor's absolute size.
The one clear win is `car` AP, which improved (0.369→0.421) — consistent with a less dominant
heatmap loss leaving more "room" for the already-strong `car` signal, but this doesn't make up for
the lack of improvement everywhere else.

**Conclusion: `hm_weight` is not the right lever, at least not a simple halving of it.** The
mechanism from §5.1 (heatmap loss pushing hardest on the class with the most GT instances) is real,
but scaling its weight down just slows convergence toward the same qualitative over-firing
behavior rather than changing it — by epoch 8, the heatmap has still learned to fire broadly across
the car channel; it's not primarily a *loss-weight-magnitude* problem. This points toward the
*other* named candidate (a lower peak LR with more epochs, addressing training-dynamics/overfitting
rather than the static loss weighting) as the more promising remaining lever, or a different kind
of fix entirely — e.g. a deployment-side score threshold (§5.1 already noted ~0.3 filters out ~92%
of the noise), or an architectural cap on how many proposals one heatmap channel can contribute per
frame — rather than continuing to tune `hm_weight` itself.

### 5.3 Non-training score-threshold exploration (2026-10-15)

§5.1/§5.2 both treat the fine-tuned checkpoint's prediction set as fixed and look for either a
root-cause explanation or a retraining fix. A third, much cheaper option needs no retraining at
all: filter predictions by confidence at inference/deployment time. One fact makes this worth
checking directly rather than assuming: `OVScanHead.decode_bbox()`'s own score filter reads
`DENSE_HEAD.POST_PROCESSING.SCORE_THRESH`, which every config in this project sets to **0.0** (not
the different, unused-by-this-path `0.1` in the top-level `MODEL.POST_PROCESSING` block) — so every
frame always returns all `NUM_PROPOSALS=200` raw proposals regardless of confidence, and every AP
number in §5/§5.1/§5.2 is computed over the *full* PR curve, which is threshold-invariant by
construction. AP can never show what choosing an actual deployment cutoff would do; only
precision/recall *at* a specific threshold can. Built `tools/sweep_score_threshold.py` to measure
exactly that, on the `hm_weight=1.0` checkpoint (the best group-level performer), swept over
`[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]` at IoU≥0.25, full 1400-frame held-out set:

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
low-confidence volume really is disposable noise, exactly the mechanism §5.1 described. A
threshold around 0.2–0.3 looks like a good general-purpose operating point: strong precision across
the board (car ≥0.98, truck ≥0.90, cyclist ≥0.74, pedestrian ≥0.66, vehicle-group ≥0.96) while
keeping recall reasonable (car ~0.29–0.30, cyclist ~0.65, vehicle-group ~0.57–0.61).

**But `van` precision never rises above ~0.27–0.30, at *any* threshold up to 0.7** — flat, while
every other class's precision keeps climbing. That is the signature of a different failure mode
than the one driving the bulk volume: thresholding removes low-confidence noise, but something
else is keeping `van` wrong even among its most confident predictions. Checked directly: reran
`investigate_cross_class_fps.py` (extended with a `--min_score` filter) restricted to `van`
predictions with score ≥0.5, on the same checkpoint:

| | all `van` preds (§5.1, unfiltered) | `van` preds with score ≥0.5 |
|---|---|---|
| total | 145,819 | 7,594 |
| `unmatched` (hallucination) | 87.0% | **0.8%** |
| `matched_car` (right place, wrong label) | 8.9% | **68.5%** |
| `matched_van` (correct) | 3.1% | 26.6% |
| median score | 0.0001 | 0.67 |

**This reopens, and now actually confirms, the original §5.1 "frozen CLIP boundary shift"
hypothesis — just for a different population than first assumed.** §5.1 disproved it as the
explanation for the *aggregate* volume (dominated by near-zero-confidence hallucination). But
restricted to confident, well-localized detections, the picture flips entirely: 68.5% of
high-confidence `van` predictions are real, correctly-localized `car`s that CLIP confidently labels
"van" instead — almost no hallucination at all (0.8%) in this slice. Two genuinely distinct
mechanisms coexist in the same checkpoint's `van` output: a low-confidence proposal-selectivity
flood (§5.1/§5.2, fixable in principle by reining in heatmap over-firing) and a separate,
high-confidence classification-boundary error on real cars (not touched by either §5.1's heatmap
story or §5.2's `hm_weight` experiment, and not removable by any score threshold since the
threshold doesn't distinguish "confidently right" from "confidently wrong").

**Practical recommendation:** adopt a deployment score threshold around 0.2–0.3 — it is a free,
no-retraining precision win for every class/group *except* `van`, including the practically more
relevant `vehicle` group (car+van combined), which already reaches ≥0.96 precision at this
threshold specifically *because* group-level scoring absorbs the car/van confusion rather than
being hurt by it. For any use case that needs the fine-grained `car`/`van` distinction specifically
(not just "is this a vehicle"), neither a score threshold nor the `hm_weight` change (§5.2) fixes
it — that would require addressing the CLIP alignment step's car/van decision boundary directly
(e.g. prompt/vocabulary changes, or training the alignment head after all), a different kind of fix
than anything tried in this phase.

## 6. Conclusion and recommendation

**Not a clean win, but a real, informative first result** — and a successful validation of the
training infrastructure itself, which is the primary deliverable of this phase (every bug found
and fixed in §2 was previously completely unexercised in this project). The group-level
improvement is substantial and likely the more practically relevant metric (an urban-intersection
safety system cares more about "is this a vehicle/pedestrian/two-wheeler" than fine-grained
car-vs-van distinction). The `van` regression turned out to have **two distinct causes**: a
low-confidence proposal-selectivity flood (§5.1) and a separate high-confidence classification
error on real cars (§5.3) — not one problem with one fix. §5.2 ruled out `hm_weight` reduction as a
fix for the first cause (it doesn't help, and costs a small amount of group-level mAP). §5.3 found
that a deployment score threshold (~0.2–0.3) is a genuine free win for every class/group except
`van` specifically, including the more practically relevant `vehicle` group — but cannot fix `van`
itself, since its high-confidence errors are confidently-wrong labels, not noise.

**Natural next steps**, not attempted in this phase:
- **Recommended default going forward:** deploy the `hm_weight=1.0` fine-tuned checkpoint (§4/§5)
  with a score threshold of ~0.2–0.3 (§5.3) — the single highest-value, zero-cost change available
  right now, and a straightforward config change (`DENSE_HEAD.POST_PROCESSING.SCORE_THRESH`) rather
  than a code or training change.
- If the fine-grained `car`/`van` distinction specifically matters for a use case: address the CLIP
  alignment step's car/van decision boundary directly (§5.3) — e.g. prompt/vocabulary tuning, or
  reconsidering `align_weight: 0.0`'s "never train the alignment head" decision from this phase's
  original scope (§1) — rather than more heatmap/loss-weight tuning, which §5.2/§5.3 showed doesn't
  touch this specific error mode.
- The remaining named candidate from §5.1/§5.2 for the *low-confidence* flood specifically: a lower
  peak LR with more epochs, to test whether the heatmap over-firing is a training-dynamics/
  convergence effect rather than a static loss-weighting one. Lower priority now that §5.3 shows a
  threshold already cleans up that flood's practical impact for free.
- The original motivating experiment: retry Phase 3 §8's widened-`POINT_CLOUD_RANGE`
  `v2i_vehicle1` config, now training on the wider window instead of zero-shot, to see whether
  fine-tuning lifts the catastrophic 0.28→0.007 collapse that started this phase.
  Phase 6's intermediate-fusion training (teaching the head to tolerate blended multi-source BEV
  features) — now unblocked by having a working `train.py` pipeline for the first time.

## 7. File manifest

**New files:**
- `tools/urbaning_v2x/build_v2i_vehicle1_multiseq_dataset.py` — multi-sequence dataset builder (§3)
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune.yaml` — fine-tuning
  config (OPTIMIZATION block, `align_weight: 0.0`)
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_hmw0p5.yaml` — §5.2's
  `hm_weight=0.5` variant of the above (ruled out — see §5.2)
- `tools/cfgs/dataset_configs/urbaning_v2x_v2i_vehicle1_multiseq_dataset.yaml` — dataset config
  pointing at the merged multi-sequence dataset
- `tools/sweep_score_threshold.py` — §5.3's precision/recall-at-threshold sweep tool (distinct from
  AP, which is threshold-invariant by construction)
- `tools/investigate_cross_class_fps.py` — the cross-class IoU-matching diagnostic used in §5.1,
  §5.2, and (with the `--min_score` filter added in §5.3) §5.3, kept as a general reusable tool
  (any class/checkpoint/confidence-slice comparison, not just `van`)

**Modified files:**
- `pcdet/datasets/nuscenes/nuscenes_dataset.py` — `create_nuscenes_info` gained the optional
  `train_scene_names` parameter (§3)
- `pcdet/models/dense_heads/ov_scan_head.py` — `.get()`-guarded `cam_ids`/`raw_imgs` reads (§2.4)
- `tools/train_utils/train_utils_wandb.py` — `wandb` import/calls made fully optional (§2.5)
- `tools/cfgs/dataset_configs/urbaning_v2x_v2i_vehicle1_dataset.yaml` — added empty
  `DATA_AUGMENTOR` block (§2.2), harmless for the existing zero-shot eval path

**New datasets:** `datasets/urbaning_v2x_v2i_vehicle1_multiseq/` — the merged 34-sequence dataset
(5400 train / 1400 val samples), kept on disk for any future fine-tuning iteration on this config.

**Checkpoints:** `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune/
v2i_vehicle1_finetune_v1/ckpt/checkpoint_epoch_{1..8}.pth` — all 8 epochs' checkpoints kept.

**MLflow:** new experiment `urbaning_v2x_finetune_phase7`, 2 runs (zero-shot baseline, fine-tuned
epoch 8) both evaluated on the identical held-out crossing3 split.
