# Fine-tuning OV-SCAN on UrbanIng-V2X — Phase 7

**Status:** First fine-tuning experiment in this project's history (every prior phase used
`pretrained/ov_scan_lidar.pth` strictly zero-shot). Built the training infrastructure this project
never needed before — a real train/val split, a usable `train.py` invocation, several blocking
bug fixes — and ran an 8-epoch fine-tune of `v2i_vehicle1` on crossing1+crossing2, evaluated on
held-out crossing3. **Mixed, genuinely interesting result**: group-level mAP improved substantially
across every IoU threshold (+38% to +97%), fine-class mAP stayed roughly flat with a notable
localization-quality improvement at strict IoU, but with a real regression in `van` precision
(explosion in false-positive count) worth investigating further before calling this a clean win.

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

**Working hypothesis for the `van` regression:** the alignment/CLIP classification head stayed
completely frozen throughout fine-tuning (by design — `align_weight=0.0`), but the *backbone and
box-regression/heatmap layers feeding into it* did change. The frozen text-alignment head was
originally trained jointly with a specific feature distribution; fine-tuning only the
upstream layers shifts that distribution without the alignment head adapting to compensate,
plausibly explaining why zero-shot classification behavior (which proposals get labeled "van" vs
"car") became less reliable for at least one class even as underlying box quality improved. Not
confirmed — would need per-proposal inspection to verify this specific mechanism.

## 6. Conclusion and recommendation

**Not a clean win, but a real, informative first result** — and a successful validation of the
training infrastructure itself, which is the primary deliverable of this phase (every bug found
and fixed in §2 was previously completely unexercised in this project). The group-level
improvement is substantial and likely the more practically relevant metric (an urban-intersection
safety system cares more about "is this a vehicle/pedestrian/two-wheeler" than fine-grained
car-vs-van distinction) — but the `van` regression needs understanding before this checkpoint
should be treated as a straightforward improvement over zero-shot.

**Natural next steps**, not attempted in this phase:
- Investigate the `van` false-positive explosion directly (which proposals are being
  mislabeled, and whether a smaller `cls_weight`/`bbox_weight` or more epochs at a lower LR
  changes this).
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
- `tools/cfgs/dataset_configs/urbaning_v2x_v2i_vehicle1_multiseq_dataset.yaml` — dataset config
  pointing at the merged multi-sequence dataset

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
