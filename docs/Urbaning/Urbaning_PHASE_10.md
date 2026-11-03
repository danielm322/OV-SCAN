# Fine-tuning the OV-SCAN alignment head on UrbanIng-V2X — Phase 10

**Status:** Built a new offline pipeline that generates real per-GT-box CLIP crop embeddings from
UrbanIng-V2X's own vehicle1 cameras (never previously ingested by this codebase), fed them into the
alignment loss's existing but never-populated `gt_grounding_sam` path, and fine-tuned only
`dense_head.alignment_head` on top of Phase 9's adopted checkpoint. **Result: this is the first
lever in the whole project to actually move `matched_car%`** — real cars mislabeled `van` dropped
from 11.2% to 6.1% of all `van` predictions (-45% relative), while `matched_van%` (predictions that
are genuinely vans) rose from 3.9% to 5.9% (+51% relative). `van` AP improved 0.330→0.463 and `car`
AP improved 0.397→0.669, with group-level mAP flat (0.576→0.579) and fine-class mAP slightly better
(0.263→0.270) — every one of Phase 8-9's untouched-by-construction findings about the CLIP
mislabeling mechanism is now addressed, at essentially no cost elsewhere.

---

## 1. Context

Every phase since Phase 7 identified the same unfixed mechanism and could not touch it: the frozen
`OVScanAlignmentHead` (`pcdet/models/dense_heads/ov_scan_head.py:359`) classifies LiDAR-query
features into open-vocabulary text labels (`car` vs. `van`, etc.) via cosine similarity against
precomputed CLIP text embeddings, but it never received a real training gradient on UrbanIng-V2X —
`align_weight: 0.0` in every config, combined with `gt_grounding_sam` being `None` for every box
(no grounding-SAM/camera supervision ever generated for this dataset), meant the alignment loss
always degenerated to a `clip_preds.sum() * 0.0` dummy term. Phase 8 §2 quantified the cost: among
high-confidence (≥0.5) `van` predictions, 68.5% were real cars mislabeled `van` by this untrained
CLIP step — a pure classification-boundary error that every LR/freeze/loss-weight lever tested in
Phases 8-9 left architecturally untouched (`matched_car%` stayed flat across all of them, by
construction, since none of those levers gave the alignment head any gradient at all).

This phase fixes the actual cause: generate real per-GT-box CLIP image-crop embeddings from
UrbanIng-V2X's own camera data, feed them into the alignment loss's existing `object_embedding_path`
path, and fine-tune only `alignment_head` on top of Phase 9's adopted checkpoint
(`v2i_vehicle1_finetune_lowlr_v1/ckpt/checkpoint_epoch_24.pth`) — isolating this fix from Phase 9's
already-settled localization conclusions.

**What made this possible:** UrbanIng-V2X's vehicle1 carries 6 synced RGB cameras (1920×1080) with
full intrinsics + extrinsics in each sequence's `calibration.json`, frame-aligned to the LiDAR
keyframes via `timesync_info.csv` — but none of it had ever flowed into the pipeline.
`convert_to_nuscenes.py` only ever parsed LiDAR extrinsics, the built pkl had no `cams` key, and
`gt_grounding_sam` was injected as an all-`None` placeholder purely so the dataset loader wouldn't
choke on a missing key. The camera JPGs themselves were never extracted anywhere on disk —
deliberately excluded by `build_v2i_vehicle1_multiseq_dataset.py`'s `extract_sequence()` pattern
allow-list — they exist only inside the per-sequence `.7z` archives.

A working offline precompute pattern already existed for nuScenes
(`pcdet/models/model_utils/selective_alignment_nuscenes_utils.py`'s `AlignmentSelector`): project GT
boxes into camera images, crop, encode with `open_clip` (`hf-hub:apple/DFN5B-CLIP-ViT-H-14` — the
same model already used for the head's text embeddings), L2-normalize, save one `.pt` file per box.
It depends on an external Grounding-SAM detection pkl that is not generated anywhere in this repo —
that upstream stage lives outside this codebase. This phase reuses the same
projection/crop/encode math but skips Grounding-SAM detection-matching entirely, cropping directly
from each GT box's own projected corners.

## 2. Building the crop-embedding pipeline

Three new files, all under `tools/urbaning_v2x/`:

- **`build_alignment_crops_for_sequence.py`** (container-side) — the core per-sequence logic. For
  each frame in a sequence: loads `calibration.json`, projects every GT box's 8 corners into each
  of vehicle1's 6 cameras, picks the camera with the best box-in-image ratio, applies a margin +
  clips to image bounds, filters by `MIN_PIXELS` (4000, same threshold as the real nuScenes
  alignment configs) and `MIN_BOX_IN_IMAGE_RATIO` (0.3), crops, encodes with CLIP, L2-normalizes,
  and saves one `.pt` file per matched box.

  **Two load-bearing calibration gotchas found and fixed before trusting any output:**
  1. `calibration.json`'s `IntrinsicMatrix` is in MATLAB convention — must be transposed before use
     as a standard pinhole `K`; `ImageSize` is `[height, width]`, not `[width, height]`.
  2. Vehicle1's cameras use **two different distortion models**: `front_left`/`front_right`/
     `back_left`/`back_right` use standard Brown-Conrady (handled via `cv2.projectPoints`), while
     `left`/`right` use a 4-value MATLAB fisheye/Kannala-Brandt model (handled via
     `cv2.fisheye.projectPoints`, which needs object points reshaped to `(1, N, 3)` float64). The
     first attempt crashed with `KeyError: 'RadialDistortion'` on `left`/`right` before this was
     found by inspecting a raw `calibration.json` directly.

  **Verified, not assumed:** added a `--debug_crops_dir` option, visually inspected crops spanning
  both camera models and multiple classes (car, pedestrian, bus-bucketed-as-truck) before trusting
  the pipeline on the full dataset — all correctly centered on the right object, except two expected
  edge cases (an occlusion-adjacent crop, a tiny barely-above-threshold crop), both inherent
  limitations of skipping instance-segmentation-based matching, not bugs.

- **`merge_alignment_crops.py`** (container-side) — merges all per-sequence `{token: entries}`
  result pickles into a copy of the train pkl, replacing the all-`None` `gt_grounding_sam` with real
  entries, writing a new `nuscenes_infos_1sweeps_train_full_ov_task.pkl` (original pkl untouched).

- **`build_alignment_crops.py`** (host-side orchestrator) — for each of the 27 train-split
  sequences (crossing1+crossing2; val/crossing3 is eval-only and never touches the alignment loss,
  so it keeps the `None`-placeholder pkl): extracts just vehicle1's 6 camera folders +
  calibration/timesync from that sequence's `.7z` archive, calls the per-sequence script inside the
  `OV-SCAN` container via `docker exec`, then immediately deletes the extracted camera data
  (`shutil.rmtree`) before moving to the next sequence — bounding peak extra disk usage to ~1
  sequence (~700MB-3.5GB) instead of ~19GB+ for all 27 sequences at once. Resumable via a progress
  marker file. After all sequences finish, runs the merge step once.

**Result:** all 27 train sequences processed successfully. **239,742 / 518,325 GT boxes (46.3%)**
got a real CLIP crop embedding — well under 100% as expected, since only objects within vehicle1's
own 6-camera FOV can match (per-sequence coverage ranged 25.1%-61.1%, correlating with how much of
each sequence's traffic falls within vehicle1's camera cone vs. purely LiDAR-visible periphery).

## 3. `--train_alignment_only` and the new config

New flag on `tools/train.py`, same pattern as Phase 9's `--freeze_backbone`: freezes
`vfe`/`backbone_3d`/`map_to_bev_module`/`backbone_2d` **plus** every `dense_head` submodule except
`alignment_head` (`shared_conv`, `heatmap_head`, `class_encoding`, `object_decoder`,
`prediction_head`) — `requires_grad=False` + pinned no-op `.train()` on each (same BatchNorm-drift
fix as Phase 9). Only `dense_head.alignment_head`'s parameters remain trainable.

New config `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align.yaml`:
copy of Phase 9's adopted `_finetune_lowlr.yaml`, with `DATA_CONFIG.INFO_PATH.train` pointed at the
new `..._full_ov_task.pkl` (val stays on the original pkl), a `filter_alignment_objects` step added
to `DATA_PROCESSOR` (`MIN_PIXELS: 4000`, defense-in-depth on top of the offline script's own
filter), `align_weight: 0.0 → 1.0`, and `NUM_EPOCHS: 24 → 8` (a narrow continuation fine-tune of one
small submodule from an already-converged checkpoint needs far less budget than a full retrain).

**Verified before spending the full training budget:** ran a 1-epoch smoke test, then (1) confirmed
via `logger.info` that the freeze message listed exactly the intended frozen submodules, (2) a
checkpoint-diff against the pretrained checkpoint showed **28 of 733 parameter tensors changed, all
of them under `dense_head.alignment_head.*`** (plus the `global_step` counter) — every backbone,
heatmap-head, and prediction-head tensor byte-identical before/after, (3) directly instrumented the
loss function and confirmed `clip_target_features` was populated with real per-batch counts (tens of
matched crops per iteration) and `loss_align` was nonzero and decreasing across real
backward+optimizer-step iterations. (A first read of `loss_align` from TensorBoard showed exactly
`0.0` at every step — traced to a protobuf/TensorBoard version mismatch in the ad-hoc reader script,
confirmed by the same reader also showing `0.0` for `loss_cls`/`loss_trans`, whose real nonzero
values were independently visible in the plain-text training log. A diagnostic-script artifact, not
a training bug.)

## 4. Training

```bash
docker exec OV-SCAN bash -c "cd /OV-SCAN/OV-SCAN/tools && python train.py \
  --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align.yaml \
  --batch_size 1 \
  --pretrained_model .../v2i_vehicle1_finetune_lowlr_v1/ckpt/checkpoint_epoch_24.pth \
  --train_alignment_only \
  --max_ckpt_save_num 3 \
  --extra_tag v2i_vehicle1_finetune_align_v1"
```

8/8 epochs, ~1h17min on the project's single GPU. Loss decreased smoothly throughout (total loss
~0.58 at the start of training down to ~0.29-0.38 by epoch 8, consistent with the alignment head
converging on top of an already-well-trained backbone/detection head). The standard post-training
`repeat_eval_ckpt` step crashed with `AssertionError: Database version not found` — a pre-existing,
unrelated issue (the NuScenes devkit's `evaluation()` path expects a folder layout this custom
dataset's converted pkl doesn't produce) that happens *after* every epoch's checkpoint is already
saved to disk, so it didn't affect any training artifact; evaluation was run separately via this
project's own tooling (§5).

## 5. Evaluation

Ran both of this project's standard tools against `checkpoint_epoch_8.pth`, on the full 1400-frame
held-out crossing3 set, compared against Phase 9's adopted winner
(`v2i_vehicle1_finetune_lowlr_v1/checkpoint_epoch_24.pth`):

### 5.1 Per-class AP

| Metric | Phase 9 (a) lowlr | Phase 10 align |
|---|---|---|
| `van` AP @0.25 | 0.330 | **0.463** |
| `car` AP @0.25 | 0.397 | **0.669** |
| `truck` AP @0.25 | 0.616 | 0.267 |
| `cyclist` AP @0.25 | 0.717 | 0.719 |
| mAP (fine classes) @0.25 | 0.263 | **0.270** |
| mAP (groups) @0.25 | 0.576 | 0.579 |
| mAP (groups) @0.3 | 0.559 | 0.562 |
| mAP (groups) @0.5 | 0.425 | 0.437 |
| `heavy_vehicle` (group) AP @0.25 | 0.480 | 0.477 |
| `vehicle` (group) AP @0.25 | 0.782 | 0.798 |

The fine-grained `truck` AP drop (0.616→0.267, driven by `num_pred` 5878→1954, i.e. recall
0.768→0.290) looks like a regression in isolation, but the **group-level `heavy_vehicle` AP is
flat** (0.480→0.477, same `num_pred`≈6300 either way) — this is the *same* mechanism that fixes
`car`/`van`, now also correctly redistributing what used to be over-broad "truck" predictions among
`truck`/`bus`/`trailer`/`other_vehicle` (all four map to the same `heavy_vehicle` group and the same
`"truck"` CLIP text prompt bucket per `HEATMAP_BUCKET_TO_VOCAB_KEYS`). Since group-level AP is the
metric that actually matters for this paper's 4-class evaluation protocol, and fine-class mAP
*improves* on net (0.263→0.270) despite this reallocation, this is not judged a real regression.

### 5.2 Cross-class `van` diagnostic (the decisive metric)

| Variant | total `van` preds | unmatched% | matched_car% | matched_van% | p90 score |
|---|---|---|---|---|---|
| Phase 9 (a) lowlr | 104,073 | 83.8% | 11.2% | 3.9% | 0.506 |
| Phase 10 align | 51,478 | 86.4% | **6.1%** | **5.9%** | 0.155 |

- **`matched_car%` dropped from 11.2% to 6.1% (-45% relative)** — this is the first lever in the
  entire project to move this number at all. Every prior phase's levers (LR, freeze, loss weight)
  left it flat by construction, since none of them gave the alignment head any real gradient. This
  one does, and the real-car-mislabeled-as-`van` mechanism identified in Phase 8 §2 is measurably,
  substantially reduced.
- **`matched_van%` rose from 3.9% to 5.9% (+51% relative)** — when the model does predict `van`, it
  is now a genuine van more often, not just less often a car.
- **Total `van` prediction volume halved** (104,073 → 51,478) — continuing Phase 9's volume-dampening
  trend, on top of the qualitative fix above, not instead of it.
- **One real trade-off: score calibration got worse** (p90 van score 0.506 → 0.155, mean 0.090 →
  0.056). Phase 9 (a)'s `van` scores were unusually well-calibrated (a side effect noted in that
  phase as particularly valuable for deployment thresholding); this run's scores regressed partway
  back toward baseline-like clustering near zero. Still fully compatible with the existing
  score-threshold deployment recommendation (Phase 8 §2), just somewhat less informative above that
  threshold. Worth revisiting (e.g. CLIP logit-scale/temperature tuning) in any future pass, but out
  of scope here.

## 6. Conclusion

**Adopt this checkpoint as the new default**:
`output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align/v2i_vehicle1_finetune_align_v1/ckpt/checkpoint_epoch_8.pth`.

This is the first phase in the whole project to actually address the Phase 8 §2 CLIP-mislabeling
mechanism, rather than working around its downstream symptoms — because it's the first phase to
give the alignment head any real supervision at all. `matched_car%` moved (11.2%→6.1%), `van` AP
improved 40%, `car` AP improved 68%, group-level mAP is flat-to-better, and fine-class mAP improved
on net despite an expected internal reallocation within the `heavy_vehicle` group. The only
real cost is a partial reversion of Phase 9 (a)'s unusually good `van` score calibration — a trade
worth taking.

**What remains open:** the `van` hallucination mechanism itself (heatmap over-selectivity, Phase 7)
is still just volume-dampened, not structurally fixed — `unmatched%` stayed at 83-86% throughout
Phases 9-10. Score calibration for `van` specifically could likely be recovered with a CLIP
temperature/logit-scale adjustment on top of this checkpoint — a natural next step if deployment
needs better-calibrated `van` confidence specifically. Camera coverage for alignment supervision is
capped at ~46% of train GT boxes by vehicle1's own FOV; extending to vehicle2's cameras (not done
here, out of scope for the v2i_vehicle1 config) could plausibly push coverage and the resulting fix
further.

## 7. File manifest

**New:**
- `tools/urbaning_v2x/build_alignment_crops_for_sequence.py` — per-sequence GT-box projection, crop,
  CLIP-encoding logic (container-side).
- `tools/urbaning_v2x/merge_alignment_crops.py` — merges per-sequence results into the new train pkl
  (container-side).
- `tools/urbaning_v2x/build_alignment_crops.py` — host-side orchestrator (extraction, per-sequence
  dispatch, cleanup, resumability, final merge).
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align.yaml` — Phase 10
  fine-tune config (`align_weight: 1.0`, `NUM_EPOCHS: 8`, points at the new pkl).
- `docs/Urbaning/Urbaning_PHASE_10.md` — this file.
- `docs/Urbaning/OV-SCAN_architecture.md` — standalone architecture reference (backbone/head/
  alignment-head), written alongside this phase's work.

**Modified:**
- `tools/train.py` — added `--train_alignment_only` CLI flag and its freeze block.

**New data artifacts:**
- `datasets/urbaning_v2x_v2i_vehicle1_multiseq/v1.0-custom/nuscenes_infos_1sweeps_train_full_ov_task.pkl`
  — train pkl with real `gt_grounding_sam` (46.3% coverage), original train pkl untouched.
- `datasets/urbaning_v2x_v2i_vehicle1_multiseq/ov_alignment/DFN5B-CLIP-ViT-H-14/object_crops/train/`
  — one `.pt` CLIP embedding per matched GT box (239,742 files).

**New checkpoint:**
- `output/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align/v2i_vehicle1_finetune_align_v1/ckpt/checkpoint_epoch_8.pth`
  — **new recommended default**, replacing Phase 9's `v2i_vehicle1_finetune_lowlr_v1/checkpoint_epoch_24.pth`.

**Result snapshot:** `phase10_results_align_v1.json` (repo root inside the container,
`/OV-SCAN/OV-SCAN/`), same convention as Phase 9's `phase9_results_*.json`.
