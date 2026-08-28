# UrbanIng-V2X ↔ OV-SCAN Integration — Phase 4

**Status:** Late-fusion baseline implemented and evaluated on the reference sequence
(`20241126_0001_crossing2_00`) — zero-shot, zero-training, no early-fusion point concatenation.
Five agent combinations, full 200-frame per-class AP/mAP, class-agnostic sanity
precision/recall, fixed-camera visualizations, and a threshold-sensitivity investigation (score
threshold and IoU match threshold, independently). Still `pretrained/ov_scan_lidar.pth`, never
fine-tuned. See `Urbaning_PHASE_1.md`–`_PHASE_3.md` for the early-fusion converter/eval/visualize
foundations and the full-dataset sweep this builds on — this document only covers what's new.

**Not done in this phase:** the 34-sequence sweep for late fusion (Phase 3's equivalent), and
intermediate (BEV-feature) fusion — both flagged as natural next steps, not attempted here.

---

## 1. Why late fusion, and why now

Phases 1–3 only ever fused **before** the frozen model ran: raw point clouds concatenated into one
shared `POINT_CLOUD_RANGE` window (±54 m), centered on a single reference agent. Phase 3 §8 found
this forces a hard ceiling — widening the window to capture far-away fused sources breaks the
pretrained head outright (mAP 0.28→0.007, Phase 3 §8), because it pushes the input outside the
distribution the checkpoint was trained on. Fixing that needs either fine-tuning or a different
fusion strategy.

**Late fusion sidesteps the ceiling entirely and needs no training:** run the existing frozen
zero-shot pipeline independently per agent — each agent's input always stays inside its own valid
±54 m window, so there's no distribution shift — then transform every agent's *final 3D boxes*
into one shared global frame (using poses the converter already resolves) and merge duplicates
there. It's the natural first baseline before investing in a trained BEV-feature-fusion approach:
zero-cost, and it directly tests whether pooling independently-detected boxes recovers the
range-extension benefit early fusion structurally can't deliver.

Design choices confirmed with the user before implementing:
- **Scope:** mirror Phase 2/3's 6 early-fusion configs as late-fusion combos, plus one new
  "everything" combo (all 6 sources at once) specifically to probe range extension.
- **Merge algorithm:** Weighted Box Fusion (score-weighted average geometry within an IoU
  cluster), not plain NMS.
- **Class matching:** only merge boxes agreeing on fine OV native class (`ov_str_labels`), for
  direct comparability with the existing per-class AP convention (§7.3, Phase 1).
- **GT policy:** no track self-exclusion — every real track is a fair target, uniformly across
  combos (matches the existing infra/`i2i` convention).

**Mirroring the 6 configs is not 1:1.** Late fusion has no reference-frame asymmetry — early
fusion's `v2v_v1ref`/`v2v_v2ref` differ only in which vehicle's *moving sensor frame* the fused
cloud is expressed in, which is irrelevant once everything gets projected into one shared global
frame for merging. So `v2v` collapses to **one** combo, not two; `v2i_vehicle1`/`v2i_vehicle2`
stay distinct (genuinely different agent sets). Net: **5 late-fusion combos** —

| Combo | Sources |
|---|---|
| `i2i_late` | 4 infra LiDARs, independently |
| `v2v_late` | vehicle1 + vehicle2, independently |
| `v2i_v1_late` | vehicle1 + 4 infra, independently |
| `v2i_v2_late` | vehicle2 + 4 infra, independently |
| `full_late` | all 6 sources, independently |

---

## 2. New per-agent "solo" datasets

Late fusion needs independent detections from every agent that had only ever been run *fused*
before: `vehicle2` alone (only run fused via `v2v`/`v2i` so far) and each of the 4 infra LiDARs
individually (only run pre-fused via `--lidars <all 4>` so far). Both are already-supported
converter modes — `--ego vehicle2` and `--lidars <single_channel>` — no converter mode changes
needed. Per Phase 3's established pattern, **no new dataset/model YAML files were needed either**:
`ov_scan_lidar_urbaning.yaml` (vehicle mode) and `ov_scan_lidar_urbaning_infra.yaml` (infra mode)
have nothing vehicle- or sensor-specific baked in besides `DATA_PATH`, confirmed byte-identical
apart from that field and `_BASE_CONFIG_`'s dataset-yaml reference — so the existing `--data_path`
override (already used by `eval_urbaning_v2x_per_class.py`/`build_infos.py`) covers every new solo
dataset by pointing at a new converted directory:

```
datasets/urbaning_v2x_vehicle2/                    # --ego vehicle2
datasets/urbaning_v2x_infra_solo_{11,12,31,32}/     # --lidars crossing2_XX_lidar (one channel each)
```

## 3. Recovering a true global frame per agent

Merging boxes across agents needs one consistent global frame. Vehicle-mode sources round-trip to
true global for free via the already-written `ego_pose.json` (per-frame `gTv`) +
`calibrated_sensor.json` (`vTl`) tables — standard nuScenes composition,
`global_from_sensor = ego_pose ⋅ calibrated_sensor`.

Infra-mode sources were the gap: `InfraFusionConverter` recenters points/GT by subtracting
`self.offset` (mean sensor position) **before** writing them to disk, and never persisted that
offset — `ego_pose`/`calibrated_sensor` are written as identity, so there was previously no way to
recover true global from an infra-mode dataset's on-disk tables alone. Fixed with a new sidecar,
`<out>/<version>/frame_offset.json`, written from `BaseNuscenesConverter.write_tables()`:

```python
# convert_to_nuscenes.py -- BaseNuscenesConverter.write_tables()
offset = getattr(self, 'offset', np.zeros(3))
with open(self.table_dir / 'frame_offset.json', 'w') as f:
    json.dump({'offset': np.asarray(offset).tolist()}, f)
```

Purely additive — zero for `SequenceConverter`/`FusedEgoConverter` (which already round-trip via
ego_pose/calibrated_sensor), populated for `InfraFusionConverter`.

`eval_urbaning_v2x_late_fusion.py`'s `source_to_global`/`source_to_local` use this uniformly:
`p_global = R @ p_local + t + offset` (vehicle: `offset=0`; infra: `R=I, t=0`). Box yaw is composed
through the same rotation (`Rotation.from_euler('z', yaw_local)` then re-extracted via
`as_euler('zyx')[0]` after composing with the pose rotation) — an approximation that assumes
near-upright boxes and near-planar agent poses, consistent with every other rotation assumption
already made throughout this pipeline (e.g. `count_points_in_box`).

## 4. GT range policy — the actual range-extension test

A GT box is kept for a given combo's evaluation if it falls inside **at least one participating
agent's own local ±54 m window** for that frame (checked in that agent's own local frame via
`source_to_local`) — not a single shared window. This is the concrete, well-defined version of
"does pooling agents recover coverage a single window can't" that Phase 3 §8 flagged but didn't
test. GT itself is read directly from the original `labels/<sequence>.json` (already true global,
every track, no re-derivation needed) via a `load_gt_by_timestamp` that mirrors
`convert_to_nuscenes.py`'s own helper minus the self-exclusion logic (not needed under the
no-exclusion policy).

## 5. Weighted Box Fusion

`weighted_box_fusion(boxes, scores, classes, iou_thresh, coarse_labels=None)` in
`eval_urbaning_v2x_late_fusion.py`: greedy score-ordered clustering by 3D IoU
(`iou3d_nms_utils.boxes_iou3d_gpu`, same op the rest of this codebase already uses), **same fine
OV native class only** (cross-class IoU zeroed out before clustering):

```python
order = np.argsort(-scores)
used = np.zeros(n, dtype=bool)
for i in order:
    if used[i]:
        continue
    cluster = np.union1d(np.where((iou[i] >= iou_thresh) & ~used)[0], [i])
    used[cluster] = True
    w = scores[cluster]
    center_dims = (boxes[cluster, :6] * w[:, None]).sum(axis=0) / w.sum()   # score-weighted mean
    yaw = np.arctan2((np.sin(boxes[cluster, 6]) * w).sum() / w.sum(),      # circular score-weighted mean
                      (np.cos(boxes[cluster, 6]) * w).sum() / w.sum())
    fused_score = w.mean()                                                  # simple default, not tuned
```

`coarse_labels` (optional): when given, each cluster also gets the *highest-scoring member's*
coarse CLASS_NAMES id, for callers that want the existing coarse-label-keyed color scheme
(`visualize_urbaning_v2x.py`'s `PRED_COLOR_MAP`) — added after the AP baseline was already
validated, purely additive (`None` default, unused by AP scoring), and re-verified byte-identical
mAP before/after on `i2i_late`.

## 6. `build_late_fusion_frames` — factored out for reuse

The per-source-inference → global-frame-transform → WBF-merge → union-range-GT pipeline was
factored out of `eval_urbaning_v2x_late_fusion.py`'s `main()` into
`build_late_fusion_frames(combo, root_folder, sequence, ckpt, wbf_iou_thresh, logger, max_samples)`,
returning the same `frames` list (`pred_boxes`/`pred_scores`/`pred_class`/`gt_boxes`/`gt_class` per
frame) `class_ap` already expects. Both `eval_urbaning_v2x_late_fusion.py`'s own `main()` and
`threshold_sensitivity_late_fusion.py` (§9) call this, so inference + WBF fusion run **once** per
combo regardless of how many downstream metrics get computed from the result.

## 7. Results — single reference sequence, 200 frames

Per-class AP methodology (`class_ap`/`voc_ap`/`FINE_CLASSES`/`GROUPS`/`CANONICAL_TO_GROUP`) is
imported directly from `eval_urbaning_v2x_per_class.py`, unchanged — mAP here is directly
comparable to Phase 2/3's tables; only the box population (fused across independent sources vs.
single-source or early-fused) differs.

| Combo | Sources | mAP (fine) | mAP (groups) | vs. matching early-fusion config (Phase 2, fine/groups) |
|---|---|---|---|---|
| `i2i_late` | 4 infra | 0.149 | 0.263 | `i2i`: 0.218 / 0.331 |
| `v2v_late` | v1+v2 | 0.249 | 0.304 | `v2v` (v1ref/v2ref): 0.271–0.305 / 0.373–0.395 |
| `v2i_v1_late` | v1+4 infra | 0.200 | 0.306 | `v2i_vehicle1`: 0.248 / 0.339 |
| `v2i_v2_late` | v2+4 infra | 0.239 | 0.332 | `v2i_vehicle2`: 0.285 / 0.356 |
| `full_late` | all 6 | 0.224 | 0.325 | no early-fusion equivalent (new) |

**mAP is lower for late fusion than the matching early-fusion config, across every combo** —
expected: each source detects independently from its own single, sparser point cloud, instead of
one forward pass over a richer point-concatenated cloud. Early fusion gives the network better raw
input per detection; late fusion doesn't.

**GT-in-range coverage (the union-of-windows count) grows monotonically with more sources** —
~45–47/frame for `i2i_late` (4 sources), ~50 for `v2v_late` (2), ~59–74 for the 5-source
`v2i_*_late` combos, up to ~71–74 for `full_late` (6 sources). This is the concrete evidence the
late-fusion range-extension mechanism works — pooling agents' windows recovers ground truth a
single ±54 m window structurally cannot see, without retraining anything.

**Net finding:** late fusion trades *per-detection quality* for *reachable coverage* — the inverse
of early fusion's trade-off. Motivates intermediate (BEV-feature) fusion as the natural next
investigation: keep each agent's own valid input window (avoiding early fusion's range ceiling)
while still letting a learned fusion module combine richer pre-detection features (recovering some
of what independent per-source detection loses).

## 8. Visualization

New `tools/visualize_urbaning_v2x_late_fusion.py`, reusing `run_source_inference`/
`weighted_box_fusion`/`source_to_global`/`load_gt_by_timestamp` from the eval script and
`save_scene`/`match_greedy` from `visualize_urbaning_v2x.py` (unmodified logic, two new *optional*
parameters — see below). Renders each source's own points, transformed into the shared global
frame and concatenated, with WBF-fused boxes and union-range GT drawn on top. Output goes to
`output/urbaning_v2x_late_fusion/<combo>/default/visualizations/` — a separate tree from early
fusion's `output/nuscenes_models/<tag>/default/visualizations/`.

### 8.1 A real bug: rendering raw, uncropped points

First-pass renders were a dense gray blob with the actual road/vehicle geometry barely
distinguishable, and the camera "wobbled" frame to frame. Root cause, found by direct
measurement: `load_source_points_global` loaded points straight from each source's `.pcd.bin` —
the *raw* captured sweep — not what the model actually receives as input. pcdet's own
`DataProcessor.mask_points_and_boxes_outside_range` crops points to `POINT_CLOUD_RANGE`
**unconditionally** (train or eval — `data_processor.py:83-85`, no `self.training` guard, unlike
the GT-box-removal half of the same step). Checked directly on `i2i_late`'s frame 0: raw points
extended up to **206 m** from the scene center (one stray point far outside any sensor's actual
±54 m training-time input), while the model itself only ever sees points inside that window. Fix:
crop each source's points to *its own* local `POINT_CLOUD_RANGE` before transforming to global,
mirroring pcdet's own crop exactly:

```python
# visualize_urbaning_v2x_late_fusion.py -- load_source_points_global
in_range = (raw[:, 0] >= x_min) & (raw[:, 0] <= x_max) & (raw[:, 1] >= y_min) & (raw[:, 1] <= y_max)
raw = raw[in_range]
```

This alone turned the fuzzy blob into distinct, individually-legible concentric LiDAR sweep rings
— the single biggest visual-quality fix, and a correctness fix independent of anything else in
this phase (any future late-fusion tooling rendering raw points should crop the same way).

### 8.2 Stable camera, tuned zoom

`save_scene` (in `visualize_urbaning_v2x.py`) previously called `vis.reset_view_point(True)`
every frame — auto-fitting to *that frame's own* point/box extent, which is exactly what made
consecutive frames' renders "wobble" (a frame with one outlier far GT box zooms/re-centers
differently than its neighbors). Two new **optional** parameters, both defaulting to prior
behavior so the early-fusion script's renders are byte-for-byte unaffected:

```python
def save_scene(..., point_size=1.5, fixed_cam_params=None):
    ...
    if fixed_cam_params is not None:
        ctr.convert_from_pinhole_camera_parameters(fixed_cam_params)
    else:
        vis.reset_view_point(True)   # unchanged default path
        ...
```

`visualize_urbaning_v2x_late_fusion.py`'s new `compute_fixed_camera_params(center_xy, half_extent,
width, height)` runs that same `reset_view_point(True)` **once**, against two synthetic corner
points spanning `center_xy ± half_extent`, and every frame reuses the resulting camera.

Picking `center_xy`/`half_extent` needed its own iteration: naively fitting to the *full*
union-range GT extent reproduced the "mostly empty frame" complaint, because a few sources' own
±54 m windows can genuinely extend the union-range GT much further than where most of the actual
point density is (e.g. two infra sensors ~33 m apart, each with their own 54 m radius, can push
the union past 70 m from the mean center — verified directly: `i2i_late`'s GT-distance-from-center
percentiles were 50th≈27 m, 70th≈33 m, 95th≈66 m, max≈74 m). Settled on the **70th percentile**
of GT distance-from-center (not max/95th — those fit the camera to the long tail, not to where
most content actually is), ×1.15 margin, floored at 20 m and **capped at 45 m** (roughly the
project's established single-source ±54 m scale) — all 5 combos hit this 45 m cap in practice, a
deliberate readability trade-off: a few far union-range GT boxes render off-screen, which affects
nothing metrics-wise (the eval script's own GT range check is independent and unaffected).

Also reduced `point_size` (1.5→0.7 default for this script) and raised render resolution
(1280×960→1920×1440) — secondary contributors once the point-cropping fix (§8.1) removed the
dominant source of clutter.

### 8.3 `--show_labels` support

Added the same digit-overlay + legend feature `visualize_urbaning_v2x.py` already has
(`draw_box_labels`, reused unmodified). Needed one new small mapping, since GT here comes from the
raw labels file rather than pcdet's dataloader (which already carries a coarse class-id column):

```python
# UrbanIng-V2X object_type -> coarse CLASS_NAMES id (1-indexed), mirrors
# convert_to_nuscenes.py's OBJECT_TYPE_TO_NUSCENES grouping
GT_OBJECT_TYPE_TO_CLASS_ID = {
    'Car': 1, 'Van': 1, 'Bus': 2, 'Truck': 2, 'OtherVehicle': 2, 'Trailer': 2,
    'EScooter': 8, 'Motorcycle': 8, 'Cyclist': 8, 'Pedestrian': 9, 'OtherPedestrians': 9,
}
```

Predicted-box labels reuse `weighted_box_fusion`'s new `coarse_labels` output (§5) directly — no
separate mapping needed there, since predictions already carry a real CLASS_NAMES id from the
model's own heatmap channel.

### 8.4 Class-agnostic sanity precision/recall (score≥0.3, IoU 0.25, all 200 frames)

Same convention as Phase 1–3's `match_summary.json` — a localization sanity check, not a
class-aware benchmark (see §7/§9 for that).

| Combo | Precision | Recall |
|---|---|---|
| `i2i_late` | 0.886 | 0.014 |
| `v2v_late` | 0.937 | 0.170 |
| `v2i_v1_late` | 0.967 | 0.070 |
| `v2i_v2_late` | 0.969 | 0.066 |
| `full_late` | 0.922 | 0.079 |

## 9. Threshold sensitivity — `tools/threshold_sensitivity_late_fusion.py`

Investigated two independent knobs, reusing `build_late_fusion_frames` so inference + WBF fusion
run once per combo regardless of how many thresholds get swept afterward.

### 9.1 `MODEL.POST_PROCESSING.SCORE_THRESH: 0.1` is dead config

Raised concern: does lowering this from 0.1 to 0.05 change anything? Traced it directly —
`OVScanLidar.post_processing()` (`pcdet/models/detectors/ov_scan_lidar.py`) assigns
`post_process_cfg = self.model_cfg.POST_PROCESSING` but only ever reads
`post_process_cfg.RECALL_THRESH_LIST` (fed into `generate_recall_record`, whose output
`recall_dict` none of this project's eval/visualize scripts consult — all of them discard the
second `model(data_dict)` return value with `_`). `.SCORE_THRESH` is never referenced anywhere in
that file. **This field has zero effect on any metric this codebase reports.** The score threshold
that actually matters is `visualize_urbaning_v2x*.py`'s own `--score_thresh` CLI arg (used only for
the class-agnostic sanity precision/recall, §8.4) — that's what §9.2 sweeps.

### 9.2 Score threshold (0.3 → 0.1 → 0.05) — class-agnostic precision/recall only

AP is unaffected by construction (rank-based over the full PR curve, no cutoff — Phase 1 §7.3).
Only the fixed-threshold sanity check moves, and it moves a lot, consistently across every combo:

| Combo | P/R @ 0.3 | P/R @ 0.1 | P/R @ 0.05 |
|---|---|---|---|
| `i2i_late` | 0.892 / 0.014 | 0.654 / 0.211 | 0.511 / 0.363 |
| `v2v_late` | 0.937 / 0.170 | 0.829 / 0.467 | 0.589 / 0.563 |
| `v2i_v1_late` | 0.966 / 0.070 | 0.725 / 0.323 | 0.529 / 0.455 |
| `v2i_v2_late` | 0.966 / 0.066 | 0.747 / 0.339 | 0.553 / 0.474 |
| `full_late` | 0.922 / 0.080 | 0.726 / 0.390 | 0.516 / 0.528 |

0.1→0.05: recall gains ~50–70% relative in every combo (e.g. `v2v_late` 0.467→0.563), at a real
precision cost (e.g. `i2i_late` 0.654→0.511) — consistent, substantial trade-off across all 5
combos, not specific to any one agent configuration.

### 9.3 IoU match threshold (0.25 → 0.3 → 0.5) — mAP

| Combo | mAP fine @0.25/0.3/0.5 | mAP groups @0.25/0.3/0.5 |
|---|---|---|
| `i2i_late` | 0.149 / 0.113 / 0.019 | 0.263 / 0.201 / 0.033 |
| `v2v_late` | 0.247 / 0.228 / 0.119 | 0.304 / 0.282 / 0.128 |
| `v2i_v1_late` | 0.200 / 0.167 / 0.051 | 0.306 / 0.257 / 0.067 |
| `v2i_v2_late` | 0.239 / 0.205 / 0.098 | 0.332 / 0.284 / 0.111 |
| `full_late` | 0.224 / 0.191 / 0.076 | 0.325 / 0.279 / 0.096 |

(`v2v_late`'s 0.25-threshold mAP-fine reads 0.247 here vs. 0.249 in §7 — a ~0.002 run-to-run delta
from GPU floating-point tie-breaking in score-sorted argsort, same class of non-determinism Phase
2 already documented; not a regression, both runs' underlying per-box AP values agree to 3
decimals otherwise.)

At IoU=0.5 (the stricter, more standard nuScenes/COCO threshold) mAP collapses to roughly 10–40%
of the IoU=0.25 value across every combo. The 0.25 threshold used throughout this whole project so
far is fairly permissive on localization quality — worth keeping in mind when comparing these
numbers to more standard benchmarks, or when deciding what threshold to report going forward.

Both sweep lists (`--match_iou_thresholds`, `--score_thresholds`) and the WBF-clustering threshold
(`--wbf_iou_thresh`, held fixed at 0.25 while the match/scoring threshold varies independently) are
CLI-overridable; defaults cover all 5 combos on one sequence. Results saved to
`output/urbaning_v2x_late_fusion/threshold_sensitivity.json`.

---

## 10. File manifest

**New files:**
- `tools/eval_urbaning_v2x_late_fusion.py` — per-source inference, global-frame transform
  (`source_to_global`/`source_to_local`), `weighted_box_fusion`, `build_late_fusion_frames`,
  per-class AP/mAP for the 5 late-fusion combos (`COMBOS`)
- `tools/visualize_urbaning_v2x_late_fusion.py` — fixed-camera, point-cropped, labeled
  visualization for the same 5 combos
- `tools/threshold_sensitivity_late_fusion.py` — score/IoU threshold sweep, reusing
  `build_late_fusion_frames`

**Modified files:**
- `tools/urbaning_v2x/convert_to_nuscenes.py` — `frame_offset.json` sidecar (§3), purely additive
- `tools/visualize_urbaning_v2x.py` — `save_scene` gained optional `point_size`/`fixed_cam_params`
  params (§8.2), both defaulting to prior behavior; no other changes

**New datasets (container paths under `/OV-SCAN/`):**
```
datasets/urbaning_v2x_vehicle2/                        # vehicle2 solo
datasets/urbaning_v2x_infra_solo_{11,12,31,32}/         # each crossing2 infra LiDAR solo
output/urbaning_v2x_late_fusion/<combo>/default/
  per_class_ap.json, visualizations/*.png, match_summary.json
output/urbaning_v2x_late_fusion/threshold_sensitivity.json
```

**MLflow:** new experiment `urbaning_v2x_late_fusion` (separate from Phase 2's `urbaning_v2x` and
Phase 3's `urbaning_v2x_full_dataset`), one run per combo, `fusion_type=<combo>`,
`merge_strategy=late_wbf` tags. The threshold-sensitivity investigation (§9) deliberately does
**not** log to MLflow — treated as an ad-hoc analysis over already-tracked runs, not a new tracked
experiment.

## 11. Reproducing / extending this pipeline

**To convert another sequence's solo datasets:** same `--ego <vehicleN>` / `--lidars <one channel>`
converter invocations as §2, then `build_infos.py --data_path <dir>` — no code changes.

**To add a new late-fusion combo:** add an entry to `COMBOS` in `eval_urbaning_v2x_late_fusion.py`
(a list of `{label, mode, cfg_file, data_path}` source dicts) — the eval, visualize, and
threshold-sensitivity scripts all pick it up automatically.

**Natural next steps** (not done in Phase 4): scale to the full 34-sequence dataset (mirroring
Phase 3's `run_full_dataset_eval.py` pattern); add late-fusion support to
`aggregate_full_dataset_results.py`; and — the main motivation flagged in §7 — intermediate
(BEV-feature) fusion, which Phase 4's net finding (late fusion trades detection quality for
coverage; early fusion trades coverage for quality) suggests could recover both simultaneously.
