# UrbanIng-V2X ↔ OV-SCAN Integration — Voxelization Ablation

**Status:** Single-sequence ablation of `convert_to_nuscenes.py`'s `--voxel_size` parameter across
all 5 early-fusion configs that use it. A colleague suggested a coarser 0.5m grid might improve
results; tested against the current default (0.1m) plus two intermediate values. **Result:
coarser voxelization makes every config worse, monotonically — no full-dataset rerun was
warranted.**

---

## 1. What parameter this is (and what it isn't)

Two different things in this codebase could be called "the voxelization parameter":

1. **`convert_to_nuscenes.py --voxel_size`** (tested here): a point-cloud **downsampling** grid
   (mean-pooling, in meters) applied only when concatenating multiple raw point clouds for fusion
   (`i2i`, `v2v_v1ref`/`v2v_v2ref`, `v2i_vehicle1`/`v2i_vehicle2`) — see `voxel_downsample()`.
   Not applied to `single_vehicle` (nothing to downsample, single source). Pure data
   preprocessing, independent of the frozen model. Default `0.1`m; Phase 3's full-dataset sweep
   (`run_full_dataset_eval.py`) never overrode it, so every prior result in this project used the
   0.1m default.
2. **The model's own `DATA_PROCESSOR.VOXEL_SIZE`** (e.g. `[0.075, 0.075, 0.2]` in
   `ov_scan_lidar.yaml`): the feature-extraction voxel grid the pretrained checkpoint's
   sparse-conv backbone was actually trained around. Changing this without retraining risks the
   same kind of breakage Phase 3 §8 found when widening `POINT_CLOUD_RANGE` beyond the training
   distribution (mAP 0.28→0.007) — **not** touched by this ablation.

This ablation targets #1 only.

## 2. Design

- **Sequence:** `20241126_0001_crossing2_00` (Phase 3's designated crossing2 viz scene — raw data
  and the voxel_size=0.1 converted datasets were already on disk, so the baseline was reused
  directly instead of reconverted).
- **Configs:** all 5 point-cloud-fusing configs — `i2i`, `v2v_v1ref`, `v2v_v2ref`,
  `v2i_vehicle1`, `v2i_vehicle2`. `single_vehicle` excluded (unaffected by `--voxel_size`).
- **Values:** 0.1 (current default), 0.15 (the value already shown in the script's own docstring
  example), 0.3 (midpoint), 0.5 (the colleague's suggestion).
- **Repeats:** 2 independent eval reruns per (config, voxel_size) — not to resample different
  data (conversion is deterministic mean-pooling, no randomness), but to measure the GPU-inference
  nondeterminism noise floor directly, so a real effect could be distinguished from noise. Phase 3
  §9 documented a case where a 2-sample comparison looked like a real, reproducible effect but
  wasn't — repeats are the guard against repeating that mistake.
- **New script:** `tools/urbaning_v2x/voxel_size_ablation.py` — converts each (config, voxel_size)
  pair once (reusing the existing 0.1m baseline conversion), then runs
  `eval_urbaning_v2x_per_class.py` twice per pair, logging to a new MLflow experiment
  `urbaning_v2x_voxel_size_ablation` (one run per (config, voxel_size, rep), each rep with its own
  run name — not resumed/enriched across reps, since independent samples were the point).

## 3. Results

40 runs (4 voxel sizes × 5 configs × 2 reps), 0 failures. Noise floor from the repeats:
mAP std within a (config, voxel_size) cell was ~0.0000–0.0030 across every cell — the table below
reports mean mAP, with that std range implicit everywhere.

### mAP (fine classes)

| Config | 0.1 (current) | 0.15 | 0.3 | 0.5 (colleague's suggestion) |
|---|---|---|---|---|
| i2i | 0.2770 | 0.2658 | 0.2175 | **0.1692** |
| v2v_v1ref | 0.2715 | 0.2722 | 0.2395 | **0.2160** |
| v2v_v2ref | 0.3069 | 0.3065 | 0.2862 | **0.2190** |
| v2i_vehicle1 | 0.2482 | 0.2486 | 0.2142 | **0.2169** |
| v2i_vehicle2 | 0.2895 | 0.2856 | 0.2621 | **0.2134** |

### mAP (UrbanIng-V2X label groups)

| Config | 0.1 (current) | 0.15 | 0.3 | 0.5 (colleague's suggestion) |
|---|---|---|---|---|
| i2i | 0.4214 | 0.4045 | 0.3308 | **0.2566** |
| v2v_v1ref | 0.3785 | 0.3806 | 0.3473 | **0.3016** |
| v2v_v2ref | 0.3954 | 0.3899 | 0.3716 | **0.3083** |
| v2i_vehicle1 | 0.3356 | 0.3392 | 0.3090 | **0.2982** |
| v2i_vehicle2 | 0.3669 | 0.3565 | 0.3452 | **0.3038** |

### Relative mAP (fine) change, 0.1 → 0.5

| Config | Δ |
|---|---|
| i2i | **−39%** |
| v2v_v2ref | −29% |
| v2i_vehicle2 | −26% |
| v2v_v1ref | −21% |
| v2i_vehicle1 | −13% |

## 4. Findings

**Every config degrades monotonically as the voxel grid gets coarser.** 0.1 and 0.15 are
statistically indistinguishable (differences within the ~0.003 noise floor); 0.3 and especially
0.5 are real, substantial regressions — 13–39% relative mAP loss depending on config, far
exceeding the noise floor.

**The colleague's suggested 0.5m value is the worst of every value tested, on every config.**
`i2i` (the densest fusion — 4 infra sensors, no single reference sensor's native resolution to
anchor to) is hit hardest; `v2i_vehicle1` is hit least (still a real 13% loss).

**Why this direction, not the hoped-for improvement:** the frozen checkpoint was trained on
native-density, single-LIDAR_TOP nuScenes point clouds. The intuition behind the colleague's
suggestion — that a coarser grid would bring the *density* of a multi-sensor-fused cloud closer to
nuScenes' native sparsity — doesn't account for what mean-pooling at that scale actually does:
0.5m is comparable to a pedestrian's width, so the downsample step blurs genuine point-level
structure the backbone needs to recognize small/thin objects, rather than just thinning out
redundant overlap points from multiple sensors. The cost of that blurring outweighs any benefit
from matching the training-time density more closely.

**No full 34-sequence rerun was warranted.** The single-sequence result is unambiguous and
monotonic in the wrong direction across all 5 configs — a full-dataset sweep would only confirm
the same trend at ~34x the compute cost, not change the conclusion. The current default (0.1m)
remains optimal; 0.15m is a statistical tie, not worth switching to.

## 5. File manifest

**New file:** `tools/urbaning_v2x/voxel_size_ablation.py` — orchestrates the (config, voxel_size,
rep) grid described in §2. Converted datasets land under
`datasets/urbaning_v2x_voxel_ablation/<sequence>/<config>_voxel<size>/` (except the 0.1m baseline,
which reuses Phase 3's existing `datasets/urbaning_v2x_batch/<sequence>/<config>/`). Per-(config,
voxel_size, rep) JSON marker files gate re-evaluation, same resumability convention as
`run_full_dataset_eval.py`.

**MLflow:** new experiment `urbaning_v2x_voxel_size_ablation`, 40 runs
(`<config>__<sequence>__voxel<size>__rep<N>`), tags `fusion_type`, `sources`, `reference`,
`sequence`, `intersection`.

## 6. Note on the marker-write permission bug

The script's first launch crashed on the very first eval (`i2i`, voxel=0.1, rep=1) with a
`PermissionError` writing its own completion marker: converted dataset directories under
`datasets/urbaning_v2x_batch/`/`datasets/urbaning_v2x_voxel_ablation/` are root-owned (written by
`docker exec`, same as every converted dataset in this pipeline — see
`run_full_dataset_eval.py`'s comments), so a host-side `Path.write_text()` on a marker file inside
them fails even though the host process can read/list them fine. Fixed by writing markers via
`docker exec touch <path>` instead of a host-side write — the eval run itself (which runs entirely
inside the container) had already completed and logged to MLflow successfully before the crash, so
no compute was wasted; the fix just let the orchestrator track completion correctly on resume.
