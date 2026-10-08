# UrbanIng-V2X ↔ OV-SCAN Integration — Phase 5

**Status:** Phase 4's late-fusion baseline (5 combos: `i2i_late`, `v2v_late`, `v2i_v1_late`,
`v2i_v2_late`, `full_late`), evaluated on only the single crossing2 reference sequence, scaled to
the full 34-sequence UrbanIng-V2X dataset (all 3 intersections) — mirroring Phase 3's early-fusion
full-dataset sweep. Also sweeps the AP-matching IoU threshold (0.25 primary + 0.3 + 0.5) from the
same WBF-fused predictions, same convention as Phase 3 §10. Still `pretrained/ov_scan_lidar.pth`,
never fine-tuned.

**The key design problem this phase solves:** Phase 4's late fusion needs independent per-agent
detections, so each of the 5 combos re-runs inference for every source it contains. Naively
calling `eval_urbaning_v2x_late_fusion.py` once per combo per sequence would redundantly re-infer
vehicle1/vehicle2 up to 3x each and every infra channel up to 4x each (22 source-inference-passes
per sequence instead of 6) — this phase restructures the pipeline so each sequence's up-to-6
unique sources run inference exactly once and all 5 combos reuse the cached predictions.

---

## 1. Compute-sharing restructure

`eval_urbaning_v2x_late_fusion.py`'s `build_late_fusion_frames` (Phase 4) did per-combo
inference+fusion in one function. Split into two reusable pieces:

- `run_all_sources_inference(sources, ckpt, logger, max_samples)` — runs `run_source_inference`
  once per source, returns `{label: predictions_by_token}` + `point_cloud_range`. Call once on the
  de-duplicated union of sources across every combo being evaluated.
- `fuse_combo_frames(combo_sources, frame_predictions, point_cloud_range, root_folder, sequence,
  wbf_iou_thresh, logger, max_samples)` — WBF-fuses one combo's own source subset from the shared
  cache into the `frames` list `class_ap` expects. Cheap (no inference), so calling it once per
  combo is free compared to the inference it reuses.

`build_late_fusion_frames` itself is kept as a thin backward-compatible wrapper (unchanged
behavior/signature) so Phase 4's `main()` and `threshold_sensitivity_late_fusion.py` are
unaffected.

Two new helpers generalize combo construction beyond the single crossing2 reference sequence the
module-level `COMBOS`/`VEHICLE1`/`INFRA_11` etc. are hardcoded to:

- `combo_sources_for(intersection, data_root)` — builds the same 5 combos pointing at
  `data_root/{vehicle1_solo, vehicle2_solo, infra_solo_<channel>}`, using the right infra channel
  suffixes per intersection (crossing3 uses 21/22, not 31/32 — `INFRA_CHANNELS`, duplicated from
  `run_full_dataset_eval.py` since that's a host-side script not meant to be imported here).
  Source dicts are shared **by reference** across every combo that uses them, not copied, so one
  `run_all_sources_inference` pass populates each source's poses/offset/ordered_tokens once and
  every combo sees it.
- `unique_sources(combos)` — de-duplicates those shared-by-reference source dicts by label, for
  the single inference pass.

Also factored the Phase 3-style IoU sweep into `compute_ap_sweep(frames, primary_iou, extra_ious,
logger)`, reused by both the single-combo CLI (`eval_urbaning_v2x_late_fusion.py --combo ...`, now
also takes `--extra_match_iou_threshs`, default `0.3,0.5`) and the new batch script below — one
implementation instead of duplicating the sweep per caller.

## 2. `eval_urbaning_v2x_late_fusion_batch.py` — one process per sequence, all 5 combos

New script: takes `--sequence`/`--data_root`/`--intersection`, builds `combo_sources_for(...)`,
de-dups via `unique_sources(...)`, runs `run_all_sources_inference` **once**, then loops over the 5
combos calling `fuse_combo_frames` + `compute_ap_sweep` + one `get_or_create_run`-resumed MLflow
run each (experiment `urbaning_v2x_late_fusion_full_dataset`). `--only_combo` restricts to one
combo for quick testing (inference still only runs for that combo's own sources).

**Verified via a sanity dry run** (symlinked data_root pointing at Phase 4's existing crossing2
solo datasets, `--max_samples 20`): log showed exactly 6 `"<label>] N frames inferred"` lines
total — one per unique source, not up to 22 — confirming the dedup actually fires. The resulting
`v2v_late` mAP (0.538/0.463 fine/groups) matched a parallel run of the unmodified Phase 4
`eval_urbaning_v2x_late_fusion.py --combo v2v_late` on the same 20 frames (0.536/0.461) within the
already-documented ~0.002-0.005 GPU-nondeterminism tolerance (Phase 3 §9) — the refactor is
behaviorally equivalent, just cheaper.

## 3. `run_full_dataset_eval_late_fusion.py` — host orchestrator

Late-fusion analogue of Phase 3's `run_full_dataset_eval.py`, same storage-efficient pattern:
extract (host-side 7z) → convert up to 6 solo-source datasets (`vehicle1_solo` via `--ego
vehicle1`, `vehicle2_solo` via `--ego vehicle2`, `infra_solo_<channel>` via `--lidars
<one_channel>`, one per intersection's 4 infra LiDARs) → run the batch eval script once (all 5
combos) → delete raw+converted data. The 3 designated per-intersection viz-scene sequences (same
ones Phase 3 used: `20241126_0008_crossing1_00`, `20241126_0001_crossing2_00`,
`20241127_0009_crossing3_00`) keep their data and additionally get
`visualize_urbaning_v2x_late_fusion.py` run for all 5 combos (that script gained `--data_root`/
`--intersection` overrides, falling back to Phase 4's original module-level `COMBOS` when omitted,
so the single-sequence crossing2 usage is unaffected).

Resumable via per-(sequence, step) JSON marker files, same convention as Phase 3.

### 3.1 Preliminary end-to-end test (real conversion, not symlinks)

Before committing to the 34-sequence run, ran the orchestrator on just the crossing2 reference
sequence (`--only-sequence 20241126_0001_crossing2_00`) end to end — real `convert_to_nuscenes.py`
invocations, not the symlink shortcut from §2. Result: 0 failures; exactly 5 MLflow runs, each
carrying both eval metrics (from the batch eval step) and viz metrics (`viz_precision`/
`viz_recall`, from the visualize step) — confirming `get_or_create_run` enriched the same run
rather than creating a duplicate. The eval numbers matched Phase 4's originally-published
single-sequence table almost exactly:

| Combo | This test (fine/groups) | Phase 4 §7 (fine/groups) |
|---|---|---|
| `i2i_late` | 0.149 / 0.263 | 0.149 / 0.263 |
| `v2v_late` | 0.249 / 0.304 | 0.249 / 0.304 |
| `v2i_v1_late` | 0.200 / 0.306 | 0.200 / 0.306 |
| `v2i_v2_late` | 0.240 / 0.332 | 0.239 / 0.332 |
| `full_late` | 0.225 / 0.326 | 0.224 / 0.325 |

(The `v2i_v2_late`/`full_late` 0.001 deltas are the same GPU-nondeterminism scale Phase 3 §9
already documented — not a regression.) IoU=0.3/0.5 values matched Phase 4 §9.3 equally closely.
This confirmed the deduped batch pipeline reproduces the original per-combo pipeline's numbers on
real data before launching the full sweep.

## 4. Full 34-sequence sweep — results

Launched after the preliminary test passed. **0 failures**, ~2h44m wall-clock (10:28→13:32 local;
well under Phase 3's early-fusion sweep, ~5.2h, because compute-sharing cuts per-sequence
inference from up to 22 source-passes to at most 6). Verified via MLflow query: exactly **170
unique runs** (34 sequences × 5 combos), all carrying the full IoU-sweep metric set
(`mAP_fine_classes[_iouT]`/`mAP_groups[_iouT]` for T∈{0.3, 0.5}, 0 missing).

Per-intersection × per-combo mean±std mAP, via `aggregate_full_dataset_results_late_fusion.py`:

### IoU = 0.25

| Intersection | Combo | N | mAP (fine) | mAP (groups) |
|---|---|---|---|---|
| crossing1 | full_late | 17 | 0.333 ± 0.064 | 0.414 ± 0.065 |
| crossing1 | i2i_late | 17 | 0.218 ± 0.051 | 0.256 ± 0.061 |
| crossing1 | v2i_v1_late | 17 | 0.276 ± 0.055 | 0.352 ± 0.060 |
| crossing1 | v2i_v2_late | 17 | 0.286 ± 0.073 | 0.351 ± 0.080 |
| crossing1 | **v2v_late** | 17 | **0.362 ± 0.081** | **0.447 ± 0.078** |
| crossing2 | full_late | 10 | 0.236 ± 0.039 | 0.319 ± 0.037 |
| crossing2 | i2i_late | 10 | 0.149 ± 0.055 | 0.193 ± 0.047 |
| crossing2 | v2i_v1_late | 10 | 0.195 ± 0.044 | 0.262 ± 0.034 |
| crossing2 | v2i_v2_late | 10 | 0.228 ± 0.049 | 0.291 ± 0.046 |
| crossing2 | **v2v_late** | 10 | **0.280 ± 0.041** | **0.373 ± 0.046** |
| crossing3 | full_late | 7 | 0.286 ± 0.068 | 0.347 ± 0.056 |
| crossing3 | i2i_late | 7 | 0.154 ± 0.040 | 0.226 ± 0.029 |
| crossing3 | v2i_v1_late | 7 | 0.253 ± 0.070 | 0.311 ± 0.061 |
| crossing3 | v2i_v2_late | 7 | 0.215 ± 0.067 | 0.280 ± 0.063 |
| crossing3 | **v2v_late** | 7 | **0.320 ± 0.079** | **0.376 ± 0.071** |

### IoU = 0.3

| Intersection | Combo | N | mAP (fine) | mAP (groups) |
|---|---|---|---|---|
| crossing1 | full_late | 17 | 0.297 ± 0.063 | 0.364 ± 0.060 |
| crossing1 | i2i_late | 17 | 0.179 ± 0.046 | 0.204 ± 0.053 |
| crossing1 | v2i_v1_late | 17 | 0.240 ± 0.051 | 0.304 ± 0.055 |
| crossing1 | v2i_v2_late | 17 | 0.248 ± 0.070 | 0.299 ± 0.072 |
| crossing1 | v2v_late | 17 | 0.328 ± 0.082 | 0.404 ± 0.074 |
| crossing2 | full_late | 10 | 0.205 ± 0.036 | 0.276 ± 0.037 |
| crossing2 | i2i_late | 10 | 0.119 ± 0.051 | 0.150 ± 0.042 |
| crossing2 | v2i_v1_late | 10 | 0.166 ± 0.043 | 0.220 ± 0.035 |
| crossing2 | v2i_v2_late | 10 | 0.197 ± 0.045 | 0.249 ± 0.043 |
| crossing2 | v2v_late | 10 | 0.251 ± 0.036 | 0.332 ± 0.039 |
| crossing3 | full_late | 7 | 0.266 ± 0.064 | 0.319 ± 0.055 |
| crossing3 | i2i_late | 7 | 0.136 ± 0.040 | 0.197 ± 0.029 |
| crossing3 | v2i_v1_late | 7 | 0.235 ± 0.067 | 0.285 ± 0.058 |
| crossing3 | v2i_v2_late | 7 | 0.195 ± 0.064 | 0.251 ± 0.061 |
| crossing3 | v2v_late | 7 | 0.300 ± 0.075 | 0.350 ± 0.072 |

### IoU = 0.5

| Intersection | Combo | N | mAP (fine) | mAP (groups) |
|---|---|---|---|---|
| crossing1 | full_late | 17 | 0.118 ± 0.038 | 0.115 ± 0.040 |
| crossing1 | i2i_late | 17 | 0.050 ± 0.022 | 0.049 ± 0.022 |
| crossing1 | v2i_v1_late | 17 | 0.085 ± 0.026 | 0.091 ± 0.036 |
| crossing1 | v2i_v2_late | 17 | 0.091 ± 0.041 | 0.087 ± 0.035 |
| crossing1 | v2v_late | 17 | 0.148 ± 0.053 | 0.150 ± 0.051 |
| crossing2 | full_late | 10 | 0.085 ± 0.023 | 0.105 ± 0.021 |
| crossing2 | i2i_late | 10 | 0.032 ± 0.034 | 0.030 ± 0.017 |
| crossing2 | v2i_v1_late | 10 | 0.061 ± 0.027 | 0.069 ± 0.017 |
| crossing2 | v2i_v2_late | 10 | 0.078 ± 0.027 | 0.088 ± 0.019 |
| crossing2 | v2v_late | 10 | 0.115 ± 0.027 | 0.139 ± 0.029 |
| crossing3 | full_late | 7 | 0.128 ± 0.024 | 0.141 ± 0.018 |
| crossing3 | i2i_late | 7 | 0.041 ± 0.016 | 0.064 ± 0.021 |
| crossing3 | v2i_v1_late | 7 | 0.101 ± 0.025 | 0.115 ± 0.022 |
| crossing3 | v2i_v2_late | 7 | 0.084 ± 0.029 | 0.103 ± 0.033 |
| crossing3 | v2v_late | 7 | 0.161 ± 0.026 | 0.169 ± 0.039 |

## 5. Findings

**`v2v_late` is the best-performing late-fusion combo at every intersection and every IoU
threshold** — contrary to the intuition that more sources (`full_late`, 6 sources) should win.
`i2i_late` (4 infra-only sources) is the weakest everywhere, same as Phase 4's single-sequence
finding and Phase 3's early-fusion finding. `full_late` lands consistently *second*, ahead of
either single-vehicle-plus-infra combo (`v2i_v1_late`/`v2i_v2_late`) but behind `v2v_late` — adding
the (individually weaker) infra detections on top of both vehicles doesn't help once the two
vehicle sources already cover the scene well, and the extra infra false positives WBF has to
absorb likely dilute precision. This reinforces Phase 4 §7's conclusion: late fusion trades
per-detection quality for coverage, and pooling more (lower-quality) sources isn't free — it's a
real trade-off per combo, not a monotonic "more sensors is always better."

**mAP degrades 60-75% from IoU=0.25→0.5, consistent with Phase 3/4's early-fusion/single-sequence
findings.** The qualitative combo ranking (`v2v_late` best, `i2i_late` worst, same ordering
everywhere) is stable across all 3 IoU thresholds — same practical lesson as Phase 3 §10: always
state the matching IoU threshold when reporting a benchmark number, since IoU=0.5 values read
roughly a third to a quarter of IoU=0.25 values here.

**Late fusion underperforms the matching early-fusion config across the board** (consistent with
Phase 4 §7's single-sequence finding, now confirmed dataset-wide) — e.g. crossing2 `v2v_late`
0.280 fine-mAP @0.25 vs. Phase 3's early-fusion `v2v` (v1ref/v2ref average) 0.295/0.335 — each
source's own single, sparser point cloud still loses to one forward pass over a richer
point-concatenated cloud, even when boxes are fused with WBF after the fact.

## 6. File manifest

**New files:**
- `tools/eval_urbaning_v2x_late_fusion_batch.py` — all 5 combos for one sequence, deduped
  per-source inference, IoU sweep via `compute_ap_sweep`
- `tools/urbaning_v2x/run_full_dataset_eval_late_fusion.py` — host orchestrator, late-fusion
  analogue of Phase 3's `run_full_dataset_eval.py`
- `tools/urbaning_v2x/aggregate_full_dataset_results_late_fusion.py` — per-IoU-threshold
  intersection×combo mean/std tables

**Modified files:**
- `tools/eval_urbaning_v2x_late_fusion.py` — `combo_sources_for`/`unique_sources` (intersection
  generalization), `run_all_sources_inference`/`fuse_combo_frames` (split out of
  `build_late_fusion_frames`, kept as a backward-compatible wrapper), `compute_ap_sweep` (shared
  IoU-sweep helper), `main()` gained `--extra_match_iou_threshs`
- `tools/visualize_urbaning_v2x_late_fusion.py` — optional `--data_root`/`--intersection`
  overrides (falls back to Phase 4's module-level `COMBOS` when omitted)

**New datasets (container paths under `/OV-SCAN/`):**
```
datasets/urbaning_v2x_late_fusion_batch/<sequence>/
  vehicle1_solo/, vehicle2_solo/, infra_solo_<channel>/   # deleted after eval except viz scenes
  eval_output/<combo>/per_class_ap.json
  visualizations/<combo>/*.png, match_summary.json        # viz scenes only
```

**MLflow:** new experiment `urbaning_v2x_late_fusion_full_dataset` (separate from Phase 4's
single-sequence `urbaning_v2x_late_fusion` and Phase 3's early-fusion `urbaning_v2x_full_dataset`),
170 runs (34 sequences × 5 combos), each tagged `fusion_type=<combo>`, `merge_strategy=late_wbf`,
`sequence`, `intersection`.

## 7. Natural next steps

Intermediate (BEV-feature) fusion remains the main motivated follow-up (Phase 4 §7/§9's
finding that late fusion trades quality for coverage, inverse of early fusion's trade-off, suggests
a learned fusion module could recover both). Also open: whether `full_late`'s per-infra-source WBF
weighting could be tuned (e.g. down-weighting infra relative to vehicle sources) to let it beat
`v2v_late` by better exploiting its extra coverage instead of being dragged down by it.
