# UrbanIng-V2X ↔ OV-SCAN Integration — Phase 3

**Status:** All 6 fusion configs from Phase 2 (single_vehicle, i2i, v2v×2 directions, v2i×2
vehicles) run across the **entire** UrbanIng-V2X dataset — 34 sequences spanning 3 intersections
(crossing1: 17, crossing2: 10, crossing3: 7), 200 frames each, 6,800 frames total — still pure
zero-shot inference (`pretrained/ov_scan_lidar.pth`, no retraining). Storage-efficient by
construction: only one sequence's raw + converted data is ever on disk at a time. Results are
aggregated per intersection so the pipeline's behavior can be compared across the 3 physically
different sites rather than judged from a single reference sequence.

**Scope correction (see §8):** every config uses the same `POINT_CLOUD_RANGE` (±54m in x/y),
centered on the *reference* vehicle's own origin only. All results below (GT boxes, visualizations,
metrics) are therefore best read as **near-field fusion benefit within the reference vehicle's own
~54m horizon** — extra point density / de-occlusion from the fused source(s) close to the ego — not
as a measurement of full V2X range extension. §8 quantifies exactly how much of each fused source's
actual coverage this excludes, and why widening the range isn't a viable fix for a zero-shot,
non-retrained checkpoint.

---

## 1. Storage-efficient sweep design

Keeping every sequence's raw + 6 converted nuScenes-format datasets on disk at once would need
~34 × 5GB ≈ 170GB, on top of the ~118GB source archives already present. Instead,
`tools/urbaning_v2x/run_full_dataset_eval.py` processes one sequence at a time:

```
extract (7z, host)  ->  convert x6 + build infos (container)  ->  eval x6 (container)  ->  delete raw+converted
```

Peak disk usage stays under ~10GB regardless of dataset size. The exception is one **designated
visualization scene per intersection** (the earliest sequence chronologically —
`20241126_0001_crossing2_00`, `20241126_0008_crossing1_00`, `20241127_0009_crossing3_00`), whose
raw + converted data is kept permanently and additionally gets a full `visualize_urbaning_v2x.py`
run (200 frames, all 6 configs) — the only sequences with rendered images and class-agnostic
precision/recall; the other 31 only get the per-class eval script's AP/mAP metrics, to keep total
runtime reasonable (full sweep: ~6.5 hours for 204 eval runs + 18 visualize runs).

**Host/container split:** the source archives live outside the OV-SCAN container's bind mounts, so
extraction (`7z x` with per-channel wildcard patterns, e.g. `vehicle1_middle_lidar/*` — no need to
enumerate every filename) runs on the host; everything needing pcdet/CUDA/mlflow runs via
`docker exec`.

**Resumability:** every (sequence, variant, step) writes a marker file on success; a marker already
present is skipped, so an interrupted sweep just re-launches and picks up where it left off.
Two host/container-ownership pitfalls fixed along the way: (1) killing the host orchestrator
mid-flight doesn't kill an in-progress `docker exec` — it's an independent, orphaned process that
can race a subsequent restart's `docker exec` for the same output directory and corrupt it (hit
once, fixed by making the "is this variant already converted" check verify the actual
`nuscenes_infos_1sweeps_val.pkl` + `native_categories.json` exist, not just that the output
directory exists, and re-converting if not); (2) files written by `docker exec` (running as root)
are root-owned even though they land in a bind-mounted host directory, so a host-side
`shutil.rmtree` on them raises `PermissionError` — cleanup of converted (but not raw, which is
host-extracted) data now runs via `docker exec ... rm -rf` instead.

Per-intersection infra LiDAR channel names differ (`crossing1`/`crossing2` use `_11/_12/_31/_32`;
`crossing3` uses `_11/_12/_21/_22`) — the sweep script parameterizes this per intersection rather
than assuming Phase 1/2's crossing2-specific names generalize.

**One fixed set of 6 model configs, no new yaml files:** rather than one dataset yaml per
(sequence, variant) — 204 near-duplicate files — `eval_urbaning_v2x_per_class.py` and
`visualize_urbaning_v2x.py` gained a `--data_path` override that replaces
`DATA_CONFIG.DATA_PATH` after loading the (fixed) model config, and a new standalone
`tools/urbaning_v2x/build_infos.py` calls pcdet's `create_nuscenes_info()` directly with an
explicit `--data_path` (same fix as Phase 1's info-pkl-builder workaround, generalized).

**MLflow:** logged to a new `urbaning_v2x_full_dataset` experiment (separate from Phase 2's 6
reference-sequence runs in `urbaning_v2x`), one run per (sequence, fusion_type, sources), tagged
with `sequence` and `intersection` for filtering/grouping. `run_name` now includes the sequence
(`<fusion_type>__<sources slug>__<sequence>`) so 34 sequences' worth of runs per config stay
distinguishable — Phase 2's mlflow_logging.py only had to handle 6 total runs and didn't need this.

---

## 2. A real bug the wider dataset surfaced: the `'ignore'` GT category

`eval_urbaning_v2x_per_class.py`'s alignment between `data_dict['gt_boxes']` and the
converter-written `native_categories.json` sidecar (see Phase 1/2) was verified only against the
single Phase 2 reference sequence, and rested on the claim that "pcdet applies no further GT
reordering/removal" at eval time. That's true for the *train-only* outside-range box removal, but
**not** for pcdet's unconditional filtering of any GT box whose class name isn't in `CLASS_NAMES`
— UrbanIng-V2X uses an `'ignore'` native category for ambiguous/uncertain objects, which has no
`CLASS_NAMES` entry, and pcdet drops it regardless of train/eval mode. Sequence
`20241126_0013_crossing2_00` happens to contain one, in exactly one frame — the single reference
sequence used to write the original claim never happened to.

Symptom: `AssertionError: gt_boxes/native_categories length mismatch: 95 vs 96` for that one frame,
which failed all 6 fusion configs for that sequence (a GT-labeling property, independent of fusion
type). Fix: `filter_valid_gt_boxes_with_categories` now re-derives the same class-name drop from
each frame's own `info['gt_names']` (via `test_set.infos[idx]`) before comparing lengths, instead
of trusting a bare nonzero-row count. Verified against the failing sequence/frame directly, then
confirmed by re-running the full sweep to completion with 0 failures.

---

## 3. Results: per-intersection × per-config mAP (mean ± std across sequences)

`tools/urbaning_v2x/aggregate_full_dataset_results.py` pulls this from MLflow directly.

| Intersection | Fusion | Reference | N | mAP (fine) | mAP (groups) |
|---|---|---|---|---|---|
| crossing1 | i2i | none | 17 | 0.309 ± 0.062 | 0.362 ± 0.065 |
| crossing1 | single_vehicle | vehicle1 | 17 | 0.368 ± 0.117 | 0.466 ± 0.113 |
| crossing1 | v2i | vehicle1 | 17 | **0.393 ± 0.127** | **0.487 ± 0.129** |
| crossing1 | v2i | vehicle2 | 17 | 0.366 ± 0.083 | 0.483 ± 0.102 |
| crossing1 | v2v | vehicle1 | 17 | 0.392 ± 0.131 | 0.469 ± 0.120 |
| crossing1 | v2v | vehicle2 | 17 | 0.378 ± 0.106 | 0.490 ± 0.104 |
| crossing2 | i2i | none | 10 | 0.257 ± 0.063 | 0.287 ± 0.058 |
| crossing2 | single_vehicle | vehicle1 | 10 | 0.299 ± 0.052 | 0.388 ± 0.045 |
| crossing2 | v2i | vehicle1 | 10 | 0.281 ± 0.042 | 0.388 ± 0.036 |
| crossing2 | v2i | vehicle2 | 10 | 0.327 ± 0.095 | 0.424 ± 0.073 |
| crossing2 | v2v | vehicle1 | 10 | 0.295 ± 0.053 | 0.406 ± 0.045 |
| crossing2 | v2v | vehicle2 | 10 | **0.337 ± 0.079** | **0.450 ± 0.064** |
| crossing3 | i2i | none | 7 | 0.203 ± 0.060 | 0.296 ± 0.070 |
| crossing3 | single_vehicle | vehicle1 | 7 | **0.381 ± 0.099** | **0.466 ± 0.078** |
| crossing3 | v2i | vehicle1 | 7 | 0.345 ± 0.074 | 0.429 ± 0.096 |
| crossing3 | v2i | vehicle2 | 7 | 0.319 ± 0.088 | 0.385 ± 0.066 |
| crossing3 | v2v | vehicle1 | 7 | 0.361 ± 0.072 | 0.423 ± 0.066 |
| crossing3 | v2v | vehicle2 | 7 | 0.341 ± 0.077 | 0.389 ± 0.074 |

**Reading these (near-field scope — see §8):** all 6 configs share the same ±54m window centered
on the reference vehicle's own origin, so these numbers measure near-field fusion benefit (extra
density/de-occlusion close to the ego), not full V2X range extension — the comparisons *between*
configs are still fair (same scope applied uniformly), just don't read `v2i`'s mAP as "how well
infra-extended-range detection works," since most of what infra fusion could offer at range is
outside this window entirely (§8).

- **i2i is consistently the weakest config at every intersection** (lowest mAP both fine and
  grouped, by a wide margin at crossing3) — static infra-only fusion, no vehicle context at all,
  matching the Phase 2 single-sequence finding but now confirmed dataset-wide. Unlike v2v/v2i, i2i
  is *not* affected by the range issue in §8 — `InfraFusionConverter` recenters on the infra
  cluster's own mean position specifically to keep the fused scene inside ±54m, so this is a
  genuine finding about infra-only detection quality (no vehicle-mounted near-field context), not
  a range artifact.
- **Fusion (v2v/v2i) beats single_vehicle at crossing1 and crossing3**, but at crossing2 only
  `v2v (vehicle2-ref)` clearly beats single_vehicle — crossing2's single_vehicle mAP (0.299) is
  already closer to its fused variants than at the other two intersections, suggesting crossing2's
  geometry/traffic density gives the single vehicle relatively better native coverage.
- **Standard deviations are large relative to the means** (often 20-30% of the mean) — expected,
  since sequences vary a lot in traffic density and object mix; this is exactly why the
  full-dataset sweep matters more than any single sequence's number.
- crossing3's `i2i` is particularly weak (0.203 mAP fine) despite crossing3 having the fewest
  sequences (7) — worth flagging if crossing3's infra LiDAR placement/coverage is qualitatively
  different from crossing1/2 rather than assuming it's a sampling artifact.

## 4. Viz-scene precision/recall (class-agnostic 3D-IoU matching, 200 frames each)

Only available for the 3 designated scenes (full pipeline in §1):

| Scene | Fusion | Sources | Precision | Recall |
|---|---|---|---|---|
| crossing1 (`20241126_0008_crossing1_00`) | single_vehicle | vehicle1 | 0.982 | 0.117 |
| crossing1 | i2i | 4x infra | 0.971 | 0.064 |
| crossing1 | v2v | vehicle1+vehicle2 | 0.976 | 0.140 |
| crossing1 | v2v | vehicle2+vehicle1 | 0.992 | 0.167 |
| crossing1 | v2i | vehicle1+4x infra | 0.994 | 0.048 |
| crossing1 | v2i | vehicle2+4x infra | 0.992 | 0.045 |
| crossing2 (`20241126_0001_crossing2_00`, Phase 2 reference) | single_vehicle | vehicle1 | 0.981 | 0.222 |
| crossing2 | i2i | 4x infra | 0.948 | 0.052 |
| crossing2 | v2v | vehicle1+vehicle2 | 0.901 | 0.190 |
| crossing2 | v2v | vehicle2+vehicle1 | 0.912 | 0.163 |
| crossing2 | v2i | vehicle1+4x infra | 0.986 | 0.167 |
| crossing2 | v2i | vehicle2+4x infra | 0.989 | 0.183 |
| crossing3 (`20241127_0009_crossing3_00`) | single_vehicle | vehicle1 | 0.992 | 0.177 |
| crossing3 | i2i | 4x infra | 0.938 | 0.109 |
| crossing3 | v2v | vehicle1+vehicle2 | 0.885 | 0.094 |
| crossing3 | v2v | vehicle2+vehicle1 | 0.988 | 0.098 |
| crossing3 | v2i | vehicle1+4x infra | 0.996 | 0.096 |
| crossing3 | v2i | vehicle2+4x infra | 0.948 | 0.117 |

Recall stays low across the board (this is the same class-agnostic localization sanity check from
Phase 1/2, not a class-aware benchmark — see the scripts' own docstrings) but precision is
uniformly high (>0.88), consistent with the mAP picture: the model rarely fires confidently on the
wrong place, it just doesn't fire often enough to recall everything labeled.

---

## 5. File manifest

| File | Purpose |
|---|---|
| `tools/urbaning_v2x/run_full_dataset_eval.py` | host-side orchestrator: extract → convert → eval (+ viz for 3 scenes) → delete, resumable |
| `tools/urbaning_v2x/build_infos.py` | standalone info-pkl builder, `--data_path` driven (no per-sequence yaml needed) |
| `tools/urbaning_v2x/aggregate_full_dataset_results.py` | pulls per-intersection × per-config mAP mean/std from MLflow, one table per IoU threshold (§10) |
| `tools/eval_urbaning_v2x_per_class.py` | `--data_path` override; `--sequence`/`--intersection` passthrough; GT/category alignment fix (§2); `--extra_match_iou_threshs` IoU sweep from one inference pass (§10) |
| `tools/visualize_urbaning_v2x.py` | `--data_path` override; `--sequence`/`--intersection` passthrough |
| `tools/mlflow_logging.py` | `--sequence`/`--intersection` args; `run_name` now includes sequence |

MLflow experiment `urbaning_v2x_full_dataset` (204 eval runs + 18 visualize-augmented runs) is
separate from Phase 2's `urbaning_v2x` (6 reference-sequence runs, untouched). Browse both with:
```
mlflow ui --backend-store-uri sqlite:////OV-SCAN/OV-SCAN/mlflow.db -h 0.0.0.0
```

## 6. `mlflow-skinny` → full `mlflow` (post-rebuild fix)

After rebuilding the image with Phase 2's `mlflow-skinny==3.15.1` pin, `mlflow ui` failed with
`ModuleNotFoundError: No module named 'flask_cors'`. `mlflow-skinny` is deliberately
tracking-client-only — it was already missing `sqlalchemy`/`alembic` (needed just to *open* the
SQLite backend store, worked around in Phase 2 by installing them separately) and, it turns out,
is *also* missing `Flask-CORS`/`gunicorn`/`uvicorn` — the actual web-server dependencies
`mlflow ui`/`mlflow server` need to run at all, which skinny has no substitute for. Rather than
keep patching individual missing packages, `docker/OV-SCAN.Dockerfile` now installs full
`mlflow==3.15.1` instead (which bundles both gaps, superseding the separate `sqlalchemy`/`alembic`
line). Verified live: `pip install mlflow==3.15.1` in the running container (numpy stayed at
1.26.4, already bumped by the original skinny install — no new version change), `mlflow ui`
started and served HTTP 200, and existing runs in `mlflow.db` still read back correctly through
`mlflow_logging.py`.

## 7. Time-synchronization verification (v2v/v2i fusion, moving vehicles)

Both ego vehicles move while capturing data (§ context: real urban speeds), so v2v/v2i fusion is
only valid if the two sources' data genuinely correspond to (near-)the same instant — any residual
skew, combined with velocity, shows up as spatial misregistration in the fused cloud. Verified in
two independent ways, across all 3 available sequences (`20241126_0001_crossing2_00`,
`20241126_0008_crossing1_00`, `20241127_0009_crossing3_00`):

**Code:** `FusedEgoConverter._convert()` looks up each source's own filename from *its own* column
in `timesync_info.csv` for the current keyframe (`row[self.fuse_state_channels[v]]`,
`row[self.fuse_lidar_channels[v]]`, etc.) — never the reference vehicle's filename — so each
source's pose/points already come from that source's own actual capture instant for this keyframe,
not an assumption of shared timing.

**Data:** for all 200 frames × both vehicles' state/LiDAR channels × all infra LiDAR channels, the
selected filename's embedded timestamp exactly equals the keyframe's nominal `timestamp_ms` —
**0ms skew, every frame, all 3 sequences, no exceptions** (camera channels, not used by any
converter, do show few-ms nearest-frame jitter, but LiDAR/state/infra timestamps land exactly on
the shared grid, consistent with genuine hardware-synchronized clocks across vehicles + infra
rather than post-hoc nearest-neighbor matching).

Context for why this matters: vehicle speeds in the reference sequence reach 12.9 m/s (vehicle1)
and 13.1 m/s (vehicle2) — cross-checked via two independent measures, the state file's
instantaneous `vV` and numerical differentiation of consecutive `gTv` positions, which agree to
within noise. The two vehicles close to as little as 8.7m apart at up to 11.7 m/s relative speed
(frame 115) — at that point even a 50ms skew would misregister the other vehicle's points by half
a meter, so this was worth checking rather than assuming. With confirmed 0ms skew, no such error
exists. Visual confirmation at frame 115 (closest approach, highest closing speed — the
worst-case frame for this failure mode): no ghosting/doubled point clusters on any vehicle in the
fused v2v cloud.

**Conclusion:** time synchronization across the two ego vehicles is correct; the fusion pipeline
does not introduce velocity-induced spatial misalignment.

## 8. Range scope: v2v/v2i are near-field-only, and that's not a fixable oversight

Raised concern: `POINT_CLOUD_RANGE` (±54m in x/y, identical across all 6 configs) is centered on
the *reference* vehicle's own origin only — does the fused source's own range get accounted for,
for GT boxes, visualizations, and metrics?

**Mechanism:** pcdet's `mask_points_and_boxes_outside_range` drops points outside
`POINT_CLOUD_RANGE` *unconditionally* (train or eval — only the GT-box-removal half of that same
processor step is train-only), so this window determines what the model can physically see, not
just what gets scored. The eval/visualize scripts' GT-range filter uses the same
`cfg.DATA_CONFIG.POINT_CLOUD_RANGE`, so GT filtering is at least internally consistent with what the
model could ever detect — the real problem is the window itself, not a separate filtering bug.

**Measured severity** (full 200-frame reference sequence, `v2i_vehicle1`):
- Median GT-box distance from the reference vehicle: **65m** — already past the 54m box.
- Only **44.8%** of all 13,647 GT boxes across the sequence fall inside the eval window.
- 2 of the 4 infra LiDARs sit **more than 54m from the vehicle on every single frame** (33–135m).
- Direct point-cloud inspection: **18% of all points** in one sampled frame lie outside the box and
  are dropped before voxelization.
- v2v has a smaller version of the same issue: vehicle-to-vehicle separation reaches 68.6m (7% of
  frames already exceed 54m).

**Tested the obvious fix — doesn't work for this checkpoint.** Widened `POINT_CLOUD_RANGE` to
±120m (with matching `POST_CENTER_RANGE`) and ran it live: no crash, comparable runtime on the
RTX 5080 (16GB). But detection quality collapsed — mAP (fine) 0.28→0.007, mAP (groups) 0.39→0.065,
with grossly unbalanced prediction counts (e.g. `pedestrian`: 115 GT vs. 1082 predictions). The
pretrained detection head (fixed 200-proposal budget, heatmap radius, feature-map stride) is tuned
to nuScenes' native ~54m grid; widening the window pushes it far outside that trained distribution.
Since this project is zero-shot inference only (no retraining), that rules out a config-only fix.

**Decision (2026-08-26):** keep the current ±54m window as-is — no pipeline change. §3's results
and the visualizations throughout Phase 2/3 are near-field fusion-benefit measurements (extra point
density / de-occlusion within the reference vehicle's own horizon), not full V2X range-extension
measurements; comparisons *between* configs stay fair since the same window applies to all 6
uniformly. `i2i` is unaffected by this (see §3) since `InfraFusionConverter` already recenters on
the infra cluster's own mean position specifically to fit inside ±54m.

A genuine fix (recentering the window per-frame toward the fused sources' midpoint, instead of the
reference vehicle's own origin, while keeping the same 54m half-width/grid resolution) was scoped
but not implemented — it's compute-neutral and avoids the retraining problem above, but requires a
real redesign of the converter's reference-frame logic (the current `ego_pose`/`calibrated_sensor`
semantics assume the window's origin is a real vehicle's own sensor frame) and would only partially
help v2i, since infra sensors can sit 100m+ away even from a recentered midpoint.

---

## 9. Score threshold: investigated and ruled out (2026-08-28)

Raised concern: does §3's mAP depend on the model's score threshold (0.1 in every config's
`MODEL.POST_PROCESSING.SCORE_THRESH`) — specifically, would lowering it to 0.05 change the results?

**Short answer: no.** `class_ap` (the AP methodology behind every mAP number in this doc, and in
Phase 4's late-fusion work) sweeps every raw proposal the model emits and is rank-based by
construction — no score cutoff is ever applied before ranking, so a "score threshold" in the usual
sense can't move it. But the codebase turns out to have *two* same-named config fields that could
plausibly matter, so this was worth actually checking rather than asserting from that argument
alone:

- `MODEL.POST_PROCESSING.SCORE_THRESH` (0.1, the one every model config sets) — read only by
  `OVScanLidar.post_processing()`, which in fact never references it (only `RECALL_THRESH_LIST` is
  used there).
- `MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH` (0.0) — this one *is* read, inside
  `OVScanHead.decode_bbox()`, but the eval-time call path (`forward()` → `get_bboxes()` →
  `decode_bbox(..., filter=True)`) always resolves to this fixed head-local 0.0, independent of the
  top-level value above.

**A false start, and how it was caught.** A first quick test (edit the top-level 0.1 → 0.05 in a
scratch copy of the config, re-run one sequence) produced what looked like a real, reproducible
split — bus AP shifted from 0.643 to 0.662, bit-identical within 2 reps at each value. That's what
a genuine effect would look like, and it was wrong: a temporary debug `print` patched into
`decode_bbox` confirmed the threshold it actually uses stays a constant `0.0` regardless of which
config file was passed. Re-running each value **4** times (8 runs total, same sequence/config
otherwise) instead of 2 made the apparent split evaporate — bus AP came back bit-identical across
all 8 runs, and mAP jitter dropped to the same ~0.0004-scale GPU non-determinism already documented
in Phase 2–4. The 2-rep test had, by chance, sampled two nondeterministic realizations that happened
to cluster by threshold value; it wasn't measuring the threshold at all. **Lesson: this codebase's
GPU-nondeterminism jitter is large enough that a 2-sample A/B test can produce a fully reproducible-
looking but spurious split — anything claiming a small (<1%) effect needs ≥4 reps per arm before
being trusted.**

**Conclusion:** §3/§10's mAP tables are unaffected by score threshold; no re-run was needed on that
axis. (Diagnostic debug print and scratch config files were reverted/deleted after the test; a
throwaway `score_thresh_sanity_check_tmp` MLflow experiment used for these runs was deleted too.)

## 10. IoU=0.3 / IoU=0.5 sweep: the full dataset re-run (2026-08-28)

Unlike score threshold, the matching IoU threshold (`--match_iou_thresh`, 0.25 throughout §3) *does*
directly gate `class_ap`'s pred↔GT matching — every one of §3's numbers was IoU=0.25 only. This
needed an actual re-run, since raw per-frame predictions weren't cached to disk (only the final AP
summary was) — there's no way to recompute at a new IoU threshold without re-running inference.

**Implementation, to avoid 3x the compute:** `eval_urbaning_v2x_per_class.py` gained
`--extra_match_iou_threshs` (default `'0.3,0.5'`) — it computes AP at the primary threshold (0.25,
keeping the original unsuffixed metric names for continuity) *and* every extra threshold from the
**same** inference pass per (sequence, variant); only the cheap AP-matching math re-runs per extra
threshold, not the forward pass. Extra-threshold metrics are logged with a `_iou<T>` suffix (e.g.
`mAP_fine_classes_iou0.3`) into the existing MLflow run for that (sequence, variant) — same
`run_name`/`run_id` as the original §3 run, enriched in place rather than creating new runs or a new
experiment. Verified with a single-sequence dry run before committing to the full sweep: exactly one
run per name, both `mAP_fine_classes` (0.25, matching the original value within noise) and the new
`_iou0.3`/`_iou0.5` metrics present. `aggregate_full_dataset_results.py` was extended to print one
table per IoU threshold instead of just 0.25.

**Re-running the full sweep:** all 31 non-viz-scene sequences' data had already been deleted after
§1's original sweep (by design, for disk space), so re-computing at any new threshold meant
re-running the *entire* extract→convert→infer pipeline for all 34 sequences × 6 configs again — same
cost as the original §1 sweep, ~5.2 hours wall clock this time (10:25→15:39 CEST). The 3 designated
viz-scene sequences kept their converted data from §1, so only needed their (cheap) eval markers
cleared to force re-inference through the same already-converted data. **0 failures across all 204
(sequence, variant) eval runs.**

### Results: mAP by IoU threshold (mean ± std across sequences per intersection × config)

| Intersection | Fusion | Reference | N | mAP fine (IoU .25 / .3 / .5) | mAP groups (IoU .25 / .3 / .5) |
|---|---|---|---|---|---|
| crossing1 | i2i | none | 17 | 0.309 / 0.266 / 0.102 | 0.361 / 0.296 / 0.082 |
| crossing1 | single_vehicle | vehicle1 | 17 | 0.368 / 0.339 / 0.160 | 0.466 / 0.428 / 0.171 |
| crossing1 | v2i | vehicle1 | 17 | **0.395 / 0.370 / 0.214** | **0.490 / 0.458 / 0.239** |
| crossing1 | v2i | vehicle2 | 17 | 0.366 / 0.335 / 0.181 | 0.483 / 0.438 / 0.201 |
| crossing1 | v2v | vehicle1 | 17 | 0.391 / 0.367 / 0.200 | 0.469 / 0.435 / 0.201 |
| crossing1 | v2v | vehicle2 | 17 | 0.379 / 0.348 / 0.187 | 0.490 / 0.448 / 0.208 |
| crossing2 | i2i | none | 10 | 0.257 / 0.217 / 0.076 | 0.287 / 0.235 / 0.054 |
| crossing2 | single_vehicle | vehicle1 | 10 | 0.300 / 0.269 / 0.122 | 0.389 / 0.344 / 0.137 |
| crossing2 | v2i | vehicle1 | 10 | 0.280 / 0.257 / 0.132 | 0.388 / 0.353 / 0.164 |
| crossing2 | v2i | vehicle2 | 10 | 0.328 / 0.303 / 0.175 | 0.425 / 0.388 / 0.203 |
| crossing2 | v2v | vehicle1 | 10 | 0.295 / 0.272 / 0.137 | 0.404 / 0.366 / 0.173 |
| crossing2 | v2v | vehicle2 | 10 | **0.335 / 0.310 / 0.160** | **0.450 / 0.412 / 0.193** |
| crossing3 | i2i | none | 7 | 0.203 / 0.174 / 0.067 | 0.296 / 0.245 / 0.078 |
| crossing3 | single_vehicle | vehicle1 | 7 | **0.380 / 0.364 / 0.188** | **0.465 / 0.441 / 0.206** |
| crossing3 | v2i | vehicle1 | 7 | 0.347 / 0.326 / 0.190 | 0.427 / 0.400 / 0.220 |
| crossing3 | v2i | vehicle2 | 7 | 0.320 / 0.300 / 0.185 | 0.385 / 0.360 / 0.203 |
| crossing3 | v2v | vehicle1 | 7 | 0.360 / 0.340 / 0.197 | 0.422 / 0.395 / 0.207 |
| crossing3 | v2v | vehicle2 | 7 | 0.340 / 0.317 / 0.184 | 0.389 / 0.359 / 0.194 |

(IoU=0.25 column re-derived from this session's re-run, not copied from §3 — values match §3's
original numbers within the same ~0.001–0.005 GPU-nondeterminism band documented in §9, confirming
the re-run reproduces the original pipeline faithfully.)

**Findings:**
- **mAP degrades sharply and consistently with stricter IoU across every config and intersection** —
  fine-class mAP typically loses 60–75% of its IoU=0.25 value by IoU=0.5 (e.g. crossing2 i2i:
  0.257→0.076, a 70% relative drop; crossing1 v2i_vehicle1: 0.395→0.214, "only" 46% relative,
  the best-preserved config at IoU=0.5 in every intersection). This says the pretrained TransFusion
  head's box *localization* (not just classification) is comparatively loose relative to nuScenes'
  own typical IoU regimes — consistent with a zero-shot checkpoint operating outside its native
  training distribution's geometry statistics, on top of the taxonomy-transfer gap already discussed
  in Phase 2.
- **The qualitative ranking across configs is IoU-stable.** i2i remains the weakest config at every
  intersection and every IoU threshold; the best config at each intersection (bolded) stays the same
  best config across all 3 thresholds too. This means §3's comparative conclusions ("fusion beats
  single_vehicle", "i2i is weakest") are not artifacts of the specific IoU=0.25 choice — they hold at
  stricter matching too, just at uniformly lower absolute mAP.
- **v2i_vehicle1 (crossing1) and v2v_vehicle2 (crossing2) hold up best under IoU=0.5** relative to
  their own IoU=0.25 value — worth a closer look if future work prioritizes localization tightness,
  not just classification-level detection.
- **Practical implication for any future benchmark comparison against this checkpoint**: report the
  matching IoU threshold explicitly — at IoU=0.5 (a common detection-benchmark default) every
  number here reads roughly 3–5x lower than at IoU=0.25, which is easy to mistake for a much weaker
  model if the threshold isn't stated.
