# UrbanIng-V2X ↔ OV-SCAN Integration — Phase 2

**Status:** V2V (vehicle-to-vehicle) and V2I (vehicle-to-infrastructure) fusion added on top of
Phase 1's single-vehicle and I2I (infrastructure-to-infrastructure) pipelines, still pure zero-shot
inference (`pretrained/ov_scan_lidar.pth`, no retraining), plus systematic MLflow experiment
tracking across all 6 resulting configs. See `Urbaning_PHASE_1.md` for the converter/eval/visualize
foundations this builds on — this document only covers what's new.

**Sequence used throughout:** `20241126_0001_crossing2_00`, same as Phase 1.

---

## 1. What v2v/v2i add, and why the frame choice matters

Phase 1 had two fusion frames: a single vehicle's own raw sensor frame (`--ego`), and infra LiDARs
fused into a *recentered global* frame (`--lidars`, since infra sensors are static and have no
natural "ego" frame of their own). V2V and V2I both fuse a *moving* vehicle with another source, so
there's a real choice: recenter into a global frame like `--lidars` does, or express everything in
the vehicle's own moving sensor frame instead.

**Chosen: the vehicle's own moving sensor frame** (confirmed with the user before implementing).
Reason: `ov_scan_lidar.pth` was trained on nuScenes' `LIDAR_TOP`, which is always vehicle-centered —
staying in that same egocentric distribution keeps v2v/v2i comparable to the single-vehicle and v2v
results, and avoids introducing an out-of-distribution recentered-global input the checkpoint never
saw at that particular geometry. `--lidars`' global-frame approach was the only option there (no
vehicle to center on); it doesn't generalize to v2v/v2i as the *better* choice, just the only one
available for pure infra fusion.

## 2. `FusedEgoConverter` — the new converter mode

Added to `tools/urbaning_v2x/convert_to_nuscenes.py` as a third mode, `--fuse_ego`, alongside the
untouched `--ego` and `--lidars` (regression-tested byte-identical against Phase 1's stored
`--ego vehicle1` output before anything else was built on top).

**Core idea:** output everything in the *reference* vehicle's own raw sensor frame — the exact frame
`--ego` already emits — so `calibrated_sensor`/`ego_pose` need no new logic; only point loading
changes. Every other fused-in source is projected into that frame per keyframe:

```
v2v (other vehicle):  p_ref = inv(vTl_ref) @ inv(gTv_ref) @ gTv_other @ vTl_other @ p_other_local
v2i (infra lidar):    p_ref = inv(vTl_ref) @ inv(gTv_ref) @ gTl_infra @ p_infra_local
```

`gTv_ref`/`gTv_other` are looked up per-frame from `{vehicle}_state/`; `vTl_ref`/`vTl_other`/`gTl_infra`
are static, from `calibration.json`. Both a fused vehicle and fused infra lidars can be combined at
once (`--fuse_vehicle` + `--fuse_lidars` together) — not exercised by the 4 runs below, but free from
the same generalization.

**Track exclusion — the one design question that needed real thought:** only the *reference*
vehicle's own GT track is excluded (it can't detect itself; self-returns are already filtered by the
existing `norm > 1.0` check). A fused-in *other* vehicle (v2v) is **kept** as a valid GT target —
excluding it would remove exactly the detections v2v fusion exists to help with. This makes v2v/v2i
track handling a straightforward reuse of `--ego`'s existing single-`skip_track_id` exclusion, scoped
to whichever vehicle is `--fuse_ego`.

**Minor refactor while touching this file:** the raw `.npz` loading + self-return filter, and the
per-box point-counting logic, were each duplicated once already between `SequenceConverter` and
`InfraFusionConverter`; pulled both into shared module-level helpers (`load_raw_lidar_points`,
`transform_points`, `count_points_in_box`) rather than adding a third copy. `SequenceConverter`
now calls these same helpers — confirmed byte-identical `.pcd.bin` output and identical
`sample_annotation.json` values (translation/size/rotation/`num_lidar_pts`) against the original
Phase 1 conversion before proceeding.

**Sanity check before trusting the AP numbers:** the fraction of GT boxes with at least one hit point
jumped substantially with fusion — a direct, model-independent signal that the new transform math is
placing fused points correctly, not as noise:

| Config | GT boxes w/ ≥1 point (of 15,814 annotations) |
|---|---|
| single_vehicle (vehicle1) | 7,847 |
| v2v (vehicle1+vehicle2) | ~10,435 |
| v2i (vehicle+4 infra lidars) | ~13,700 |

Sample visualizations for both a v2v and a v2i frame were also inspected directly: predicted boxes
align with GT outlines, and the point cloud geometry (road/sidewalk lines, building outlines) is
coherent with no misaligned/ghosted structure — see
`output/nuscenes_models/ov_scan_lidar_urbaning_v2v_v1ref/default/visualizations/` and the
`..._v2i_vehicle1/...` equivalent.

## 3. New CLI usage

```bash
# v2v: vehicle1 as reference, vehicle2's lidar fused in
python convert_to_nuscenes.py --root_folder /OV-SCAN/datasets/urbaning_v2x_raw \
    --sequence 20241126_0001_crossing2_00 --fuse_ego vehicle1 --fuse_vehicle vehicle2 \
    --out /OV-SCAN/datasets/urbaning_v2x_v2v_v1ref

# v2i: vehicle1 as reference, all 4 infra lidars fused in
python convert_to_nuscenes.py --root_folder /OV-SCAN/datasets/urbaning_v2x_raw \
    --sequence 20241126_0001_crossing2_00 --fuse_ego vehicle1 \
    --fuse_lidars crossing2_11_lidar,crossing2_12_lidar,crossing2_31_lidar,crossing2_32_lidar \
    --voxel_size 0.15 --out /OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1
```

Both v2v directions (`vehicle1` and `vehicle2` as reference) and both v2i vehicles were converted,
mirroring the established one-line-diff config pattern:

| Dataset | Dataset config | Model config |
|---|---|---|
| `urbaning_v2x_v2v_v1ref` | `urbaning_v2x_v2v_v1ref_dataset.yaml` | `ov_scan_lidar_urbaning_v2v_v1ref.yaml` |
| `urbaning_v2x_v2v_v2ref` | `urbaning_v2x_v2v_v2ref_dataset.yaml` | `ov_scan_lidar_urbaning_v2v_v2ref.yaml` |
| `urbaning_v2x_v2i_vehicle1` | `urbaning_v2x_v2i_vehicle1_dataset.yaml` | `ov_scan_lidar_urbaning_v2i_vehicle1.yaml` |
| `urbaning_v2x_v2i_vehicle2` | `urbaning_v2x_v2i_vehicle2_dataset.yaml` | `ov_scan_lidar_urbaning_v2i_vehicle2.yaml` |

`POINT_CLOUD_RANGE`/`VOXEL_SIZE`/`MODEL`/`ALIGNMENT` block kept byte-identical to Phase 1's pair —
no evidence of range truncation was found that would justify diverging (both vehicles stay within a
few tens of meters of each other for this sequence, well inside ±54 m).

**Raw data note:** vehicle2 and the 4 infra LiDAR/state folders had been deleted from
`datasets/urbaning_v2x_raw/` after Phase 1 (per its own storage-efficiency step). Re-extracted via
the same selective 7z technique from Phase 1 §1, against the source archive at
`~/Projects/CVDatasets/UrbanIng-V2X/dataset/20241126_0001_crossing2_00.7z.001` — note the archive's
internal paths have **no** `<sequence>/` prefix (unlike what §1's filelist example might suggest),
so the filelist entries need to be sensor-relative (`vehicle2_middle_lidar/<file>`, not
`20241126_0001_crossing2_00/vehicle2_middle_lidar/<file>`).

## 4. MLflow experiment tracking

**Setup:** backend store is a local SQLite database at `OV-SCAN/OV-SCAN/mlflow.db` (inside the docker
bind-mounted subtree, so it persists across container recreation and is host-visible).
`mlflow-skinny==3.15.1` added to `docker/OV-SCAN.Dockerfile` in its own `RUN pip install` step,
separate from the pinned dependency block above it — mlflow needs newer `numpy`/`protobuf` than that
block pins, and resolving both in one `pip install` call fails outright. Installed live in the
running container first and smoke-tested (model forward pass, CUDA IoU ops, CLIP path) before
touching the Dockerfile — no regression, just unrelated pip-check warnings against `wandb`/`tensorflow`
(neither is used by these scripts).

**File store → SQLite migration:** the first pass of this pipeline used a plain filesystem backend
(`mlruns/` + `MLFLOW_ALLOW_FILE_STORE=true`), which is enough for the Python client
(`mlflow.start_run`/`log_metric`/etc.) but `mlflow ui`/`mlflow server` refuse it outright — MLflow 3.x
deprecated the file store for the *server* specifically ("SQL-based tracking required for full
functionality"), even with the client-side opt-out set. Migrated losslessly via the bundled tool:

```bash
mlflow migrate-filestore --source /OV-SCAN/OV-SCAN --target sqlite:////OV-SCAN/OV-SCAN/mlflow.db --progress
```

(`--source` is the directory *containing* `mlruns/`, not `mlruns/` itself.) This moved only the
backend store (params/metrics/tags, previously scattered `meta.yaml` files) into the SQLite DB;
**artifacts stay exactly where they were**, as plain files under
`OV-SCAN/OV-SCAN/mlruns/<experiment_id>/<run_id>/artifacts/` — the backend store and artifact store
are decoupled in MLflow, migration only touches the former. `tools/mlflow_logging.py` was updated to
point `mlflow.set_tracking_uri` at the SQLite DB instead of the file store, and the
`MLFLOW_ALLOW_FILE_STORE` workaround was removed (no longer needed). Verified: all 6 runs' params/
metrics/tags survived the migration intact, new runs logged against the SQLite backend land in the
same pre-existing `urbaning_v2x` experiment (so new artifacts keep landing in the original `mlruns/`
location too, since an experiment's artifact root is fixed at creation time), and `mlflow ui` starts
and serves correctly against it.

**Browse:** `mlflow ui --backend-store-uri sqlite:////OV-SCAN/OV-SCAN/mlflow.db -h 0.0.0.0` from
inside the container (add `-p <port>` if 5000 is taken).

**Design:** new shared module `tools/mlflow_logging.py` (`add_common_args`, `run_name_for`,
`get_or_create_run`) used by both `eval_urbaning_v2x_per_class.py` and `visualize_urbaning_v2x.py`.
Each script takes `--fusion_type {single_vehicle,i2i,v2v,v2i}` and `--sources` (free text, e.g.
`vehicle1+vehicle2`) plus `--reference {vehicle1,vehicle2,none}`; the derived run name
(`<fusion_type>__<sources slug>`) is looked up via `mlflow.search_runs` and *resumed* if it already
exists. Since eval and visualize are separate process invocations, this lets both land in the
**same** MLflow run per experiment — AP/mAP, class-agnostic precision/recall, and a handful of
sample visualization PNGs (`--mlflow_max_artifacts`, default 5) all in one comparable row, rather
than fragmented across two runs.

## 5. Results across all 6 experiments

All 6 configs (Phase 1's 2 + Phase 2's 4 new) re-run through the MLflow-instrumented scripts,
200 frames, `match_iou_thresh=0.25`, visualize `score_thresh=0.3`:

| Run | fusion_type | reference | mAP (fine) | mAP (groups) | precision | recall |
|---|---|---|---|---|---|---|
| `single_vehicle__vehicle1` | single_vehicle | vehicle1 | 0.276 | 0.365 | 0.980 | 0.221 |
| `i2i__crossing2_11-12-31-32` | i2i | none | 0.218 | 0.331 | 0.987 | 0.032 |
| `v2v__vehicle1_vehicle2` | v2v | vehicle1 | 0.271 | 0.373 | 0.893 | 0.189 |
| `v2v__vehicle2_vehicle1` | v2v | vehicle2 | **0.305** | **0.395** | 0.912 | 0.164 |
| `v2i__vehicle1_...infra` | v2i | vehicle1 | 0.248 | 0.339 | **0.991** | 0.165 |
| `v2i__vehicle2_...infra` | v2i | vehicle2 | 0.285 | 0.356 | **0.991** | 0.175 |

**Reading these:** v2v (vehicle2-referenced) gives the best mAP on both axes, beating single-vehicle
outright — consistent with the point-coverage sanity check in §2 (more GT boxes get hit points, so
more objects are even detectable). v2i buys a large precision gain (0.991 vs. 0.980) at some mAP
cost relative to single-vehicle — plausible given the far denser, more cluttered fused point cloud
(4 extra sensors) the class-agnostic score threshold has to sort through. i2i remains the weakest
config on every axis, as in Phase 1 — no vehicle-mounted sensor at all means every detection has to
come from the infra viewpoint alone. As in Phase 1 §6, `precision`/`recall` here use a fixed
`score_thresh=0.3` and are **not** directly comparable across configs with different implicit score
distributions — mAP (rank-based) is the more reliable cross-config comparison metric; both are kept
per the original request to evaluate the same metrics as before.

## 6. File manifest (new in Phase 2)

| File | Purpose |
|---|---|
| `tools/urbaning_v2x/convert_to_nuscenes.py` | `FusedEgoConverter` + `--fuse_ego`/`--fuse_vehicle`/`--fuse_lidars` CLI, shared point-loading/counting helpers |
| `tools/mlflow_logging.py` | shared MLflow run/tagging/resume helper |
| `tools/eval_urbaning_v2x_per_class.py` | `--fusion_type`/`--sources`/`--reference`/`--mlflow_experiment`/`--run_name` args + metric/artifact logging |
| `tools/visualize_urbaning_v2x.py` | same args + `--mlflow_max_artifacts`, metric/artifact logging |
| `tools/cfgs/dataset_configs/urbaning_v2x_{v2v_v1ref,v2v_v2ref,v2i_vehicle1,v2i_vehicle2}_dataset.yaml` | new dataset configs |
| `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_{v2v_v1ref,v2v_v2ref,v2i_vehicle1,v2i_vehicle2}.yaml` | new model configs |
| `docker/OV-SCAN.Dockerfile` | `mlflow-skinny==3.15.1` pip install |
| `OV-SCAN/.gitignore` | `mlruns/`, `mlflow.db` |
