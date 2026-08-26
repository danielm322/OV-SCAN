# UrbanIng-V2X ↔ OV-SCAN Integration — Phase 1

**Status:** Working end-to-end inference pipeline (no retraining) for both a single vehicle-mounted
LiDAR and a fused multi-sensor infrastructure LiDAR, evaluated with class-aware AP/mAP using
OV-SCAN's open-vocabulary head, plus a set of zero-shot fixes that materially improved AP on the
weakest classes (pedestrian, two-wheelers).

**Goal of this document:** a technical reference detailed enough to resume or extend this work
without re-deriving the pipeline from scratch — every non-obvious decision, the bugs that were
found and why they mattered, and the exact mechanics of the open-vocabulary exploitation.

**Checkpoint used throughout:** `pretrained/ov_scan_lidar.pth` (the stock nuScenes-trained
OV-SCAN LiDAR-only checkpoint — never fine-tuned on UrbanIng-V2X).

**Sequence used throughout:** `20241126_0001_crossing2_00` (one intersection sequence, 200
synchronized 10 Hz keyframes).

---

## 1. Data source

UrbanIng-V2X ships as one `.7z` split-archive per sequence
(`CVDatasets/UrbanIng-V2X/dataset/<sequence>.7z.00{1,2}`), each bundling **every** sensor for
that sequence: two vehicles' cameras + LiDAR + state, and several fixed infrastructure LiDARs +
thermal cameras at the crossing. A sequence archive is 2.5–3.5 GB compressed; extracting it whole
is wasteful when only one or two sensors are needed.

Per-sequence layout after extraction:

```
dataset/<sequence>/
  calibration.json        # per-sensor extrinsics (+ intrinsics for cameras)
  timesync_info.csv       # 200 synchronized keyframe columns; one row per sensor channel,
                           # cell = that sensor's filename for that keyframe
  vehicle1_middle_lidar/   vehicle1_state/   vehicle2_middle_lidar/   vehicle2_state/
  crossing2_11_lidar/  crossing2_12_lidar/  crossing2_31_lidar/  crossing2_32_lidar/   (infra)
  vehicle{1,2}_*_camera/   crossing2_*_thermal_camera/
labels/<sequence>.json    # per-track 3D boxes (position, orientation, dimension per timestamp)
labels_av_track_ids.json  # {sequence: {vehicle1: track_id, vehicle2: track_id}}
```

**LiDAR file format:** each `.npz` holds separate 1-D arrays `x, y, z` (float32),
`intensity` (uint8), `time_offset_ms` (uint8) — points are in the **sensor's own local frame**.

**Calibration:** each sensor's `calibration.json` entry has an `extrinsics` dict with *either*:
- `vTl` (vehicle-mounted sensors): lidar → vehicle-body transform (4×4), or
- `gTl` (fixed infrastructure sensors): lidar → **global/map** transform (4×4), **static** —
  infra sensors don't move, so there's no per-frame pose to track for them.

`vehicle{1,2}_state/<file>.json` holds `gTv`: vehicle-body → global transform, **one file per
keyframe** (the vehicle moves).

**Labels:** `labels/<sequence>.json` has `tracks: [{track_id, object_type, timestamps[],
positions[], orientations[], dimensions[]}]`. Positions/orientations are given directly in the
**global/map frame** — no transform needed to use them as-is. `object_type` is one of: `Car, Van,
Bus, Truck, OtherVehicle, Trailer, EScooter, Motorcycle, Cyclist, Pedestrian, OtherPedestrians,
Animal, Other`.

### Selective extraction (storage efficiency)

Because `timesync_info.csv` gives the exact filename each sensor needs *per keyframe*, we never
extract a sensor's full raw-rate folder (~2× more files than keyframes, since sensors run faster
than the 10 Hz sync rate). Instead, build an explicit file list and hand it to `7z x @listfile`
(the archives are **non-solid**, so partial extraction doesn't decompress the whole blob):

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('<seq>/timesync_info.csv').set_index('Unnamed: 0')
with open('filelist.txt', 'w') as f:
    for sensor in ['crossing2_11_lidar', 'crossing2_12_lidar']:  # whichever sensors you need
        for col in df.columns:
            f.write(f'{sensor}/{df.loc[sensor, col]}\n')
"
7z x -y -o<dest> <sequence>.7z.001 "@filelist.txt"
```

For 4 infra LiDARs at 200 keyframes each, this extracts ~248 MB instead of the ~700 MB+ a full
per-sensor folder extraction would cost, and a small fraction of the sequence's multi-GB archive.
**Delete the raw `.npz` after conversion** (see §3) — only the compact converted output needs to
persist.

---

## 2. Why convert to nuScenes format instead of writing a new pcdet Dataset class

OV-SCAN's `NuScenesOVDataset` (`pcdet/datasets/nuscenes/nuscenes_ov_dataset.py`) already
implements everything needed: multi-sweep LiDAR loading, GT box loading/filtering, the
`DataProcessor`/voxelization pipeline, and — critically — is what the pretrained checkpoint's
`POINT_CLOUD_RANGE`, voxel sizes, and class taxonomy were built around. Writing a
UrbanIng-V2X-native `Dataset` class would mean re-implementing all of that. Converting
UrbanIng-V2X into nuscenes-devkit's on-disk format instead means the *existing, unmodified*
`NuScenesOVDataset` + info-pkl builder can consume it directly.

**On-disk convention** (reverse-engineered from the real nuScenes-mini install, since parts of
pcdet's own nuScenes info-builder are internally inconsistent about it — see §4):

```
<DATA_PATH>/<version>/*.json        # nuscenes-devkit tables (category, sample, sample_data, ...)
<DATA_PATH>/<version>/samples -> ../samples   # symlink
<DATA_PATH>/samples/LIDAR_TOP/*.pcd.bin       # raw point clouds, sibling of <version>/
<DATA_PATH>/<version>/nuscenes_infos_1sweeps_{train,val}.pkl   # pcdet's own derived info-pkl
```

`NuScenes(version, dataroot)` computes `table_root = dataroot/version` internally and expects raw
sensor files at `dataroot/samples/<channel>/...` — hence the symlink bridging the two.

pcdet's nuScenes info-pkl builder **hardcodes the LiDAR channel name as `'LIDAR_TOP'`** in
`fill_trainval_infos` (both `ref_chan` and `chan`), so the converter always writes to
`samples/LIDAR_TOP/`, whatever the UrbanIng-V2X source sensor actually was.

---

## 3. The converter — `tools/urbaning_v2x/convert_to_nuscenes.py`

One converter, two mutually exclusive modes, sharing a `BaseNuscenesConverter` for
category/instance bookkeeping and json-table writing.

### 3.1 `--ego {vehicle1,vehicle2}` — single vehicle-mounted LiDAR

Closest match to nuScenes' own single ego-vehicle-mounted `LIDAR_TOP`, so it's the fairest
zero-shot test of the pretrained checkpoint.

- Points are written **exactly as captured, in the LiDAR's own sensor frame** — no extrinsic
  applied. This matches nuScenes convention: pcdet loads the *current* frame's points with zero
  transform (only past sweeps in a multi-sweep window get transformed), and instead reprojects
  GLOBAL-frame GT boxes into that same sensor frame via `calibrated_sensor` (sensor→ego, from
  `vTl`) composed with `ego_pose` (ego→global, from `gTv`, one entry per keyframe). Points and
  boxes end up aligned without any manual point-cloud transform in the converter.
- A point-radius filter `norm(xyz) > 1.0` strips the vehicle's own body/mount self-returns.
- The ego vehicle's own label track is excluded (`labels_av_track_ids.json`) — it can't detect
  itself.
- `MAX_SWEEPS: 1` (single-sweep only) → `timestamp` column is written as `0.0` for all points.

```python
# tools/urbaning_v2x/convert_to_nuscenes.py — per-frame point loading (vehicle mode)
lidar_file = self.seq_dir / self.lidar_channel / row[self.lidar_channel]
raw = np.load(lidar_file)
points = np.stack([raw['x'], raw['y'], raw['z'], raw['intensity']], axis=1).astype(np.float32)
points = points[np.linalg.norm(points[:, :3], axis=1) > 1.0]
```

### 3.2 `--lidars <comma-separated channel names>` — infrastructure LiDAR fusion

Fuses one or more **static** infra LiDARs into a single per-frame point cloud. Config-driven
sensor selection (any subset of a sequence's infra sensors), not hardcoded.

Since infra sensors are fixed, each has a static `gTl` (lidar→global). Points from every selected
sensor are transformed straight into the global frame and concatenated — **no per-frame ego
motion to track at all**, unlike the vehicle case:

```python
def _load_lidar_points_global(self, chan, filename):
    raw = np.load(self.seq_dir / chan / filename)
    points = np.stack([raw['x'], raw['y'], raw['z'], raw['intensity']], axis=1).astype(np.float32)
    points = points[np.linalg.norm(points[:, :3], axis=1) > 1.0]
    gTl = self.gTl_by_lidar[chan]
    points_h = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    points_global = (gTl @ points_h.T).T[:, :3]
    return np.concatenate([points_global - self.offset, points[:, 3:4]], axis=1).astype(np.float32)
```

Both points and GT boxes (GT is already stored in the global frame — see §1) are then
**recentered** by subtracting `self.offset = mean(gTl[:3,3] for each selected sensor)`, so the
fused scene sits inside OV-SCAN's ±54 m point-cloud range regardless of where the map's raw
coordinate origin happens to be (infra LiDAR range can span up to ~100 m from a single sensor, so
this matters — see §7 for the discovery of why this recentering was necessary).
`calibrated_sensor`/`ego_pose` are written as **identity** — the recentered global frame *is* the
"ego" frame here.

**No label track is excluded** in this mode: both vehicles are legitimate detection targets from
an infrastructure viewpoint (unlike the vehicle mode, where the ego vehicle can't see itself).

**Point density / voxel downsampling:** 4 fused infra LiDARs give ~120–160k points/frame, about
4–5× denser than nuScenes' `LIDAR_TOP` (~30–40k), which the checkpoint was trained on. A
dependency-free voxel-grid mean-pool (`--voxel_size`, default targets ~35k pts/frame — `0.3` m
worked well for 4 sensors) brings density back in line, both for domain-gap and storage reasons:

```python
def voxel_downsample(points, voxel_size):
    """Mean-pool points falling in the same voxel. points: (N, 4) xyz+intensity."""
    if not voxel_size or voxel_size <= 0 or len(points) == 0:
        return points
    voxel_idx = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    df = pd.DataFrame({'vx': voxel_idx[:, 0], 'vy': voxel_idx[:, 1], 'vz': voxel_idx[:, 2],
                        'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2], 'i': points[:, 3]})
    agg = df.groupby(['vx', 'vy', 'vz'], sort=False).mean()
    return agg[['x', 'y', 'z', 'i']].to_numpy().astype(np.float32)
```

### 3.3 Category mapping

UrbanIng-V2X's fine-grained `object_type` is collapsed to nuScenes' 10-class detection taxonomy
(needed because pcdet's GT loading/eval machinery, and the model's 10-channel heatmap head, are
hard-coded to that taxonomy):

```python
OBJECT_TYPE_TO_NUSCENES = {
    "Car": "vehicle.car", "Van": "vehicle.car",
    "Bus": "vehicle.truck", "Truck": "vehicle.truck", "OtherVehicle": "vehicle.truck", "Trailer": "vehicle.truck",
    "EScooter": "vehicle.bicycle", "Motorcycle": "vehicle.bicycle", "Cyclist": "vehicle.bicycle",
    "Pedestrian": "human.pedestrian.adult", "OtherPedestrians": "human.pedestrian.adult",
    "Animal": "animal", "Other": "movable_object.barrier",
}
```

### 3.4 `native_categories.json` sidecar — recovering fine-grained GT for per-class eval

Coarse-only GT is a real limitation once you want per-class metrics against UrbanIng-V2X's *own*
taxonomy (see §6). The converter also writes a sidecar mapping `sample_token -> [object_type, ...]`
so the original fine-grained category survives alongside the coarse one, **in the exact
order/subset pcdet's own pipeline keeps GT boxes in** — this alignment is not incidental, it's
load-bearing, and worth spelling out because it's easy to get subtly wrong:

1. nuscenes-devkit's reverse index (`NuScenes.__make_reverse_index__`) builds each sample's
   `anns` list by scanning the **entire `sample_annotation` table in file order** and appending
   matches — i.e., `sample_annotation.json` insertion order determines everything downstream.
2. The converter appends to `sample_annotation_table` *and*, in the same loop iteration, to the
   `native_categories_by_sample` sidecar — so both lists share the exact same per-frame ordering
   by construction.
3. pcdet's info-pkl builder (`nuscenes_utils.fill_trainval_infos`) keeps only annotations with
   `num_lidar_pts + num_radar_pts > 0` (`mask = (num_lidar_pts + num_radar_pts > 0)`), preserving
   order. The converter mirrors that *exact* condition when deciding whether to append to the
   sidecar:

```python
# tools/urbaning_v2x/convert_to_nuscenes.py — inside the per-object annotation loop
if num_lidar_pts > 0:
    self.native_categories_by_sample.setdefault(sample_token, []).append(obj['object_type'])
```

4. At eval time (`training=False`), pcdet applies no further GT reordering/removal: the
   outside-range box removal in `mask_points_and_boxes_outside_range` only fires `and
   self.training` (confirmed by reading `pcdet/datasets/processor/data_processor.py`), and
   `FILTER_MIN_POINTS_IN_GT` re-applies the same already-satisfied `num_lidar_pts > 0` condition
   a second time, a no-op.

**Verified empirically**, not just by code reading: for all 200 samples in both converted
datasets, `len(sidecar[sample_token]) == len(info['gt_boxes'])`, with matching per-box coarse
category (`gt_names[i]` derived from `native[i]` via the mapping in §3.3) — see the sanity-check
snippet in `docs/Urbaning/` conversation history if this needs to be re-verified after any pcdet
pipeline change; the check is cheap (a `pickle.load` + `json.load` + a length/name comparison
loop).

### 3.5 CLI

```bash
# single vehicle LiDAR
python convert_to_nuscenes.py \
    --root_folder /OV-SCAN/datasets/urbaning_v2x_raw \
    --sequence 20241126_0001_crossing2_00 \
    --ego vehicle1 \
    --out /OV-SCAN/datasets/urbaning_v2x

# infrastructure-fused LiDAR (any subset of the sequence's infra sensors)
python convert_to_nuscenes.py \
    --root_folder /OV-SCAN/datasets/urbaning_v2x_raw \
    --sequence 20241126_0001_crossing2_00 \
    --lidars crossing2_11_lidar,crossing2_12_lidar,crossing2_31_lidar,crossing2_32_lidar \
    --voxel_size 0.3 \
    --out /OV-SCAN/datasets/urbaning_v2x_infra
```

`--root_folder` must contain `dataset/<sequence>/...`, `labels/<sequence>.json`, and (vehicle mode
only) `labels_av_track_ids.json`.

---

## 4. Bugs found and fixed in pcdet's stock nuScenes pipeline

None of these are UrbanIng-V2X-specific — they're latent bugs in
`pcdet/datasets/nuscenes/nuscenes_dataset.py` / `nuscenes_utils.py` that only surface for a
non-official/converted dataset. Documented here because they'll bite again for any future
third-party nuScenes-format conversion.

**Bug 1 — official-scene-name split filtering silently drops everything.**
`create_nuscenes_info` filtered scenes against nuscenes-devkit's hardcoded `splits.train` /
`splits.val` (real nuScenes scene names). A converted dataset's scene names never match those,
so `train_scenes`/`val_scenes` end up empty and 0 samples get built. Fix: recognize a
`v1.0-custom` version that instead treats every available scene as `val`:

```python
elif version == 'v1.0-custom':
    train_scenes = []
    val_scenes = None   # resolved to every available scene name once NuScenes() has loaded them
```

**Bug 2 — unconditional camera lookup breaks camera-less datasets.**
`fill_trainval_infos` unconditionally read `sample['data']['CAM_FRONT']` even when
`with_cam=False`. Fixed by moving that block inside the existing `if with_cam:` guard — a one-line
fix, but silent `KeyError` otherwise for any LiDAR-only dataset.

**Bug 3 — double-suffixed dataroot.**
`create_nuscenes_info` reassigned `data_path = data_path / version` before calling
`NuScenes(dataroot=data_path)` — but `NuScenes()` **already** appends `/version` internally
(`table_root = dataroot/version`), producing a double-suffixed path
(`.../v1.0-custom/v1.0-custom`) and a "Database version not found" error. Root-caused by
cross-checking against a real, working nuscenes-mini info-pkl's stored `lidar_path`
(`samples/LIDAR_TOP/...`, relative to the **un-suffixed** root). Fix: only `save_path` gets
`/version` appended (matching where the real `.pkl` lives); `data_path` stays un-suffixed
throughout, matching how stored relative paths are interpreted.

**Practical note:** because of pcdet's `__main__` argparse block hardcoding
`data_path=ROOT_DIR/'data'/'nuscenes'` (ignoring the dataset config's own `DATA_PATH`), info-pkl
generation for UrbanIng-V2X is invoked by calling `create_nuscenes_info(...)` directly rather than
through the CLI entrypoint:

```python
from pcdet.datasets.nuscenes.nuscenes_dataset import create_nuscenes_info
create_nuscenes_info(
    version='v1.0-custom',
    data_path=(Path('tools').resolve() / cfg.DATA_CONFIG.DATA_PATH).resolve(),
    save_path=(Path('tools').resolve() / cfg.DATA_CONFIG.DATA_PATH).resolve(),
    max_sweeps=cfg.DATA_CONFIG.MAX_SWEEPS,
    with_cam=False,
)
```

---

## 5. Dataset / model configs

Four config files, mirroring each other for the two LiDAR variants:

| File | Purpose |
|---|---|
| `tools/cfgs/dataset_configs/urbaning_v2x_lidar_dataset.yaml` | vehicle-mode dataset config |
| `tools/cfgs/dataset_configs/urbaning_v2x_infra_lidar_dataset.yaml` | infra-mode dataset config |
| `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning.yaml` | vehicle-mode model config |
| `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_infra.yaml` | infra-mode model config |

Dataset configs: `DATASET: 'NuScenesOVDataset'`, `VERSION: 'v1.0-custom'`, `MAX_SWEEPS: 1`, same
`POINT_FEATURE_ENCODING` as stock nuScenes (`['x','y','z','intensity','timestamp']`), and a
minimal `DATA_PROCESSOR` (`mask_points_and_boxes_outside_range` → `shuffle_points` →
`transform_points_to_voxels`, `VOXEL_SIZE: [0.075, 0.075, 0.2]`). No `DATA_AUGMENTOR` /
`SELECTIVE_ALIGNMENT` / `BALANCED_RESAMPLING` — confirmed unused when `training=False`, since
`DataAugmentor` is only built `if self.training`.

Model configs: identical `MODEL`/`POST_PROCESSING` architecture to the stock
`ov_scan_lidar.yaml` (`OVScanLidar`, same `POINT_CLOUD_RANGE: [-54, -54, -5, 54, 54, 3]`), with
`CAMERA_CONFIG` dropped (`OVScanLidar` never consumes images) and `ALIGNMENT.NUSCENES_TO_OV_CLASSES`
+ new open-vocab-routing fields overridden — see §6–§7.

---

## 6. Inference + GT-comparison visualization — `tools/visualize_urbaning_v2x.py`

Adapted from the stock `visualize_nuscenes_mini.py`, extended with class-agnostic greedy 3D-IoU
matching (`pcdet.ops.iou3d_nms.iou3d_nms_utils.boxes_iou3d_gpu`) for TP/FP/FN sanity stats, since
GT categories are coarser than OV-SCAN's classes and a class-aware benchmark isn't meaningful at
this granularity (that's what §8's per-class eval script is for).

**Important correctness fix — GT range capping.** pcdet's `mask_points_and_boxes_outside_range`
only strips out-of-range GT boxes `if config.REMOVE_OUTSIDE_BOXES and self.training`; at
inference (`training=False`) GT keeps *every* labeled track for the whole scene, including
objects the sensor physically cannot see (e.g. vehicles on approach roads up to 187 m away, well
past the ±54 m point-cloud range and past predictions' own ±61.2 m `POST_CENTER_RANGE`). Matching
against those inflates FN and deflates recall for something no input point cloud could ever
contain. Fixed by explicitly capping GT to the model's own x/y range before matching:

```python
def filter_gt_boxes_in_range(gt_boxes, point_cloud_range):
    if len(gt_boxes) == 0:
        return gt_boxes
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    in_range = (gt_boxes[:, 0] >= x_min) & (gt_boxes[:, 0] <= x_max) & \
               (gt_boxes[:, 1] >= y_min) & (gt_boxes[:, 1] <= y_max)
    return gt_boxes[in_range]
```

This is a general correctness fix that applies to **any** class-agnostic geometric matching done
at inference time on this codebase, not just UrbanIng-V2X.

### Results snapshot (class-agnostic, IoU ≥ 0.25, `score_thresh=0.3`, 200 frames, GT range-capped)

| Variant | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Single-vehicle LiDAR | 1468 | 35 | 3654 | 0.977 | 0.287 |
| Infra-fused LiDAR (4 sensors) | 394 | 5 | 6930 | 0.987 | 0.054 |

Both variants: near-perfect precision, low recall — consistent with real domain gap (elevated,
wide-area, multi-sensor-fused infra geometry looks nothing like nuScenes' single ego-vehicle
LiDAR) rather than a pipeline defect; visually verified (point clouds align correctly with road
geometry, GT boxes correctly bound vehicles/pedestrians).

**These numbers are stale as of §8** — the open-vocab fixes in §8 (channel suppression zeroing
scores for non-routed proposals, and `RESCALE_BY_MARGIN_TEMPERATURE` scaling every remaining
score down by `sigmoid(margin/T) ≤ 1`) both reduce raw prediction scores, so rerunning this exact
script+config afterward gives a **lower recall at the same fixed `score_thresh=0.3` cutoff** —
not because detections got worse, but because this script uses a single absolute-score threshold,
while §7.3's AP is threshold-independent (rank-based) and genuinely improved. Rerun with the
current configs: single-vehicle → tp=1134, fp=21, fn=3988, **precision=0.982, recall=0.221**
(was 0.287); infra-fused → tp=231, fp=3, fn=7093, **precision=0.987, recall=0.032** (was 0.054).
**Practical implication:** `score_thresh` is not directly comparable across configs with
different `RESCALE_BY_MARGIN_TEMPERATURE` settings — if this script is used to compare configs
side-by-side in future work, either disable margin rescaling for the comparison or sweep
`score_thresh` per config rather than trusting one fixed default.

### Optional per-box class-id labels — `--show_labels`

Overlays a small digit (the box's coarse `CLASS_NAMES` id, 1-indexed, matching `pred_labels`/
`gt_boxes[:, -1]`) at each box's center, colored to match its wireframe (`PRED_COLOR_MAP`), plus
a legend (digit → class name) listing only the classes actually present in that frame — kept
deliberately lightweight (a digit, not the full class string) so dense scenes stay readable. GT
boxes are still always drawn with a fixed blue wireframe regardless of class (a legend line notes
this convention); only the digit reveals their class. Off by default (`add_boxes`/`save_scene`
behavior is unchanged unless `--show_labels` is passed).

Implementation note: open3d's legacy `Visualizer` (used here for headless rendering) has no 3D
text-label primitive, so labels are drawn as a **2D post-process** with PIL after the point
cloud/box render: right after `vis.update_renderer()` (before `destroy_window()`), the exact
camera parameters used for that render are pulled via
`ctr.convert_to_pinhole_camera_parameters()`, and each box center is projected to pixel
coordinates with the standard OpenCV/Open3D pinhole convention (extrinsic = world→camera, camera
looks down +Z):

```python
def _project_point(point_xyz, intrinsic, extrinsic):
    p_cam = extrinsic @ np.append(point_xyz, 1.0)
    if p_cam[2] <= 1e-6:
        return None
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    return fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy
```

Verified empirically against known 3D marker positions before integrating (projected pixel
lands exactly on the rendered marker) — safe to reuse this helper for any future annotation
overlay on this script's top-down renders (e.g. distance rings, track IDs).

---

## 7. Exploiting the open-vocabulary head

### 7.1 Mechanism

OV-SCAN's *final* label for every detection isn't the 10-channel heatmap's argmax — it comes from
a separate, frozen-CLIP zero-shot classification step
(`pcdet/models/dense_heads/ov_scan_head.py::OVScanHead.predict`). Each detected object's learned
BEV feature is projected (via a trainable `OVScanAlignmentHead`) into CLIP's embedding space and
matched by cosine similarity against a list of free-text class strings defined in the model
config's `ALIGNMENT.NUSCENES_TO_OV_CLASSES` (e.g. `"car": ["sedan", "van", "minivan", ...]`):

```python
res_layer['clip_preds'] = self.alignment_head(global_query_feat, top_proposals_class)
clip_preds = res_layer['clip_preds'].detach()
normalized_clip_preds = clip_preds / clip_preds.norm(dim=-1, keepdim=True)
clip_logits = self.clip_logit_scale_exp * normalized_clip_preds @ self.text_features.t()
ov_labels = torch.argmax(clip_logits, dim=-1)          # <- fine-grained open-vocab label
# ... pred_labels (coarse, 1 of 10) is then *derived* from ov_labels via a fixed dict,
#     not the other way around.
```

Because `self.text_features` is computed by encoding whatever strings are in the config at model
build time, **swapping the vocabulary requires no retraining** — CLIP's text tower runs fresh on
new strings at inference time. This is the exploitable "open vocabulary" property: point the
vocabulary at UrbanIng-V2X's own category names and the model's zero-shot guess is scored
directly against the dataset's real taxonomy instead of a lossy 10-class collapse.

### 7.2 Native vocabulary swap

```yaml
# tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning{,_infra}.yaml — ALIGNMENT.NUSCENES_TO_OV_CLASSES
NUSCENES_TO_OV_CLASSES: {
  "car": ["car", "van"],
  "truck": ["truck", "bus", "trailer", "other vehicle"],
  "bicycle": ["e-scooter", "motorcycle", "cyclist"],
  "pedestrian": ["pedestrian"],
}
```

Each dict *key* must still be one of the 10 `CLASS_NAMES` (needed for `pred_labels` bookkeeping);
grouped to match the same coarse nuScenes bucket the GT converter already collapses each
UrbanIng-V2X category into (§3.3), so predicted-coarse and GT-coarse stay consistent for members
of the same native group. (Why `"pedestrian"` has only one entry, not two, is explained in §8.2.)

### 7.3 Per-class AP/mAP evaluation — `tools/eval_urbaning_v2x_per_class.py`

Standard-practice **class-specific matching** (as in nuScenes/COCO): for each class of interest,
GT and predictions of *only that class* are matched by greedy IoU per frame — a right-place-
wrong-label detection is simultaneously a miss (FN) for the true class and a false alarm (FP) for
the predicted class. AP is the Pascal-VOC-2012 all-points-interpolated area under the resulting
precision-recall curve, computed over **every raw proposal** the model emits (no score cutoff, so
the full PR curve is covered — the model's `DENSE_HEAD.POST_PROCESSING.SCORE_THRESH: 0.0` already
means `filter=True` inference keeps essentially all `NUM_PROPOSALS=200` candidates per frame).

Two granularities are scored per run:
- **Fine native classes** (11): `car, van, bus, truck, other_vehicle, trailer, escooter,
  motorcycle, cyclist, pedestrian, other_pedestrian`.
- **UrbanIng-V2X's own 4 label groups**: `vehicle={car,van}`, `two_wheelers={cyclist,motorcycle,
  escooter}`, `heavy_vehicle={truck,bus,trailer,other_vehicle}`, `pedestrian={pedestrian,
  other_pedestrian}`.

```python
def voc_ap(recall, precision):
    """Pascal VOC 2012 all-points-interpolated average precision."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))
```

**AP edge cases, handled explicitly:** if a class has zero GT in the run, AP is `NaN` (genuinely
undefined — excluded from the mAP average). If a class has GT but the model *never* predicts it,
AP is **0.0**, not NaN (recall is uniformly 0 at every threshold — a well-defined value, and this
distinction matters a lot once you've hit the pedestrian-labeling bug in §8.2, where "never
predicted" was literally the failure mode).

The GT side reuses the `native_categories.json` sidecar (§3.4) and applies the same range-capping
as §6 before matching. Output: `output/nuscenes_models/<tag>/default/per_class_ap.json`
(`fine_classes`, `groups`, `mAP_fine_classes`, `mAP_groups`).

---

## 8. Zero-shot AP fixes for pedestrian / two-wheelers

Starting point (native-vocab, class-specific-matched, before any of §8's fixes):

| Group | Vehicle AP | Infra AP |
|---|---|---|
| vehicle | 0.583 | 0.299 |
| heavy_vehicle | 0.188 | 0.070 |
| two_wheelers | 0.005 | 0.180 |
| pedestrian | 0.289 | 0.205 |
| **mAP (groups)** | **0.266** | **0.189** |

### 8.1 Diagnosis — cross-bucket CLIP relabeling leakage

`clip_logits`'s argmax runs over the **entire flat vocabulary** for every proposal, independent of
which of the 10 heatmap channels generated it, and independent of the proposal's objectness score
(`final_scores = heatmap.max(1)` — purely geometric confidence, computed before any CLIP
involvement). A car-heatmap-channel proposal can therefore be relabeled "trailer" on pure
embedding-similarity noise. Quantified with a small standalone diagnostic (log
`heatmap_bucket` — the proposal's originating channel — alongside its final `pred_labels`):

```
ALL proposals (score>=0, 200 frames × up to 200 each): 12768/40000 leaked cross-bucket (31.9%)
Top cross-bucket leaks (heatmap_bucket -> final_bucket):
  pedestrian -> truck : 3283      barrier -> pedestrian : 2244
  car -> truck         : 1349      traffic_cone -> pedestrian : 1214
  trailer -> truck      : 1262      bus -> truck : 1007
  motorcycle -> bicycle  : 658      truck -> car : 630
  motorcycle -> pedestrian: 221     bicycle -> pedestrian : 196
```

`barrier→pedestrian`/`traffic_cone→pedestrian` directly flood the pedestrian vocabulary with
static-street-furniture false positives; `motorcycle→bicycle`/`motorcycle→pedestrian` drain real
two-wheeler-channel proposals away. This dominates the poor pedestrian/two-wheeler numbers above.

### 8.2 Fix 1 — explicit per-heatmap-channel routing table (`ov_scan_head.py`)

A naive "same coarse-bucket name only" constraint is **not enough**: 6 of the 10 heatmap channels
(`construction_vehicle, barrier, motorcycle, bus, trailer, traffic_cone`) have no vocabulary key of
their own in `NUSCENES_TO_OV_CLASSES` (bus/trailer/motorcycle route to differently-named vocab
buckets — `truck`/`bicycle` — by design; barrier/traffic_cone/construction_vehicle have no
UrbanIng-V2X analog at all), so a same-name-only mask leaves them all unconstrained. The fix is an
explicit routing table plus outright suppression for channels with nothing to route to:

```yaml
# ALIGNMENT config
CONSTRAIN_TO_HEATMAP_BUCKET: True
HEATMAP_BUCKET_TO_VOCAB_KEYS: {
  "car": ["car"], "truck": ["truck"], "bus": ["truck"], "trailer": ["truck"],
  "motorcycle": ["bicycle"], "bicycle": ["bicycle"], "pedestrian": ["pedestrian"],
}
# barrier, traffic_cone, construction_vehicle: omitted -> proposals from those channels suppressed
```

```python
# ov_scan_head.py __init__ — build a (num_heatmap_channels, num_ov_classes) boolean routing mask
routing_table = self.model_cfg.ALIGNMENT.get('HEATMAP_BUCKET_TO_VOCAB_KEYS', None) \
    or {k: [k] for k in self.nuscenes_to_ov_classes.keys()}     # fallback: same-name only
heatmap_to_vocab_mask = torch.zeros(len(self.class_names), len(self.ov_classes), dtype=torch.bool)
for heatmap_channel, vocab_keys in routing_table.items():
    h_idx = self.class_names.index(heatmap_channel)
    allowed = [i for i, c in enumerate(self.ov_classes) if self.ov_to_nuscenes_classes[c] in vocab_keys]
    heatmap_to_vocab_mask[h_idx, allowed] = True
self.heatmap_to_vocab_mask = heatmap_to_vocab_mask.cuda()
self.suppressed_heatmap_channels = (heatmap_to_vocab_mask.sum(dim=-1) == 0).cuda()
```

```python
# predict() — mask clip_logits to this proposal's own channel's allowed vocabulary
allowed_mask = self.heatmap_to_vocab_mask[top_proposals_class]
safe_mask = torch.where(allowed_mask.any(-1, keepdim=True), allowed_mask, torch.ones_like(allowed_mask))
clip_logits = clip_logits.masked_fill(~safe_mask, float('-inf'))
```

```python
# decode_bbox() — proposals from a fully-suppressed channel get their score zeroed, so they're
# dropped by the score_thresh filter instead of surfacing under an arbitrary argmax(-inf...) pick
if self.constrain_ov_to_bucket and query_labels is not None:
    final_scores = final_scores.masked_fill(self.suppressed_heatmap_channels[query_labels], 0.0)
```

(`query_labels` — the proposal's originating heatmap channel — is threaded through
`predict()→get_bboxes()→decode_bbox()` as a new optional parameter, defaulting to `None` so
training-time code paths that don't pass it are unaffected; this is the same value exposed as
`heatmap_bucket` in the final `predictions_dict`, useful for future diagnostics.)

Effect (vehicle variant, this fix alone): `heavy_vehicle` 0.188→0.318, `vehicle` 0.583→0.691,
`pedestrian` 0.289→0.403 (via the group; the fine class was still 0 at this point — see next),
`two_wheelers` unchanged (0.005→0.004 — expected: two-wheelers' problem turned out to be
*localization* noise, not labeling; see §8.4).

### 8.3 Fix 2 — vocabulary consolidation (the "other pedestrian" attractor)

Even after the routing fix, the fine `pedestrian` class still had **0 predictions** — CLIP's
argmax within the pedestrian bucket *always* preferred `"other pedestrian"` over `"pedestrian"`
for every pedestrian-shaped proposal, a pure prompt-text artifact with no detection-quality signal
behind it (confirmed: precisely 0/N pedestrian-string picks across the whole 200-frame run).
Since evaluation scores both the fine class *and* the group (`pedestrian_group =
Pedestrian+OtherPedestrians` pooled), dropping the split loses nothing at the group level:

```yaml
NUSCENES_TO_OV_CLASSES: { ..., "pedestrian": ["pedestrian"] }   # was ["pedestrian", "other pedestrian"]
```

Effect: fine `pedestrian` AP 0.000 → 0.403 (now identical to the group, as expected).

### 8.4 Fix 3 — CLIP-confidence score rescaling

A complementary, continuous heuristic: multiply each box's objectness score by
`sigmoid(margin / T)`, where `margin` is the top-1-vs-runner-up gap in the (already
bucket-constrained) CLIP logits — proposals CLIP itself is unsure about within its own bucket get
down-weighted in the score used for *ranking* (hence AP), instead of competing on equal footing
with confident detections:

```python
# predict()
if clip_logits.shape[-1] > 1:
    top2_vals = torch.topk(clip_logits, k=2, dim=-1).values
    ov_margin = top2_vals[..., 0] - top2_vals[..., 1]
else:
    ov_margin = torch.full_like(ov_labels, float('inf'), dtype=clip_logits.dtype)
res_layer["ov_margin"] = ov_margin
```
```python
# decode_bbox()
if self.ov_rescale_temperature is not None and ov_margin is not None:
    final_scores = final_scores * torch.sigmoid(ov_margin / self.ov_rescale_temperature)
```
```yaml
RESCALE_BY_MARGIN_TEMPERATURE: 0.3   # margins empirically range ~0.0-1.6, median ~0.5
```

Effect: mostly benefited `heavy_vehicle`/`truck`/`van` (0.315→0.354, 0.055→0.061, 0.176→0.187 on
the vehicle variant) — classes with genuine multi-string competition (4 and 2 candidates
respectively). **No measurable effect on two-wheelers** — see §8.5 for why.

A related but separate knob, `MARGIN_THRESH`, was also implemented (discrete fallback to a fixed
default vocab string when the margin is below threshold, rather than continuous rescaling) but
showed no benefit over the continuous version in testing and was left disabled (`None`) in the
final configs. Kept in the code as `ALIGNMENT.MARGIN_THRESH` for future experimentation.

### 8.5 Final results

| Group | Vehicle: before → after | Infra: before → after |
|---|---|---|
| pedestrian | 0.289 → **0.403** | 0.205 → **0.287** |
| two_wheelers | 0.005 → **0.011** | 0.180 → **0.256** |
| heavy_vehicle | 0.188 → 0.354 | 0.070 → 0.352 |
| vehicle | 0.583 → 0.690 | 0.299 → 0.427 |
| **mAP (groups)** | **0.266 → 0.364** | **0.189 → 0.330** |

Every class improved in both variants. **Known limitation, worth being explicit about:**
two-wheelers on the single-vehicle LiDAR barely moved (0.005→0.011) despite the same fixes lifting
infra-fused substantially (0.180→0.256, driven by 68 vs. 363 GT instances and much denser point
coverage of cyclists from the elevated infra view). Root cause isolated: for single-vehicle LiDAR,
the `bicycle`/`motorcycle` heatmap channels themselves generate many high-objectness-score
proposals that aren't real objects — a **localization/scoring** problem, not a **labeling**
problem, so no amount of zero-shot *vocabulary* engineering fixes it (margin-rescaling doesn't
touch objectness score's underlying calibration, only the label choice and a post-hoc CLIP-
confidence multiplier). Closing this gap fully would need either retraining or a different
score-calibration approach for that specific channel/geometry — flagged here as the natural next
investigation, not attempted in Phase 1.

All new `ALIGNMENT` fields default to `False`/`None` (`CONSTRAIN_TO_HEATMAP_BUCKET`,
`HEATMAP_BUCKET_TO_VOCAB_KEYS`, `MARGIN_THRESH`, `RESCALE_BY_MARGIN_TEMPERATURE`), so the stock
`ov_scan_lidar.yaml` (real nuScenes eval) is completely unaffected by any of this — verified by
inspection (every new code path is gated behind these config reads).

---

## 9. File manifest

**New files:**
- `tools/urbaning_v2x/convert_to_nuscenes.py` — the converter (§3)
- `tools/cfgs/dataset_configs/urbaning_v2x_lidar_dataset.yaml` — vehicle dataset config
- `tools/cfgs/dataset_configs/urbaning_v2x_infra_lidar_dataset.yaml` — infra dataset config
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning.yaml` — vehicle model config
- `tools/cfgs/nuscenes_models/ov_scan_lidar_urbaning_infra.yaml` — infra model config
- `tools/visualize_urbaning_v2x.py` — visualization + class-agnostic match stats (§6)
- `tools/eval_urbaning_v2x_per_class.py` — per-class AP/mAP eval (§7.3)

**Modified files:**
- `pcdet/datasets/nuscenes/nuscenes_dataset.py` — bug fixes 1 & 3 (§4)
- `pcdet/datasets/nuscenes/nuscenes_utils.py` — bug fix 2 (§4)
- `pcdet/models/dense_heads/ov_scan_head.py` — open-vocab routing/suppression/margin-rescale
  (§8.2–§8.4); every addition is opt-in via new `ALIGNMENT` config fields, no change to default
  behavior.

**Data layout on disk (container paths under `/OV-SCAN/`):**
```
datasets/urbaning_v2x_raw/dataset/<sequence>/     # selectively-extracted raw UrbanIng-V2X data
datasets/urbaning_v2x/v1.0-custom/                # converted, vehicle mode (json tables + pkls + sidecar)
datasets/urbaning_v2x/samples/LIDAR_TOP/          # converted point clouds, vehicle mode
datasets/urbaning_v2x_infra/v1.0-custom/          # converted, infra mode
datasets/urbaning_v2x_infra/samples/LIDAR_TOP/    # converted point clouds, infra mode (fused)
OV-SCAN/output/nuscenes_models/ov_scan_lidar_urbaning{,_infra}/default/
  visualizations/*.png, match_summary.json        # §6 output
  per_class_ap.json                                # §7.3 output
```

---

## 10. Reproducing / extending this pipeline

**To convert another sequence** (vehicle mode):
```bash
7z x -y -o<raw_dest> <sequence>.7z.001 "@filelist.txt"   # selective extraction, see §1
python tools/urbaning_v2x/convert_to_nuscenes.py \
    --root_folder <raw_dest_parent> --sequence <sequence> --ego vehicle1 --out <converted_dest>
# then regenerate the info pkl (§4's direct-call snippet) and rerun visualize/eval scripts.
```

**To try a different infra sensor subset:** just change `--lidars` — no code changes needed, the
converter and `HEATMAP_BUCKET_TO_VOCAB_KEYS` routing are both sensor-count-agnostic.

**To extend the open-vocab routing to more classes** (e.g. if a future sequence has
construction-vehicle or barrier GT): add the corresponding key(s) to both
`NUSCENES_TO_OV_CLASSES` and `HEATMAP_BUCKET_TO_VOCAB_KEYS` in the model config — no code changes
needed, §8.2's mask-building is fully data-driven from those two dicts.

**Natural next steps** (not done in Phase 1): scale to more/all 34 sequences; investigate the
single-vehicle two-wheeler localization gap (§8.5); try CLIP prompt-template ensembling (the
`PROMPT` config field already supports a list, encoded and mean-pooled — implemented but not
tuned/evaluated in Phase 1); consider fine-tuning if zero-shot gains plateau.
