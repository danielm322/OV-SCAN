"""
Convert one UrbanIng-V2X sequence into a nuscenes-devkit-style raw dataset (json tables +
samples/LIDAR_TOP/*.pcd.bin) that OV-SCAN's stock pcdet nuScenes info-pkl builder
(pcdet/datasets/nuscenes/nuscenes_dataset.py --func create_nuscenes_infos) can consume
unmodified.

Three modes, mutually exclusive:

  --ego {vehicle1,vehicle2}
      Single vehicle-mounted LiDAR, no fusion. Points are stored exactly as captured in the
      LiDAR's own sensor frame (no extrinsic applied), matching nuScenes convention -- pcdet
      transforms GLOBAL-frame GT boxes into that same sensor frame via calibrated_sensor +
      ego_pose, so points and boxes end up aligned. The ego vehicle's own label track is
      excluded (it can't detect itself).

  --lidars <comma-separated calibration.json channel names, e.g. crossing2_11_lidar,crossing2_31_lidar>
      Fuses one or more *static* infrastructure LiDARs. Each has a static lidar->global
      extrinsic (gTl) in calibration.json, so points from every selected sensor are
      transformed straight into the global frame and concatenated -- no per-frame ego motion
      to track. Both points and GT boxes (GT is stored in the global frame already) are then
      recentered by subtracting the mean position of the selected sensors, so the fused scene
      sits inside OV-SCAN's +-54m point-cloud range regardless of where the map's raw origin
      is. calibrated_sensor/ego_pose are written as identity (the recentered global frame IS
      the "ego" frame here). No label track is excluded: the observed vehicles are legitimate
      detection targets from an infrastructure viewpoint. Optionally voxel-downsampled
      (--voxel_size) since fusing multiple lidars is several times denser than the single
      nuScenes LIDAR_TOP the model was trained on.

  --fuse_ego {vehicle1,vehicle2} [--fuse_vehicle {vehicle1,vehicle2}] [--fuse_lidars <...>]
      V2V / V2I: fuses --fuse_ego's own LiDAR with the other vehicle's LiDAR (--fuse_vehicle,
      v2v) and/or one or more static infra LiDARs (--fuse_lidars, v2i) into --fuse_ego's own raw
      sensor frame -- the exact frame the plain --ego mode already emits, so calibrated_sensor
      and ego_pose reuse that mode's logic unchanged; only point loading differs, and the fused
      cloud stays in the same egocentric distribution the checkpoint was trained on (unlike
      --lidars' recentered-global frame). Every other source's points are transformed into
      --fuse_ego's sensor frame per keyframe via inv(vTl_ref) @ inv(gTv_ref) @ <source's own
      global transform>. Only --fuse_ego's own GT track is excluded (same rationale as --ego); a
      fused-in *other* vehicle (v2v) is kept as a legitimate GT target since it's exactly what
      v2v fusion is meant to help detect. See FusedEgoConverter's docstring for the full math.

Usage:
    # single vehicle lidar
    python convert_to_nuscenes.py \
        --root_folder /OV-SCAN/datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --ego vehicle1 \
        --out /OV-SCAN/datasets/urbaning_v2x

    # infrastructure-fused lidar (pick any subset of the sequence's infra sensors)
    python convert_to_nuscenes.py \
        --root_folder /OV-SCAN/datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --lidars crossing2_11_lidar,crossing2_12_lidar,crossing2_31_lidar,crossing2_32_lidar \
        --voxel_size 0.15 \
        --out /OV-SCAN/datasets/urbaning_v2x_infra

    # v2v: vehicle1 as reference, vehicle2's lidar fused in
    python convert_to_nuscenes.py \
        --root_folder /OV-SCAN/datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --fuse_ego vehicle1 --fuse_vehicle vehicle2 \
        --out /OV-SCAN/datasets/urbaning_v2x_v2v_v1ref

    # v2i: vehicle1 as reference, all 4 infra lidars fused in
    python convert_to_nuscenes.py \
        --root_folder /OV-SCAN/datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --fuse_ego vehicle1 \
        --fuse_lidars crossing2_11_lidar,crossing2_12_lidar,crossing2_31_lidar,crossing2_32_lidar \
        --voxel_size 0.15 \
        --out /OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1
"""
import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Car/Van -> car, Bus/Truck/OtherVehicle/Trailer -> truck, EScooter/Motorcycle/Cyclist -> bicycle,
# Pedestrian/OtherPedestrians -> pedestrian. Same mapping UrbanIng-V2X's own converter uses, so it's
# already vetted against pcdet's map_name_from_general_to_detection keys.
OBJECT_TYPE_TO_NUSCENES = {
    "Car": "vehicle.car",
    "Van": "vehicle.car",
    "Bus": "vehicle.truck",
    "Truck": "vehicle.truck",
    "OtherVehicle": "vehicle.truck",
    "Trailer": "vehicle.truck",
    "EScooter": "vehicle.bicycle",
    "Motorcycle": "vehicle.bicycle",
    "Cyclist": "vehicle.bicycle",
    "Pedestrian": "human.pedestrian.adult",
    "OtherPedestrians": "human.pedestrian.adult",
    "Animal": "animal",
    "Other": "movable_object.barrier",
}


def create_token():
    return uuid.uuid4().hex


def rotation_matrix_to_nuscenes_quaternion(r):
    # nuScenes stores quaternions as [w, x, y, z]
    return Rotation.from_matrix(r).as_quat(scalar_first=True).tolist()


def load_labels_by_timestamp(labels_path, skip_track_id=None):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    by_ts = {}
    for track in labels['tracks']:
        if skip_track_id is not None and track['track_id'] == skip_track_id:
            continue
        for i, ts in enumerate(track['timestamps']):
            by_ts.setdefault(ts, []).append({
                'track_id': track['track_id'],
                'object_type': track['object_type'],
                'position': track['positions'][i],
                'orientation': track['orientations'][i],
                'dimension': track['dimensions'][0] if len(track['dimensions']) == 1 else track['dimensions'][i],
            })
    return by_ts


def voxel_downsample(points, voxel_size):
    """Mean-pool points falling in the same voxel. points: (N, 4) xyz+intensity."""
    if not voxel_size or voxel_size <= 0 or len(points) == 0:
        return points
    voxel_idx = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    df = pd.DataFrame({
        'vx': voxel_idx[:, 0], 'vy': voxel_idx[:, 1], 'vz': voxel_idx[:, 2],
        'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2], 'i': points[:, 3],
    })
    agg = df.groupby(['vx', 'vy', 'vz'], sort=False).mean()
    return agg[['x', 'y', 'z', 'i']].to_numpy().astype(np.float32)


def load_raw_lidar_points(seq_dir, channel, filename):
    """Load one UrbanIng-V2X .npz lidar frame in the sensor's own local frame, with the
    self-vehicle-body return filter applied. Returns (N, 4) xyz+intensity float32."""
    raw = np.load(seq_dir / channel / filename)
    points = np.stack([raw['x'], raw['y'], raw['z'], raw['intensity']], axis=1).astype(np.float32)
    return points[np.linalg.norm(points[:, :3], axis=1) > 1.0]


def transform_points(points_xyz, transform):
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float32)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def count_points_in_box(points_xyz, center, orientation, dimension):
    """Points of a point cloud (already in the same frame as `center`) that fall inside a
    z-rotated 3D box. Returns (num_points, box_rotation_matrix)."""
    center = np.asarray(center)
    gRobj = Rotation.from_euler('z', orientation).as_matrix()
    pts_obj = (points_xyz - center) @ gRobj
    half = np.asarray(dimension) / 2.0
    num_pts = int(np.all(np.abs(pts_obj) <= half, axis=1).sum())
    return num_pts, gRobj


class BaseNuscenesConverter:
    """Shared instance/category bookkeeping + json-table writing for both converter modes.

    Subclasses populate self.{sample,sample_data,ego_pose,sample_annotation}_table plus the
    token attributes below in their own _convert(), then call write_tables().
    """

    def _instance_token(self, track_id):
        if not hasattr(self, '_instance_tokens'):
            self._instance_tokens = {}
        if track_id not in self._instance_tokens:
            self._instance_tokens[track_id] = create_token()
        return self._instance_tokens[track_id]

    def finalize_instances(self):
        self.instance_table = []
        for track_id, tok in getattr(self, '_instance_tokens', {}).items():
            ts_tokens = self._ann_tokens_by_track[track_id]
            sorted_ts = sorted(ts_tokens.keys())
            obj_type = next(o['object_type'] for ts in self.labels_by_ts.values() for o in ts if o['track_id'] == track_id)
            self.instance_table.append({
                'token': tok,
                'category_token': self.category_tokens[OBJECT_TYPE_TO_NUSCENES[obj_type]],
                'nbr_annotations': len(ts_tokens),
                'first_annotation_token': ts_tokens[sorted_ts[0]],
                'last_annotation_token': ts_tokens[sorted_ts[-1]],
            })

    def write_tables(self):
        self.finalize_instances()

        category_table = [{'token': tok, 'name': name, 'description': ''}
                           for name, tok in self.category_tokens.items()]
        visibility_table = [{'token': self.visibility_token, 'level': '1', 'description': 'Fully visible'}]
        attribute_table = []
        sensor_table = [{'token': self.sensor_token, 'channel': 'LIDAR_TOP', 'modality': 'lidar'}]
        calibrated_sensor_table = [{
            'token': self.calibrated_sensor_token,
            'sensor_token': self.sensor_token,
            'translation': self.calib_translation.tolist(),
            'rotation': rotation_matrix_to_nuscenes_quaternion(self.calib_rotation_matrix),
            'camera_intrinsic': [],
        }]
        log_table = [{'token': self.log_token, 'logfile': '', 'date_captured': self.sequence.split('_')[0],
                      'location': 'Ingolstadt'}]
        scene_table = [{
            'token': self.scene_token,
            'log_token': self.log_token,
            'nbr_samples': len(self.sample_table),
            'first_sample_token': self.sample_table[0]['token'],
            'last_sample_token': self.sample_table[-1]['token'],
            'name': self.scene_name,
            'description': '',
        }]
        map_table = [{'category': '', 'token': create_token(), 'filename': '', 'log_tokens': [self.log_token]}]

        # sample.json must not carry the transient 'data' helper key we used internally.
        sample_table = [{k: v for k, v in s.items() if k != 'data'} for s in self.sample_table]

        tables = {
            'category': category_table,
            'attribute': attribute_table,
            'visibility': visibility_table,
            'instance': self.instance_table,
            'sensor': sensor_table,
            'calibrated_sensor': calibrated_sensor_table,
            'ego_pose': self.ego_pose_table,
            'log': log_table,
            'scene': scene_table,
            'sample': sample_table,
            'sample_data': self.sample_data_table,
            'sample_annotation': self.sample_annotation_table,
            'map': map_table,
        }
        for name, data in tables.items():
            with open(self.table_dir / f'{name}.json', 'w') as f:
                json.dump(data, f)

        # Sidecar for per-class eval: original UrbanIng-V2X category per GT box, in the exact
        # order/subset pcdet's info-pkl builder keeps (annotations with num_lidar_pts > 0, in
        # sample_annotation insertion order -- see module docstring note in the eval script for
        # why this ordering is safe to rely on).
        with open(self.table_dir / 'native_categories.json', 'w') as f:
            json.dump(self.native_categories_by_sample, f)

        # Sidecar recording the recentering offset (if any) subtracted from points/GT before
        # writing, so downstream tooling (e.g. late-fusion box merging) can recover true global
        # coordinates: true_global = local + offset. Zero for converters that don't recenter
        # (SequenceConverter, FusedEgoConverter both already emit true-global-recoverable frames
        # via ego_pose/calibrated_sensor); populated for InfraFusionConverter, whose recentered
        # frame has no other on-disk record of the offset that was applied.
        offset = getattr(self, 'offset', np.zeros(3))
        with open(self.table_dir / 'frame_offset.json', 'w') as f:
            json.dump({'offset': np.asarray(offset).tolist()}, f)

        print(f'Wrote {len(sample_table)} samples, {len(self.sample_annotation_table)} annotations, '
              f'{len(self.instance_table)} instances to {self.table_dir}')


class SequenceConverter(BaseNuscenesConverter):
    def __init__(self, root_folder, labels_folder, av_track_ids_path, sequence, ego, out_root, version):
        self.root_folder = Path(root_folder)
        self.sequence = sequence
        self.ego = ego
        self.seq_dir = self.root_folder / 'dataset' / sequence
        self.lidar_channel = f'{ego}_middle_lidar'
        self.state_channel = f'{ego}_state'

        self.table_dir = Path(out_root) / version
        self.samples_dir = Path(out_root) / 'samples' / 'LIDAR_TOP'
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if not (self.table_dir / 'samples').exists():
            os.symlink('../samples', self.table_dir / 'samples')

        with open(self.seq_dir / 'calibration.json') as f:
            calib = json.load(f)
        self.vTl = np.asarray(calib[self.lidar_channel]['extrinsics']['vTl'])
        self.calib_translation = self.vTl[:3, 3]
        self.calib_rotation_matrix = self.vTl[:3, :3]
        self.scene_name = f'{sequence}_{ego}'

        with open(av_track_ids_path) as f:
            av_track_ids = json.load(f)
        self.ego_track_id = av_track_ids[sequence][ego]

        self.labels_by_ts = load_labels_by_timestamp(
            Path(labels_folder) / f'{sequence}.json', skip_track_id=self.ego_track_id)

        self.time_sync = pd.read_csv(self.seq_dir / 'timesync_info.csv').set_index('Unnamed: 0')

        self.category_tokens = {name: create_token() for name in sorted(set(OBJECT_TYPE_TO_NUSCENES.values()))}
        self.visibility_token = create_token()
        self.sensor_token = create_token()
        self.calibrated_sensor_token = create_token()
        self.log_token = create_token()
        self.scene_token = create_token()

        self.sample_table = []
        self.sample_data_table = []
        self.ego_pose_table = []
        self.sample_annotation_table = []
        self.instance_table = []
        self.native_categories_by_sample = {}

        self._convert()

    def _state_gTv(self, state_filename):
        with open(self.seq_dir / self.state_channel / state_filename) as f:
            state = json.load(f)
        return np.asarray(state['gTv'])

    def _convert(self):
        columns = list(self.time_sync.columns)
        sample_tokens = [create_token() for _ in columns]
        pose_tokens = [create_token() for _ in columns]
        # track_id -> {timestamp_ms: annotation_token}
        ann_tokens_by_track = {}
        for col in columns:
            ts_ms = float(self.time_sync[col]['timestamp_ms'])
            for obj in self.labels_by_ts.get(ts_ms / 1000.0, []):
                ann_tokens_by_track.setdefault(obj['track_id'], {})[int(ts_ms)] = create_token()

        for idx, col in enumerate(columns):
            row = self.time_sync[col]
            ts_ms = int(row['timestamp_ms'])
            ts_us = ts_ms * 1000
            sample_token = sample_tokens[idx]
            pose_token = pose_tokens[idx]

            gTv = self._state_gTv(row[self.state_channel])
            self.ego_pose_table.append({
                'token': pose_token,
                'timestamp': ts_us,
                'translation': gTv[:3, 3].tolist(),
                'rotation': rotation_matrix_to_nuscenes_quaternion(gTv[:3, :3]),
            })

            self.sample_table.append({
                'token': sample_token,
                'timestamp': ts_us,
                'prev': sample_tokens[idx - 1] if idx > 0 else '',
                'next': sample_tokens[idx + 1] if idx < len(columns) - 1 else '',
                'scene_token': self.scene_token,
            })

            points = load_raw_lidar_points(self.seq_dir, self.lidar_channel, row[self.lidar_channel])
            out_name = f'{self.sequence}_{self.ego}_{ts_ms:06d}.pcd.bin'
            points_with_time = np.concatenate(
                [points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)  # single sweep -> timestamp=0
            points_with_time.tofile(self.samples_dir / out_name)

            sd_token = create_token()
            self.sample_data_table.append({
                'token': sd_token,
                'sample_token': sample_token,
                'ego_pose_token': pose_token,
                'calibrated_sensor_token': self.calibrated_sensor_token,
                'filename': f'samples/LIDAR_TOP/{out_name}',
                'fileformat': 'pcd',
                'timestamp': ts_us,
                'is_key_frame': True,
                'height': None,
                'width': None,
                'prev': '',
                'next': '',
            })
            # nuscenes-devkit indexes sample.data by channel via sample_data's own linkage below.
            self.sample_table[-1].setdefault('data', {})
            self.sample_table[-1]['data'] = {'LIDAR_TOP': sd_token}

            # Points in global frame, only to count how many fall inside each GT box (num_lidar_pts).
            gTl = gTv @ self.vTl
            points_global = transform_points(points[:, :3], gTl)

            for obj in self.labels_by_ts.get(ts_ms / 1000.0, []):
                track_id = obj['track_id']
                l, w, h = obj['dimension']
                center = np.asarray(obj['position'])
                num_lidar_pts, gRobj = count_points_in_box(
                    points_global, obj['position'], obj['orientation'], obj['dimension'])

                ts_tokens = ann_tokens_by_track[track_id]
                ann_token = ts_tokens[ts_ms]
                sorted_ts = sorted(ts_tokens.keys())
                pos = sorted_ts.index(ts_ms)
                self.sample_annotation_table.append({
                    'token': ann_token,
                    'sample_token': sample_token,
                    'instance_token': self._instance_token(track_id),
                    'visibility_token': self.visibility_token,
                    'attribute_tokens': [],
                    'translation': center.tolist(),
                    'size': [w, l, h],
                    'rotation': rotation_matrix_to_nuscenes_quaternion(gRobj),
                    'prev': ts_tokens[sorted_ts[pos - 1]] if pos > 0 else '',
                    'next': ts_tokens[sorted_ts[pos + 1]] if pos < len(sorted_ts) - 1 else '',
                    'num_lidar_pts': num_lidar_pts,
                    'num_radar_pts': 0,
                })
                # pcdet's info-pkl builder drops annotations with num_lidar_pts == 0 (see
                # nuscenes_utils.fill_trainval_infos); mirror that here so this list lines up
                # 1:1, in order, with info['gt_boxes'] / data_dict['gt_boxes'] for this sample.
                if num_lidar_pts > 0:
                    self.native_categories_by_sample.setdefault(sample_token, []).append(obj['object_type'])

        self._ann_tokens_by_track = ann_tokens_by_track
        self._sample_tokens = sample_tokens


class InfraFusionConverter(BaseNuscenesConverter):
    """Fuses one or more static infrastructure LiDARs into a single per-frame point cloud in a
    recentered global frame. See module docstring for the --lidars mode."""

    def __init__(self, root_folder, labels_folder, sequence, lidars, out_root, version, voxel_size):
        self.root_folder = Path(root_folder)
        self.sequence = sequence
        self.lidars = lidars
        self.voxel_size = voxel_size
        self.seq_dir = self.root_folder / 'dataset' / sequence

        self.table_dir = Path(out_root) / version
        self.samples_dir = Path(out_root) / 'samples' / 'LIDAR_TOP'
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if not (self.table_dir / 'samples').exists():
            os.symlink('../samples', self.table_dir / 'samples')

        with open(self.seq_dir / 'calibration.json') as f:
            calib = json.load(f)
        self.gTl_by_lidar = {}
        for chan in lidars:
            extrinsics = calib[chan]['extrinsics']
            if 'gTl' not in extrinsics:
                raise ValueError(f"'{chan}' has no static 'gTl' extrinsic in calibration.json "
                                  f"(available: {list(extrinsics.keys())}) -- is it a vehicle-mounted sensor?")
            self.gTl_by_lidar[chan] = np.asarray(extrinsics['gTl'])

        # Recenter around the mean position of the selected sensors so the fused scene sits
        # inside the model's +-54m point-cloud range regardless of the map's raw origin.
        self.offset = np.mean([gTl[:3, 3] for gTl in self.gTl_by_lidar.values()], axis=0)

        self.calib_translation = np.zeros(3)
        self.calib_rotation_matrix = np.eye(3)
        self.scene_name = f'{sequence}_infra_{"-".join(lidars)}'

        # No ego track to exclude: the observed vehicles are legitimate detection targets from
        # an infrastructure viewpoint.
        self.labels_by_ts = load_labels_by_timestamp(Path(labels_folder) / f'{sequence}.json')

        self.time_sync = pd.read_csv(self.seq_dir / 'timesync_info.csv').set_index('Unnamed: 0')

        self.category_tokens = {name: create_token() for name in sorted(set(OBJECT_TYPE_TO_NUSCENES.values()))}
        self.visibility_token = create_token()
        self.sensor_token = create_token()
        self.calibrated_sensor_token = create_token()
        self.log_token = create_token()
        self.scene_token = create_token()

        self.sample_table = []
        self.sample_data_table = []
        self.ego_pose_table = []
        self.sample_annotation_table = []
        self.instance_table = []
        self.native_categories_by_sample = {}

        self._convert()

    def _load_lidar_points_global(self, chan, filename):
        points = load_raw_lidar_points(self.seq_dir, chan, filename)
        points_global = transform_points(points[:, :3], self.gTl_by_lidar[chan])
        return np.concatenate([points_global - self.offset, points[:, 3:4]], axis=1).astype(np.float32)

    def _convert(self):
        columns = list(self.time_sync.columns)
        sample_tokens = [create_token() for _ in columns]
        pose_tokens = [create_token() for _ in columns]
        ann_tokens_by_track = {}
        for col in columns:
            ts_ms = float(self.time_sync[col]['timestamp_ms'])
            for obj in self.labels_by_ts.get(ts_ms / 1000.0, []):
                ann_tokens_by_track.setdefault(obj['track_id'], {})[int(ts_ms)] = create_token()

        for idx, col in enumerate(columns):
            row = self.time_sync[col]
            ts_ms = int(row['timestamp_ms'])
            ts_us = ts_ms * 1000
            sample_token = sample_tokens[idx]
            pose_token = pose_tokens[idx]

            # Identity: the recentered global frame IS the "ego" frame for a fused static rig.
            self.ego_pose_table.append({
                'token': pose_token,
                'timestamp': ts_us,
                'translation': [0.0, 0.0, 0.0],
                'rotation': [1.0, 0.0, 0.0, 0.0],
            })

            self.sample_table.append({
                'token': sample_token,
                'timestamp': ts_us,
                'prev': sample_tokens[idx - 1] if idx > 0 else '',
                'next': sample_tokens[idx + 1] if idx < len(columns) - 1 else '',
                'scene_token': self.scene_token,
            })

            fused_points = np.concatenate(
                [self._load_lidar_points_global(chan, row[chan]) for chan in self.lidars], axis=0)
            fused_points = voxel_downsample(fused_points, self.voxel_size)

            out_name = f'{self.sequence}_infra_{ts_ms:06d}.pcd.bin'
            points_with_time = np.concatenate(
                [fused_points, np.zeros((fused_points.shape[0], 1), dtype=np.float32)], axis=1)
            points_with_time.tofile(self.samples_dir / out_name)

            sd_token = create_token()
            self.sample_data_table.append({
                'token': sd_token,
                'sample_token': sample_token,
                'ego_pose_token': pose_token,
                'calibrated_sensor_token': self.calibrated_sensor_token,
                'filename': f'samples/LIDAR_TOP/{out_name}',
                'fileformat': 'pcd',
                'timestamp': ts_us,
                'is_key_frame': True,
                'height': None,
                'width': None,
                'prev': '',
                'next': '',
            })
            self.sample_table[-1].setdefault('data', {})
            self.sample_table[-1]['data'] = {'LIDAR_TOP': sd_token}

            for obj in self.labels_by_ts.get(ts_ms / 1000.0, []):
                track_id = obj['track_id']
                l, w, h = obj['dimension']
                center = np.asarray(obj['position']) - self.offset
                num_lidar_pts, gRobj = count_points_in_box(
                    fused_points[:, :3], center, obj['orientation'], obj['dimension'])

                ts_tokens = ann_tokens_by_track[track_id]
                ann_token = ts_tokens[ts_ms]
                sorted_ts = sorted(ts_tokens.keys())
                pos = sorted_ts.index(ts_ms)
                self.sample_annotation_table.append({
                    'token': ann_token,
                    'sample_token': sample_token,
                    'instance_token': self._instance_token(track_id),
                    'visibility_token': self.visibility_token,
                    'attribute_tokens': [],
                    'translation': center.tolist(),
                    'size': [w, l, h],
                    'rotation': rotation_matrix_to_nuscenes_quaternion(gRobj),
                    'prev': ts_tokens[sorted_ts[pos - 1]] if pos > 0 else '',
                    'next': ts_tokens[sorted_ts[pos + 1]] if pos < len(sorted_ts) - 1 else '',
                    'num_lidar_pts': num_lidar_pts,
                    'num_radar_pts': 0,
                })
                if num_lidar_pts > 0:
                    self.native_categories_by_sample.setdefault(sample_token, []).append(obj['object_type'])

        self._ann_tokens_by_track = ann_tokens_by_track
        self._sample_tokens = sample_tokens


class FusedEgoConverter(BaseNuscenesConverter):
    """V2V / V2I: fuses one reference vehicle's own LiDAR with one or more other sources (the
    other vehicle's LiDAR for v2v, and/or static infra LiDARs for v2i) into the reference
    vehicle's own raw sensor frame -- the exact frame SequenceConverter already emits. That way
    calibrated_sensor (vTl_reference) + ego_pose (gTv_reference, per frame) need no new logic;
    only the point-loading step changes, and the fused cloud stays inside the same egocentric
    distribution the checkpoint was trained on (rather than a recentered-global frame).

    Every other source's points are transformed into the reference's sensor frame per keyframe:
      other vehicle (v2v):  p_ref = inv(vTl_ref) @ inv(gTv_ref) @ gTv_other @ vTl_other @ p_other
      infra lidar   (v2i):  p_ref = inv(vTl_ref) @ inv(gTv_ref) @ gTl_infra @ p_infra

    Only the reference vehicle's own GT track is excluded (it can't detect itself; self-returns
    are already filtered by the norm>1.0 check in load_raw_lidar_points). A fused-in *other*
    vehicle (v2v) is kept as a legitimate GT target -- excluding it would remove exactly the
    detections v2v fusion exists to help with.
    """

    def __init__(self, root_folder, labels_folder, av_track_ids_path, sequence, reference,
                 fuse_vehicles, fuse_lidars, out_root, version, voxel_size):
        self.root_folder = Path(root_folder)
        self.sequence = sequence
        self.reference = reference
        self.fuse_vehicles = fuse_vehicles
        self.fuse_lidars = fuse_lidars
        self.voxel_size = voxel_size
        self.seq_dir = self.root_folder / 'dataset' / sequence

        self.lidar_channel = f'{reference}_middle_lidar'
        self.state_channel = f'{reference}_state'
        self.fuse_state_channels = {v: f'{v}_state' for v in fuse_vehicles}
        self.fuse_lidar_channels = {v: f'{v}_middle_lidar' for v in fuse_vehicles}

        self.table_dir = Path(out_root) / version
        self.samples_dir = Path(out_root) / 'samples' / 'LIDAR_TOP'
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        if not (self.table_dir / 'samples').exists():
            os.symlink('../samples', self.table_dir / 'samples')

        with open(self.seq_dir / 'calibration.json') as f:
            calib = json.load(f)
        self.vTl = np.asarray(calib[self.lidar_channel]['extrinsics']['vTl'])
        self.calib_translation = self.vTl[:3, 3]
        self.calib_rotation_matrix = self.vTl[:3, :3]

        self.vTl_by_vehicle = {
            v: np.asarray(calib[self.fuse_lidar_channels[v]]['extrinsics']['vTl']) for v in fuse_vehicles
        }
        self.gTl_by_lidar = {}
        for chan in fuse_lidars:
            extrinsics = calib[chan]['extrinsics']
            if 'gTl' not in extrinsics:
                raise ValueError(f"'{chan}' has no static 'gTl' extrinsic in calibration.json "
                                  f"(available: {list(extrinsics.keys())}) -- is it a vehicle-mounted sensor?")
            self.gTl_by_lidar[chan] = np.asarray(extrinsics['gTl'])

        name_bits = [reference]
        if fuse_vehicles:
            name_bits.append('v-' + '-'.join(fuse_vehicles))
        if fuse_lidars:
            name_bits.append('i-' + '-'.join(fuse_lidars))
        self.scene_name = f'{sequence}_fused_' + '_'.join(name_bits)

        with open(av_track_ids_path) as f:
            av_track_ids = json.load(f)
        self.ego_track_id = av_track_ids[sequence][reference]

        self.labels_by_ts = load_labels_by_timestamp(
            Path(labels_folder) / f'{sequence}.json', skip_track_id=self.ego_track_id)

        self.time_sync = pd.read_csv(self.seq_dir / 'timesync_info.csv').set_index('Unnamed: 0')

        self.category_tokens = {name: create_token() for name in sorted(set(OBJECT_TYPE_TO_NUSCENES.values()))}
        self.visibility_token = create_token()
        self.sensor_token = create_token()
        self.calibrated_sensor_token = create_token()
        self.log_token = create_token()
        self.scene_token = create_token()

        self.sample_table = []
        self.sample_data_table = []
        self.ego_pose_table = []
        self.sample_annotation_table = []
        self.instance_table = []
        self.native_categories_by_sample = {}

        self._convert()

    def _state_gTv(self, state_channel, state_filename):
        with open(self.seq_dir / state_channel / state_filename) as f:
            state = json.load(f)
        return np.asarray(state['gTv'])

    def _convert(self):
        columns = list(self.time_sync.columns)
        sample_tokens = [create_token() for _ in columns]
        pose_tokens = [create_token() for _ in columns]
        ann_tokens_by_track = {}
        for col in columns:
            ts_ms = float(self.time_sync[col]['timestamp_ms'])
            for obj in self.labels_by_ts.get(ts_ms / 1000.0, []):
                ann_tokens_by_track.setdefault(obj['track_id'], {})[int(ts_ms)] = create_token()

        for idx, col in enumerate(columns):
            row = self.time_sync[col]
            ts_ms = int(row['timestamp_ms'])
            ts_us = ts_ms * 1000
            sample_token = sample_tokens[idx]
            pose_token = pose_tokens[idx]

            gTv_ref = self._state_gTv(self.state_channel, row[self.state_channel])
            self.ego_pose_table.append({
                'token': pose_token,
                'timestamp': ts_us,
                'translation': gTv_ref[:3, 3].tolist(),
                'rotation': rotation_matrix_to_nuscenes_quaternion(gTv_ref[:3, :3]),
            })

            self.sample_table.append({
                'token': sample_token,
                'timestamp': ts_us,
                'prev': sample_tokens[idx - 1] if idx > 0 else '',
                'next': sample_tokens[idx + 1] if idx < len(columns) - 1 else '',
                'scene_token': self.scene_token,
            })

            # global_from_ref: maps the reference's own raw sensor frame -> global (same
            # composition SequenceConverter uses for its own ego). ref_from_global is its
            # inverse, and is what every other fused-in source gets projected through so the
            # whole fused cloud lands in the reference's sensor frame.
            global_from_ref = gTv_ref @ self.vTl
            ref_from_global = np.linalg.inv(global_from_ref)

            ref_points = load_raw_lidar_points(self.seq_dir, self.lidar_channel, row[self.lidar_channel])
            all_points = [ref_points]

            for v in self.fuse_vehicles:
                gTv_other = self._state_gTv(self.fuse_state_channels[v], row[self.fuse_state_channels[v]])
                global_from_other = gTv_other @ self.vTl_by_vehicle[v]
                other_to_ref = ref_from_global @ global_from_other
                other_points = load_raw_lidar_points(self.seq_dir, self.fuse_lidar_channels[v], row[self.fuse_lidar_channels[v]])
                other_xyz_ref = transform_points(other_points[:, :3], other_to_ref)
                all_points.append(np.concatenate([other_xyz_ref, other_points[:, 3:4]], axis=1).astype(np.float32))

            for chan in self.fuse_lidars:
                infra_to_ref = ref_from_global @ self.gTl_by_lidar[chan]
                infra_points = load_raw_lidar_points(self.seq_dir, chan, row[chan])
                infra_xyz_ref = transform_points(infra_points[:, :3], infra_to_ref)
                all_points.append(np.concatenate([infra_xyz_ref, infra_points[:, 3:4]], axis=1).astype(np.float32))

            fused_points = np.concatenate(all_points, axis=0)
            fused_points = voxel_downsample(fused_points, self.voxel_size)

            out_name = f'{self.sequence}_fused_{self.reference}_{ts_ms:06d}.pcd.bin'
            points_with_time = np.concatenate(
                [fused_points, np.zeros((fused_points.shape[0], 1), dtype=np.float32)], axis=1)
            points_with_time.tofile(self.samples_dir / out_name)

            sd_token = create_token()
            self.sample_data_table.append({
                'token': sd_token,
                'sample_token': sample_token,
                'ego_pose_token': pose_token,
                'calibrated_sensor_token': self.calibrated_sensor_token,
                'filename': f'samples/LIDAR_TOP/{out_name}',
                'fileformat': 'pcd',
                'timestamp': ts_us,
                'is_key_frame': True,
                'height': None,
                'width': None,
                'prev': '',
                'next': '',
            })
            self.sample_table[-1].setdefault('data', {})
            self.sample_table[-1]['data'] = {'LIDAR_TOP': sd_token}

            # Fused cloud, in global frame, only to count how many points fall inside each GT box.
            points_global = transform_points(fused_points[:, :3], global_from_ref)

            for obj in self.labels_by_ts.get(ts_ms / 1000.0, []):
                track_id = obj['track_id']
                l, w, h = obj['dimension']
                center = np.asarray(obj['position'])
                num_lidar_pts, gRobj = count_points_in_box(
                    points_global, obj['position'], obj['orientation'], obj['dimension'])

                ts_tokens = ann_tokens_by_track[track_id]
                ann_token = ts_tokens[ts_ms]
                sorted_ts = sorted(ts_tokens.keys())
                pos = sorted_ts.index(ts_ms)
                self.sample_annotation_table.append({
                    'token': ann_token,
                    'sample_token': sample_token,
                    'instance_token': self._instance_token(track_id),
                    'visibility_token': self.visibility_token,
                    'attribute_tokens': [],
                    'translation': center.tolist(),
                    'size': [w, l, h],
                    'rotation': rotation_matrix_to_nuscenes_quaternion(gRobj),
                    'prev': ts_tokens[sorted_ts[pos - 1]] if pos > 0 else '',
                    'next': ts_tokens[sorted_ts[pos + 1]] if pos < len(sorted_ts) - 1 else '',
                    'num_lidar_pts': num_lidar_pts,
                    'num_radar_pts': 0,
                })
                if num_lidar_pts > 0:
                    self.native_categories_by_sample.setdefault(sample_token, []).append(obj['object_type'])

        self._ann_tokens_by_track = ann_tokens_by_track
        self._sample_tokens = sample_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', required=True, help='dir containing dataset/ and labels/')
    parser.add_argument('--sequence', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ego', choices=['vehicle1', 'vehicle2'],
                       help='single vehicle-mounted lidar, no fusion')
    mode.add_argument('--lidars', type=str,
                       help='comma-separated static infra lidar channel names to fuse, '
                            'e.g. crossing2_11_lidar,crossing2_31_lidar')
    mode.add_argument('--fuse_ego', choices=['vehicle1', 'vehicle2'],
                       help='reference vehicle for a v2v/v2i fused egocentric capture -- combine '
                            'with --fuse_vehicle (v2v) and/or --fuse_lidars (v2i)')
    parser.add_argument('--fuse_vehicle', choices=['vehicle1', 'vehicle2'], default=None,
                         help='--fuse_ego mode only: the other vehicle-mounted lidar to fuse in (v2v)')
    parser.add_argument('--fuse_lidars', type=str, default=None,
                         help='--fuse_ego mode only: comma-separated infra lidar channel names to '
                              'fuse in (v2i), e.g. crossing2_11_lidar,crossing2_31_lidar')
    parser.add_argument('--voxel_size', type=float, default=0.1,
                         help='--lidars/--fuse_ego modes only: mean-pool points into voxels of this '
                              'size (meters) to bring fused density closer to nuScenes LIDAR_TOP; '
                              '0 disables it')
    parser.add_argument('--out', required=True, help='output nuscenes-format root')
    parser.add_argument('--version', default='v1.0-custom')
    args = parser.parse_args()

    if args.fuse_vehicle or args.fuse_lidars:
        if not args.fuse_ego:
            parser.error('--fuse_vehicle/--fuse_lidars require --fuse_ego')
        if args.fuse_vehicle == args.fuse_ego:
            parser.error('--fuse_vehicle must be the OTHER vehicle, not the same as --fuse_ego')
    elif args.fuse_ego:
        parser.error('--fuse_ego requires at least one of --fuse_vehicle or --fuse_lidars')

    if args.ego:
        converter = SequenceConverter(
            root_folder=args.root_folder,
            labels_folder=os.path.join(args.root_folder, 'labels'),
            av_track_ids_path=os.path.join(args.root_folder, 'labels_av_track_ids.json'),
            sequence=args.sequence,
            ego=args.ego,
            out_root=args.out,
            version=args.version,
        )
    elif args.lidars:
        converter = InfraFusionConverter(
            root_folder=args.root_folder,
            labels_folder=os.path.join(args.root_folder, 'labels'),
            sequence=args.sequence,
            lidars=[c.strip() for c in args.lidars.split(',') if c.strip()],
            out_root=args.out,
            version=args.version,
            voxel_size=args.voxel_size,
        )
    else:
        converter = FusedEgoConverter(
            root_folder=args.root_folder,
            labels_folder=os.path.join(args.root_folder, 'labels'),
            av_track_ids_path=os.path.join(args.root_folder, 'labels_av_track_ids.json'),
            sequence=args.sequence,
            reference=args.fuse_ego,
            fuse_vehicles=[args.fuse_vehicle] if args.fuse_vehicle else [],
            fuse_lidars=[c.strip() for c in args.fuse_lidars.split(',') if c.strip()] if args.fuse_lidars else [],
            out_root=args.out,
            version=args.version,
            voxel_size=args.voxel_size,
        )
    converter.write_tables()


if __name__ == '__main__':
    main()
