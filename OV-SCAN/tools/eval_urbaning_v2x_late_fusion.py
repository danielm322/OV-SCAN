"""
Late-fusion baseline for UrbanIng-V2X: run OV-SCAN's frozen zero-shot pipeline independently per
agent (each agent stays inside its own valid +-54m POINT_CLOUD_RANGE window -- no distribution
shift, unlike early/input-level fusion), then merge each agent's *final 3D boxes* in one shared
global frame via score-weighted box fusion (WBF). See docs/Urbaning/Urbaning_PHASE_1.md through
_PHASE_3.md for the early-fusion pipeline this builds on -- in particular Phase 3 Sec.8, which found
that early fusion's single shared input window caps achievable range and that fixing it needs
either fine-tuning or a different fusion strategy, not a config change. Late fusion sidesteps that
ceiling entirely: no source's input ever leaves the distribution the checkpoint was trained on, and
no retraining is needed.

Design choices (confirmed with the user before implementing -- see the Phase 4 plan):
  - Combos mirror Phase 2/3's 6 fusion configs, minus the v2v reference-frame duplication (late
    fusion has no reference-frame concept -- everything lands in one global frame regardless of
    which agent "hosts" it), plus one new "everything" combo to specifically probe range extension.
  - Merge algorithm: greedy score-ordered Weighted Box Fusion (score-weighted mean geometry within
    an IoU cluster), not plain NMS.
  - Only boxes agreeing on fine OV native class (pred['ov_str_labels']) are ever merged together.
  - GT policy: no track self-exclusion (every real track is a fair target, uniformly across combos).
  - GT range policy: a GT box is kept if it falls inside *any* participating source's own local
    +-54m window for that frame (checked in that source's own frame) -- the concrete test of
    whether pooling agents recovers coverage a single window can't, which early fusion couldn't do
    without retraining.

Per-class AP methodology (class_ap/voc_ap/FINE_CLASSES/GROUPS/CANONICAL_TO_GROUP/
PRED_NATIVE_TO_CANONICAL/GT_NATIVE_TO_CANONICAL) is imported directly from
eval_urbaning_v2x_per_class.py, unchanged -- so mAP numbers here are directly comparable to Phase
2/3's tables; only the box population (fused across sources vs single-source) differs.

Usage:
    xvfb-run -a python eval_urbaning_v2x_late_fusion.py \
        --combo i2i_late \
        --root_folder ../datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --ckpt ../../pretrained/ov_scan_lidar.pth
"""
import argparse
import json
import re
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils

from mlflow_logging import get_or_create_run
from eval_urbaning_v2x_per_class import (
    CANONICAL_TO_GROUP, FINE_CLASSES, GROUPS, GT_NATIVE_TO_CANONICAL, PRED_NATIVE_TO_CANONICAL,
    class_ap,
)

# Per-intersection infra LiDAR channel names (crossing3 uses 21/22, not 31/32). Duplicated from
# run_full_dataset_eval.py's INFRA_CHANNELS -- that module is a host-side orchestrator script
# (shells out to docker/7z), not meant to be imported into this container-side eval code.
INFRA_CHANNELS = {
    'crossing1': ['crossing1_11_lidar', 'crossing1_12_lidar', 'crossing1_31_lidar', 'crossing1_32_lidar'],
    'crossing2': ['crossing2_11_lidar', 'crossing2_12_lidar', 'crossing2_31_lidar', 'crossing2_32_lidar'],
    'crossing3': ['crossing3_11_lidar', 'crossing3_12_lidar', 'crossing3_21_lidar', 'crossing3_22_lidar'],
}

VEHICLE_CFG = 'cfgs/nuscenes_models/ov_scan_lidar_urbaning.yaml'
INFRA_CFG = 'cfgs/nuscenes_models/ov_scan_lidar_urbaning_infra.yaml'

VEHICLE1 = {'label': 'vehicle1', 'mode': 'vehicle', 'cfg_file': VEHICLE_CFG,
            'data_path': '../../datasets/urbaning_v2x'}
VEHICLE2 = {'label': 'vehicle2', 'mode': 'vehicle', 'cfg_file': VEHICLE_CFG,
            'data_path': '../../datasets/urbaning_v2x_vehicle2'}
INFRA_11 = {'label': 'crossing2_11', 'mode': 'infra', 'cfg_file': INFRA_CFG,
            'data_path': '../../datasets/urbaning_v2x_infra_solo_11'}
INFRA_12 = {'label': 'crossing2_12', 'mode': 'infra', 'cfg_file': INFRA_CFG,
            'data_path': '../../datasets/urbaning_v2x_infra_solo_12'}
INFRA_31 = {'label': 'crossing2_31', 'mode': 'infra', 'cfg_file': INFRA_CFG,
            'data_path': '../../datasets/urbaning_v2x_infra_solo_31'}
INFRA_32 = {'label': 'crossing2_32', 'mode': 'infra', 'cfg_file': INFRA_CFG,
            'data_path': '../../datasets/urbaning_v2x_infra_solo_32'}
INFRA_ALL = [INFRA_11, INFRA_12, INFRA_31, INFRA_32]

# Named combos, mirroring Phase 2/3's 6 configs (v2v collapses from 2 reference-frame variants to
# 1, since late fusion has no reference-frame concept) plus one new "everything" combo.
COMBOS = {
    'i2i_late': INFRA_ALL,
    'v2v_late': [VEHICLE1, VEHICLE2],
    'v2i_v1_late': [VEHICLE1] + INFRA_ALL,
    'v2i_v2_late': [VEHICLE2] + INFRA_ALL,
    'full_late': [VEHICLE1, VEHICLE2] + INFRA_ALL,
}


def combo_sources_for(intersection, data_root):
    """Generalizes the module-level COMBOS (hardcoded to the single crossing2 reference sequence
    Phase 4 was developed against) to an arbitrary intersection/sequence, pointing at solo-source
    datasets converted under data_root/{vehicle1_solo, vehicle2_solo, infra_solo_<channel>}.
    Source dicts are shared BY REFERENCE across every combo that uses them (not copied per combo),
    so a single run_all_sources_inference pass over unique_sources(...) populates each source's
    poses/offset/ordered_tokens once and every combo sees it."""
    vehicle1 = {'label': 'vehicle1', 'mode': 'vehicle', 'cfg_file': VEHICLE_CFG,
                'data_path': f'{data_root}/vehicle1_solo'}
    vehicle2 = {'label': 'vehicle2', 'mode': 'vehicle', 'cfg_file': VEHICLE_CFG,
                'data_path': f'{data_root}/vehicle2_solo'}
    infra_all = [{'label': ch.rsplit('_', 1)[0], 'mode': 'infra', 'cfg_file': INFRA_CFG,
                  'data_path': f"{data_root}/infra_solo_{ch.split('_')[1]}"}
                 for ch in INFRA_CHANNELS[intersection]]
    return {
        'i2i_late': infra_all,
        'v2v_late': [vehicle1, vehicle2],
        'v2i_v1_late': [vehicle1] + infra_all,
        'v2i_v2_late': [vehicle2] + infra_all,
        'full_late': [vehicle1, vehicle2] + infra_all,
    }


def unique_sources(combos):
    """De-duplicates source dicts shared by reference across multiple combos (see
    combo_sources_for), by label, for a single run_all_sources_inference pass covering every combo
    in `combos` at once."""
    by_label = {}
    for sources in combos.values():
        for s in sources:
            by_label[s['label']] = s
    return list(by_label.values())


def load_gt_by_timestamp(labels_path):
    """Every real track, no self-exclusion (this pipeline's chosen GT policy) -- unlike
    convert_to_nuscenes.py's load_labels_by_timestamp, which optionally skips one ego track."""
    with open(labels_path) as f:
        labels = json.load(f)
    by_ts = {}
    for track in labels['tracks']:
        for i, ts in enumerate(track['timestamps']):
            by_ts.setdefault(ts, []).append({
                'object_type': track['object_type'],
                'position': track['positions'][i],
                'orientation': track['orientations'][i],
                'dimension': track['dimensions'][0] if len(track['dimensions']) == 1 else track['dimensions'][i],
            })
    return by_ts


def load_source_frame_index(dataset_root, version):
    """sample.json is written in the same per-keyframe-column order every converter mode iterates
    in (see convert_to_nuscenes.py), so file order IS the shared frame index across every source
    of the same sequence -- this is what lets predictions from differently-tokened converted
    datasets be aligned frame-by-frame."""
    table_dir = Path(dataset_root) / version
    with open(table_dir / 'sample.json') as f:
        return [s['token'] for s in json.load(f)]


def load_vehicle_poses(dataset_root, version):
    """token -> (R, t) global_from_sensor, composed the same way nuscenes-devkit does:
    ego_pose (gTv, per-frame) @ calibrated_sensor (vTl, fixed)."""
    table_dir = Path(dataset_root) / version
    with open(table_dir / 'sample_data.json') as f:
        sample_data = {sd['sample_token']: sd for sd in json.load(f)}
    with open(table_dir / 'ego_pose.json') as f:
        ego_pose = {e['token']: e for e in json.load(f)}
    with open(table_dir / 'calibrated_sensor.json') as f:
        calib = {e['token']: e for e in json.load(f)}
    poses = {}
    for tok, sd in sample_data.items():
        ep = ego_pose[sd['ego_pose_token']]
        cs = calib[sd['calibrated_sensor_token']]
        R_ep = Rotation.from_quat(ep['rotation'], scalar_first=True).as_matrix()
        t_ep = np.asarray(ep['translation'])
        R_cs = Rotation.from_quat(cs['rotation'], scalar_first=True).as_matrix()
        t_cs = np.asarray(cs['translation'])
        poses[tok] = (R_ep @ R_cs, R_ep @ t_cs + t_ep)
    return poses


def load_infra_offset(dataset_root, version):
    table_dir = Path(dataset_root) / version
    with open(table_dir / 'frame_offset.json') as f:
        return np.asarray(json.load(f)['offset'])


def source_to_global(source, token, xyz, yaw):
    """Local (sensor frame for vehicle sources, recentered-global for infra sources) -> true
    global. xyz: (N,3), yaw: (N,). Boxes are assumed near-upright (yaw-only local rotation), the
    same assumption count_points_in_box/rotation_matrix_to_nuscenes_quaternion make elsewhere in
    this pipeline."""
    if source['mode'] == 'vehicle':
        R, t = source['poses'][token]
        xyz_g = xyz @ R.T + t
        R_obj = Rotation.from_euler('z', yaw).as_matrix()
        R_obj_g = R[None] @ R_obj
        yaw_g = Rotation.from_matrix(R_obj_g).as_euler('zyx')[:, 0]
        return xyz_g, yaw_g
    return xyz + source['offset'], yaw


def source_to_local(source, token, xyz_g):
    """Inverse of source_to_global's translation+rotation, position only (used for the GT
    per-source range check, where orientation doesn't matter)."""
    if source['mode'] == 'vehicle':
        R, t = source['poses'][token]
        return (xyz_g - t) @ R
    return xyz_g - source['offset']


def run_source_inference(source, ckpt, logger, max_samples=None):
    """Builds this source's own dataloader+model, runs inference over every frame, and returns
    (predictions_by_token, point_cloud_range). Loads pose/offset metadata into `source` in place."""
    cfg_from_yaml_file(source['cfg_file'], cfg)
    cfg.DATA_CONFIG.DATA_PATH = source['data_path']
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=2, logger=logger, training=False)

    dataset_root = Path(cfg.ROOT_DIR) / 'tools' / cfg.DATA_CONFIG.DATA_PATH
    version = cfg.DATA_CONFIG.VERSION
    if source['mode'] == 'vehicle':
        source['poses'] = load_vehicle_poses(dataset_root, version)
    else:
        source['offset'] = load_infra_offset(dataset_root, version)
    source['ordered_tokens'] = load_source_frame_index(dataset_root, version)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    num_samples = len(test_loader) if max_samples is None else min(max_samples, len(test_loader))
    predictions_by_token = {}
    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            if idx >= num_samples:
                break
            token = data_dict['metadata'][0]['token']
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model(data_dict)
            pred = pred_dicts[0]
            pred_boxes = pred['pred_boxes'].cpu().numpy()
            pred_scores = pred['pred_scores'].cpu().numpy()
            pred_class = [PRED_NATIVE_TO_CANONICAL.get(s) for s in pred['ov_str_labels']]
            # Coarse CLASS_NAMES id (1-indexed) -- unused by AP scoring (which matches on fine OV
            # native class only) but carried through for visualize_urbaning_v2x_late_fusion.py,
            # which reuses visualize_urbaning_v2x.py's existing coarse-label color/legend scheme.
            pred_coarse_labels = pred['pred_labels'].cpu().numpy()
            predictions_by_token[token] = (pred_boxes, pred_scores, pred_class, pred_coarse_labels)
    logger.info(f"[{source['label']}] {len(predictions_by_token)} frames inferred")
    return predictions_by_token, cfg.DATA_CONFIG.POINT_CLOUD_RANGE


def weighted_box_fusion(boxes, scores, classes, iou_thresh, coarse_labels=None):
    """Greedy score-ordered clustering by 3D IoU, same fine OV native class only. Fused box =
    score-weighted mean of [x,y,z,dx,dy,dz], circular score-weighted mean of yaw. Fused score =
    mean of cluster member scores (the simple default -- not tuned against alternatives such as a
    cluster-size confidence bonus).

    coarse_labels (optional, parallel array to boxes/scores/classes): when given, each fused
    cluster also gets a fused_labels entry -- the *highest-scoring cluster member's* coarse
    CLASS_NAMES id, for callers that want to reuse a coarse-label-keyed color scheme (per-class AP
    scoring itself never uses this, only fine `classes`). None (default) skips this and returns
    None in that slot, unchanged from before this parameter existed."""
    n = len(boxes)
    if n == 0:
        empty_labels = np.zeros(0, dtype=np.int64) if coarse_labels is not None else None
        return boxes.reshape(0, 7).astype(np.float32), scores.astype(np.float32), [], empty_labels

    classes_arr = np.asarray(classes, dtype=object)
    boxes_t = torch.from_numpy(boxes[:, :7]).float().cuda()
    iou = iou3d_nms_utils.boxes_iou3d_gpu(boxes_t, boxes_t).cpu().numpy()
    iou = np.where(classes_arr[:, None] == classes_arr[None, :], iou, 0.0)

    order = np.argsort(-scores)
    used = np.zeros(n, dtype=bool)
    fused_boxes, fused_scores, fused_classes, fused_labels = [], [], [], []
    for i in order:
        if used[i]:
            continue
        cluster = np.where((iou[i] >= iou_thresh) & ~used)[0]
        cluster = np.union1d(cluster, [i])
        used[cluster] = True
        w = scores[cluster]
        wsum = w.sum()
        center_dims = (boxes[cluster, :6] * w[:, None]).sum(axis=0) / wsum
        sin_yaw = float((np.sin(boxes[cluster, 6]) * w).sum() / wsum)
        cos_yaw = float((np.cos(boxes[cluster, 6]) * w).sum() / wsum)
        yaw = np.arctan2(sin_yaw, cos_yaw)
        fused_boxes.append(np.concatenate([center_dims, [yaw]]))
        fused_scores.append(float(w.mean()))
        fused_classes.append(classes_arr[i])
        if coarse_labels is not None:
            fused_labels.append(int(coarse_labels[i]))
    fused_labels_arr = np.array(fused_labels, dtype=np.int64) if coarse_labels is not None else None
    return np.array(fused_boxes, dtype=np.float32), np.array(fused_scores, dtype=np.float32), fused_classes, fused_labels_arr


def run_all_sources_inference(sources, ckpt, logger, max_samples=None):
    """Runs run_source_inference once per source (mutating each source dict in place with
    poses/offset/ordered_tokens) and returns ({label: predictions_by_token}, point_cloud_range).

    Call this ONCE on the de-duplicated union of sources across every combo being evaluated (see
    unique_sources) -- each combo's fuse_combo_frames below then reuses the same cached
    predictions, instead of re-inferring a source once per combo it happens to appear in (up to 3x
    redundant for vehicle1/vehicle2, up to 4x for each infra channel -- see
    eval_urbaning_v2x_late_fusion_batch.py, which is what this split exists for)."""
    frame_predictions = {}
    point_cloud_range = None
    for source in sources:
        preds, pcr = run_source_inference(source, ckpt, logger, max_samples)
        frame_predictions[source['label']] = preds
        point_cloud_range = pcr  # identical across all model configs (Phase 1 SS5)
    return frame_predictions, point_cloud_range


def fuse_combo_frames(combo_sources, frame_predictions, point_cloud_range, root_folder, sequence,
                       wbf_iou_thresh, logger, max_samples=None):
    """WBF-fuses one combo's sources (a subset of what run_all_sources_inference already ran,
    cached in frame_predictions) into per-frame fused predictions, plus that combo's own
    union-range GT. Returns a `frames` list (pred_boxes/pred_scores/pred_class/gt_boxes/gt_class
    per frame) in exactly the structure class_ap expects."""
    seq_dir = Path(root_folder) / 'dataset' / sequence
    time_sync = pd.read_csv(seq_dir / 'timesync_info.csv').set_index('Unnamed: 0')
    columns = list(time_sync.columns)
    gt_by_ts = load_gt_by_timestamp(Path(root_folder) / 'labels' / f'{sequence}.json')

    num_frames = len(columns) if max_samples is None else min(max_samples, len(columns))
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range

    frames = []
    for idx in range(num_frames):
        ts_ms = int(time_sync[columns[idx]]['timestamp_ms'])

        all_boxes, all_scores, all_classes = [], [], []
        for source in combo_sources:
            token = source['ordered_tokens'][idx]
            pred_boxes, pred_scores, pred_class, _ = frame_predictions[source['label']][token]
            if len(pred_boxes) == 0:
                continue
            xyz_g, yaw_g = source_to_global(source, token, pred_boxes[:, :3], pred_boxes[:, 6])
            boxes_g = np.concatenate([xyz_g, pred_boxes[:, 3:6], yaw_g[:, None]], axis=1)
            all_boxes.append(boxes_g)
            all_scores.append(pred_scores)
            all_classes.extend(pred_class)
        boxes_cat = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 7), dtype=np.float32)
        scores_cat = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float32)
        fused_boxes, fused_scores, fused_classes, _ = weighted_box_fusion(
            boxes_cat, scores_cat, all_classes, wbf_iou_thresh)

        gt_boxes, gt_class = [], []
        for obj in gt_by_ts.get(ts_ms / 1000.0, []):
            p_global = np.asarray(obj['position'])
            in_range = False
            for source in combo_sources:
                local = source_to_local(source, source['ordered_tokens'][idx], p_global)
                if x_min <= local[0] <= x_max and y_min <= local[1] <= y_max:
                    in_range = True
                    break
            if not in_range:
                continue
            l, w, h = obj['dimension']
            gt_boxes.append([*p_global, l, w, h, obj['orientation']])
            gt_class.append(GT_NATIVE_TO_CANONICAL.get(obj['object_type']))
        gt_boxes = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 7), dtype=np.float32)

        frames.append({
            'pred_boxes': fused_boxes, 'pred_scores': fused_scores, 'pred_class': fused_classes,
            'gt_boxes': gt_boxes, 'gt_class': gt_class,
        })
        logger.info(f'[{idx + 1}/{num_frames}] {len(boxes_cat)} raw preds -> {len(fused_boxes)} fused, '
                    f'{len(gt_boxes)} gt boxes in range (union of {len(combo_sources)} sources)')
    return frames


def build_late_fusion_frames(combo, root_folder, sequence, ckpt, wbf_iou_thresh, logger, max_samples=None):
    """Backward-compatible single-combo wrapper (used by main() below and
    threshold_sensitivity_late_fusion.py): runs inference for just this combo's own sources, then
    fuses. When evaluating multiple combos on the same sequence, prefer calling
    run_all_sources_inference once on unique_sources(combos) and fuse_combo_frames per combo
    instead, to avoid redundant inference on shared sources -- see
    eval_urbaning_v2x_late_fusion_batch.py."""
    sources = [dict(s) for s in COMBOS[combo]]  # copy: run_source_inference mutates in place
    frame_predictions, point_cloud_range = run_all_sources_inference(sources, ckpt, logger, max_samples)
    frames = fuse_combo_frames(sources, frame_predictions, point_cloud_range, root_folder, sequence,
                                wbf_iou_thresh, logger, max_samples)
    return frames, sources


def compute_ap_sweep(frames, primary_iou, extra_ious, logger=None):
    """Computes per-class/group AP + mAP at `primary_iou` and each of `extra_ious`, all from the
    SAME already-fused `frames` (no extra inference or WBF re-fusion) -- mirrors
    eval_urbaning_v2x_per_class.py's compute_ap_at sweep pattern, reused here so single-combo and
    full-dataset-batch callers share one implementation. Returns (summary_dict, mlflow_metrics_dict);
    summary_dict's 'by_iou_thresh' key holds the extra thresholds' results, mirroring
    eval_urbaning_v2x_per_class.py's on-disk JSON shape."""
    def compute_ap_at(iou_thresh):
        fine = {cls: class_ap(frames, cls, iou_thresh, class_of=lambda c: c) for cls in FINE_CLASSES}
        group = {grp: class_ap(frames, grp, iou_thresh, class_of=lambda c: CANONICAL_TO_GROUP.get(c))
                 for grp in GROUPS}
        valid_fine = [r['ap'] for r in fine.values() if r['num_gt'] > 0]
        valid_group = [r['ap'] for r in group.values() if r['num_gt'] > 0]
        mAP_fine = float(np.mean(valid_fine)) if valid_fine else float('nan')
        mAP_groups = float(np.mean(valid_group)) if valid_group else float('nan')
        return fine, group, mAP_fine, mAP_groups

    fine_results, group_results, mAP_fine, mAP_groups = compute_ap_at(primary_iou)
    summary = {
        'match_iou_thresh': primary_iou, 'num_frames': len(frames),
        'fine_classes': fine_results, 'groups': group_results,
        'mAP_fine_classes': mAP_fine, 'mAP_groups': mAP_groups, 'by_iou_thresh': {},
    }
    mlflow_metrics = {'mAP_fine_classes': mAP_fine, 'mAP_groups': mAP_groups}
    for cls, r in fine_results.items():
        if r['num_gt'] > 0:
            mlflow_metrics[f'ap_fine_{cls}'] = r['ap']
    for grp, r in group_results.items():
        if r['num_gt'] > 0:
            mlflow_metrics[f'ap_group_{grp}'] = r['ap']

    for iou_thresh in extra_ious:
        fine_x, group_x, mAP_fine_x, mAP_groups_x = compute_ap_at(iou_thresh)
        suffix = f'_iou{iou_thresh}'
        summary['by_iou_thresh'][str(iou_thresh)] = {
            'fine_classes': fine_x, 'groups': group_x,
            'mAP_fine_classes': mAP_fine_x, 'mAP_groups': mAP_groups_x,
        }
        mlflow_metrics[f'mAP_fine_classes{suffix}'] = mAP_fine_x
        mlflow_metrics[f'mAP_groups{suffix}'] = mAP_groups_x
        for cls, r in fine_x.items():
            if r['num_gt'] > 0:
                mlflow_metrics[f'ap_fine_{cls}{suffix}'] = r['ap']
        for grp, r in group_x.items():
            if r['num_gt'] > 0:
                mlflow_metrics[f'ap_group_{grp}{suffix}'] = r['ap']
        if logger:
            logger.info(f'-- IoU={iou_thresh}: mAP_fine={mAP_fine_x:.3f}  mAP_groups={mAP_groups_x:.3f} --')
    return summary, mlflow_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Late-fusion (WBF) baseline for UrbanIng-V2X')
    parser.add_argument('--combo', type=str, required=True, choices=list(COMBOS.keys()))
    parser.add_argument('--root_folder', type=str, required=True,
                         help='dir containing dataset/<sequence>/timesync_info.csv and labels/<sequence>.json')
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--match_iou_thresh', type=float, default=0.25,
                         help='3D IoU threshold, used both for WBF clustering and the primary per-class AP '
                              'matching pass -- metrics at this threshold keep unsuffixed names')
    parser.add_argument('--extra_match_iou_threshs', type=str, default='0.3,0.5',
                         help='comma-separated additional AP-matching IoU thresholds, swept from the same '
                              'WBF-fused frames (WBF clustering itself stays at --match_iou_thresh) -- '
                              'logged/saved with a "_iou<thresh>" suffix. Empty string to disable.')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--mlflow_experiment', type=str, default='urbaning_v2x_late_fusion')
    parser.add_argument('--run_name', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = common_utils.create_logger()
    logger.info(f'----------------- UrbanIng-V2X late fusion: {args.combo} -----------------')

    frames, sources = build_late_fusion_frames(
        args.combo, args.root_folder, args.sequence, args.ckpt, args.match_iou_thresh, logger, args.max_samples)

    extra_threshs = [float(x) for x in args.extra_match_iou_threshs.split(',') if x.strip()]
    logger.info(f'Computing per-class AP at IoU={args.match_iou_thresh} (primary)'
                + (f' and IoU={extra_threshs} (extra)' if extra_threshs else '') + '...')
    summary, mlflow_metrics = compute_ap_sweep(frames, args.match_iou_thresh, extra_threshs, logger)
    summary['combo'] = args.combo
    summary['sources'] = [s['label'] for s in sources]
    summary['sequence'] = args.sequence
    fine_results, group_results = summary['fine_classes'], summary['groups']
    mAP_fine, mAP_groups = summary['mAP_fine_classes'], summary['mAP_groups']

    save_path = Path(args.save_path) if args.save_path is not None else \
        Path(cfg.ROOT_DIR) / 'output' / 'urbaning_v2x_late_fusion' / args.combo / 'default' / 'per_class_ap.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    intersection = re.search(r'crossing\d+', args.sequence)
    intersection = intersection.group(0) if intersection else None
    run_name = args.run_name or f'{args.combo}__{args.sequence}'
    run = get_or_create_run(cfg.ROOT_DIR, args.mlflow_experiment, run_name, tags={
        'fusion_type': args.combo, 'sources': '+'.join(s['label'] for s in sources),
        'merge_strategy': 'late_wbf', 'sequence': args.sequence,
        **({'intersection': intersection} if intersection else {}),
    })
    logger.info(f'MLflow run: {args.mlflow_experiment}/{run_name} ({run.info.run_id})')
    mlflow.log_params({
        'combo': args.combo, 'sources': '+'.join(s['label'] for s in sources),
        'merge_strategy': 'late_wbf', 'sequence': args.sequence, 'ckpt': args.ckpt,
        'match_iou_thresh': args.match_iou_thresh, 'extra_match_iou_threshs': args.extra_match_iou_threshs,
        **({'intersection': intersection} if intersection else {}),
    })
    mlflow.log_metrics(mlflow_metrics)
    mlflow.log_artifact(str(save_path))

    logger.info('=== Fine-grained native classes ===')
    for cls, r in fine_results.items():
        logger.info(f'{cls:>16s}: AP={r["ap"]:.3f}  num_gt={r["num_gt"]:5d}  num_pred={r["num_pred"]:5d}'
                    if r['num_gt'] > 0 else f'{cls:>16s}: no GT in this run')
    logger.info(f'mAP (fine classes with GT): {mAP_fine:.3f}')
    logger.info('=== UrbanIng-V2X label groups ===')
    for grp, r in group_results.items():
        logger.info(f'{grp:>16s}: AP={r["ap"]:.3f}  num_gt={r["num_gt"]:5d}  num_pred={r["num_pred"]:5d}'
                    if r['num_gt'] > 0 else f'{grp:>16s}: no GT in this run')
    logger.info(f'mAP (groups): {mAP_groups:.3f}')
    logger.info(f'Wrote {save_path}')
    logger.info('Done.')
    mlflow.end_run()


if __name__ == '__main__':
    main()
