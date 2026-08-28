"""
Visualization for the late-fusion baseline (see eval_urbaning_v2x_late_fusion.py): renders each
frame's combined point cloud (every source's own points, transformed into the shared global frame
and concatenated -- the same visual convention early fusion's rendered scenes already use) with
the WBF-fused predicted boxes and the union-range GT boxes drawn on top, plus the same class-
agnostic 3D-IoU match sanity stats (precision/recall) visualize_urbaning_v2x.py computes for
single-source runs.

Reuses (imports, no duplication):
  - eval_urbaning_v2x_late_fusion.py: COMBOS, run_source_inference, weighted_box_fusion,
    source_to_global, load_gt_by_timestamp -- the exact same per-source inference, global-frame
    transform, and WBF merge this baseline's AP numbers are built from.
  - visualize_urbaning_v2x.py: save_scene / add_boxes / draw_box_labels (unmodified -- coarse
    CLASS_NAMES-id coloring/legend, now populated for fused boxes via weighted_box_fusion's
    optional coarse_labels output) and match_greedy (class-agnostic sanity matching).

Output goes to output/urbaning_v2x_late_fusion/<combo>/default/visualizations/ -- a separate tree
from early fusion's output/nuscenes_models/<tag>/default/visualizations/, so nothing overwrites
the Phase 1-3 renders.

Usage:
    xvfb-run -a python visualize_urbaning_v2x_late_fusion.py \
        --combo i2i_late \
        --root_folder ../datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --ckpt ../../pretrained/ov_scan_lidar.pth
"""
import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import open3d
import pandas as pd

from pcdet.config import cfg
from pcdet.utils import common_utils

from mlflow_logging import get_or_create_run
from eval_urbaning_v2x_late_fusion import (
    COMBOS, load_gt_by_timestamp, run_source_inference, source_to_global, source_to_local,
    weighted_box_fusion,
)
from visualize_urbaning_v2x import match_greedy, save_scene


def load_points_paths(dataset_root, version):
    """token -> absolute path to that frame's raw .pcd.bin, from sample_data.json's own filename
    field (dataset_root / filename, matching how convert_to_nuscenes.py wrote it)."""
    table_dir = Path(dataset_root) / version
    with open(table_dir / 'sample_data.json') as f:
        sample_data = json.load(f)
    return {sd['sample_token']: Path(dataset_root) / sd['filename'] for sd in sample_data}


def load_source_points_global(source, token, x_min, y_min, x_max, y_max):
    """This source's raw single-sweep points (x,y,z,intensity,timestamp -- see
    convert_to_nuscenes.py's points_with_time), cropped to this source's own local
    POINT_CLOUD_RANGE window (matching pcdet's unconditional -- train or eval --
    mask_points_and_boxes_outside_range point crop, data_processor.py:83-85, so this renders
    exactly what the model actually receives as input, not the raw un-cropped capture) and
    transformed into the shared global frame."""
    raw = np.fromfile(source['points_paths'][token], dtype=np.float32).reshape(-1, 5)
    in_range = (raw[:, 0] >= x_min) & (raw[:, 0] <= x_max) & (raw[:, 1] >= y_min) & (raw[:, 1] <= y_max)
    raw = raw[in_range]
    xyz_g, _ = source_to_global(source, token, raw[:, :3], np.zeros(len(raw)))
    return np.concatenate([xyz_g, raw[:, 3:4]], axis=1)


# UrbanIng-V2X object_type -> coarse CLASS_NAMES id (1-indexed, matching PRED_COLOR_MAP /
# CLASS_NAMES = ['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle',
# 'bicycle','pedestrian','traffic_cone']), for the optional --show_labels digit overlay only --
# mirrors the same Car/Van, Bus/Truck/OtherVehicle/Trailer, EScooter/Motorcycle/Cyclist,
# Pedestrian/OtherPedestrians grouping convert_to_nuscenes.py's OBJECT_TYPE_TO_NUSCENES already
# uses. 'Animal'/'Other'/'ignore' have no clean coarse bucket and fall back to 0 (draws as "?" in
# the legend, per draw_box_labels) -- their blue GT wireframe still renders correctly either way.
GT_OBJECT_TYPE_TO_CLASS_ID = {
    'Car': 1, 'Van': 1, 'Bus': 2, 'Truck': 2, 'OtherVehicle': 2, 'Trailer': 2,
    'EScooter': 8, 'Motorcycle': 8, 'Cyclist': 8, 'Pedestrian': 9, 'OtherPedestrians': 9,
}


def gt_boxes_for_frame(sources, gt_by_ts, idx, ts_ms, x_min, y_min, x_max, y_max):
    """Every GT track at this timestamp that falls inside *any* participating source's own local
    +-54m window (checked in that source's own frame) -- the same union-range policy
    eval_urbaning_v2x_late_fusion.py uses. Boxes are returned in the shared GLOBAL frame, with a
    trailing coarse-class-id column (see GT_OBJECT_TYPE_TO_CLASS_ID) for --show_labels."""
    gt_boxes = []
    for obj in gt_by_ts.get(ts_ms / 1000.0, []):
        p_global = np.asarray(obj['position'])
        in_range = False
        for source in sources:
            local = source_to_local(source, source['ordered_tokens'][idx], p_global)
            if x_min <= local[0] <= x_max and y_min <= local[1] <= y_max:
                in_range = True
                break
        if not in_range:
            continue
        l, w, h = obj['dimension']
        class_id = GT_OBJECT_TYPE_TO_CLASS_ID.get(obj['object_type'], 0)
        gt_boxes.append([*p_global, l, w, h, obj['orientation'], class_id])
    return np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 8), dtype=np.float32)


def compute_fixed_camera_params(center_xy, half_extent, width, height):
    """Runs open3d's own auto-fit (the same reset_view_point(True) call save_scene makes by
    default) ONCE, against two synthetic points spanning center_xy +- half_extent, and hands back
    the resulting camera. Every actual frame's save_scene() call then reuses this one fixed
    camera (see save_scene's fixed_cam_params) instead of auto-fitting to that frame's own
    point/box extent -- which is what made per-frame renders "wobble": a frame with one outlier
    far-away point or GT box would zoom out and re-center relative to neighboring frames."""
    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(np.array([
        [center_xy[0] - half_extent, center_xy[1] - half_extent, 0.0],
        [center_xy[0] + half_extent, center_xy[1] + half_extent, 0.0],
    ]))
    vis.add_geometry(pts)
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, 1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    vis.poll_events()
    vis.update_renderer()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    return cam_params


def parse_args():
    parser = argparse.ArgumentParser(description='Late-fusion (WBF) visualization for UrbanIng-V2X')
    parser.add_argument('--combo', type=str, required=True, choices=list(COMBOS.keys()))
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None,
                         help='output dir for images (default: output/urbaning_v2x_late_fusion/<combo>/default/visualizations)')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='min fused score for a box to be drawn')
    parser.add_argument('--match_iou_thresh', type=float, default=0.25,
                         help='3D IoU threshold, used for WBF clustering, GT range union, and match sanity stats')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1440)
    parser.add_argument('--point_size', type=float, default=0.7, help='open3d render point size (smaller = crisper lidar lines)')
    parser.add_argument('--zoom_margin', type=float, default=1.15,
                         help='multiplier applied to the 70th-percentile GT extent when picking the fixed camera radius')
    parser.add_argument('--min_half_extent', type=float, default=20.0,
                         help='minimum fixed-camera half-extent in meters, so a very tight GT cluster does not over-zoom')
    parser.add_argument('--max_half_extent', type=float, default=45.0,
                         help='maximum fixed-camera half-extent in meters -- caps the zoom at roughly the single-source '
                              '+-54m scale used throughout the rest of this project, even for combos whose union-range GT '
                              'legitimately spans further (a few far GT boxes may render off-screen; unaffected metrics-wise, '
                              'this only trims the debug visualization)')
    parser.add_argument('--show_labels', action='store_true',
                         help='overlay each box with a small class-id digit (same color as its wireframe) '
                              'plus a legend mapping digit -> class name, for both GT and predicted boxes '
                              '(same convention as visualize_urbaning_v2x.py)')
    parser.add_argument('--mlflow_max_artifacts', type=int, default=5)
    parser.add_argument('--mlflow_experiment', type=str, default='urbaning_v2x_late_fusion')
    parser.add_argument('--run_name', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = common_utils.create_logger()
    logger.info(f'----------------- UrbanIng-V2X late-fusion visualization: {args.combo} -----------------')

    sources = [dict(s) for s in COMBOS[args.combo]]
    frame_predictions = {}
    point_cloud_range = None
    class_names = None
    for source in sources:
        preds, pcr = run_source_inference(source, args.ckpt, logger, args.max_samples)
        frame_predictions[source['label']] = preds
        point_cloud_range = pcr
        class_names = cfg.CLASS_NAMES  # identical across all model configs -- captured after each load
        source['points_paths'] = load_points_paths(
            Path(cfg.ROOT_DIR) / 'tools' / cfg.DATA_CONFIG.DATA_PATH, cfg.DATA_CONFIG.VERSION)

    save_dir = Path(args.save_dir) if args.save_dir is not None else \
        Path(cfg.ROOT_DIR) / 'output' / 'urbaning_v2x_late_fusion' / args.combo / 'default' / 'visualizations'
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving visualizations to: {save_dir}')

    seq_dir = Path(args.root_folder) / 'dataset' / args.sequence
    time_sync = pd.read_csv(seq_dir / 'timesync_info.csv').set_index('Unnamed: 0')
    columns = list(time_sync.columns)
    gt_by_ts = load_gt_by_timestamp(Path(args.root_folder) / 'labels' / f'{args.sequence}.json')

    num_frames = len(columns) if args.max_samples is None else min(args.max_samples, len(columns))
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range

    # Pre-pass: compute every frame's (union-range-filtered) GT boxes once -- both to reuse below
    # and to derive a single FIXED camera (center + zoom) from the whole sequence's GT extent,
    # instead of auto-fitting per frame. Per-frame auto-fit is what made earlier renders "wobble":
    # whichever GT box or stray far-away point a given frame happened to contain would pull the
    # zoom/center around relative to its neighbors.
    frame_ts_ms = [int(time_sync[columns[idx]]['timestamp_ms']) for idx in range(num_frames)]
    gt_boxes_by_idx = [gt_boxes_for_frame(sources, gt_by_ts, idx, ts_ms, x_min, y_min, x_max, y_max)
                        for idx, ts_ms in enumerate(frame_ts_ms)]

    all_gt_xy = np.concatenate([b[:, :2] for b in gt_boxes_by_idx if len(b) > 0], axis=0)
    center_xy = all_gt_xy.mean(axis=0)
    dist_from_center = np.max(np.abs(all_gt_xy - center_xy), axis=1)  # Chebyshev radius per box
    # 70th percentile (not max/95th): a few sources' own +-54m windows can genuinely extend the
    # union-range GT much further than where most of the actual point-cloud density is (e.g. two
    # infra sensors ~33m apart each with their own 54m radius) -- fitting the camera to that full
    # tail reproduces the "mostly empty frame" problem this exists to fix. max_half_extent below is
    # a hard backstop for the same reason.
    half_extent = min(args.max_half_extent,
                       max(args.min_half_extent, args.zoom_margin * float(np.percentile(dist_from_center, 70))))
    fixed_cam = compute_fixed_camera_params(center_xy, half_extent, args.width, args.height)
    logger.info(f'Fixed camera: center=({center_xy[0]:.1f}, {center_xy[1]:.1f}), half_extent={half_extent:.1f}m')

    total_tp, total_fp, total_fn = 0, 0, 0
    per_frame_stats = []
    sample_viz_paths = []

    for idx in range(num_frames):
        ts_ms = frame_ts_ms[idx]

        all_boxes, all_scores, all_classes, all_labels, all_points = [], [], [], [], []
        for source in sources:
            token = source['ordered_tokens'][idx]
            pred_boxes, pred_scores, pred_class, pred_labels = frame_predictions[source['label']][token]
            all_points.append(load_source_points_global(source, token, x_min, y_min, x_max, y_max))
            if len(pred_boxes) == 0:
                continue
            xyz_g, yaw_g = source_to_global(source, token, pred_boxes[:, :3], pred_boxes[:, 6])
            boxes_g = np.concatenate([xyz_g, pred_boxes[:, 3:6], yaw_g[:, None]], axis=1)
            all_boxes.append(boxes_g)
            all_scores.append(pred_scores)
            all_classes.extend(pred_class)
            all_labels.append(pred_labels)
        points_global = np.concatenate(all_points, axis=0)
        boxes_cat = np.concatenate(all_boxes, axis=0) if all_boxes else np.zeros((0, 7), dtype=np.float32)
        scores_cat = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float32)
        labels_cat = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)
        fused_boxes, fused_scores, _, fused_labels = weighted_box_fusion(
            boxes_cat, scores_cat, all_classes, args.match_iou_thresh, coarse_labels=labels_cat)

        keep = fused_scores >= args.score_thresh
        fused_boxes, fused_scores, fused_labels = fused_boxes[keep], fused_scores[keep], fused_labels[keep]

        gt_boxes = gt_boxes_by_idx[idx]  # precomputed in the pre-pass above

        tp, fp, fn = match_greedy(fused_boxes, fused_scores, gt_boxes, args.match_iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_frame_stats.append({'idx': idx, 'tp': tp, 'fp': fp, 'fn': fn,
                                 'num_preds': len(fused_boxes), 'num_gt': len(gt_boxes)})

        save_path = save_dir / f'{idx:04d}.png'
        save_scene(points_global, gt_boxes, fused_boxes, fused_labels, save_path, args.width, args.height,
                   show_labels=args.show_labels, class_names=class_names,
                   point_size=args.point_size, fixed_cam_params=fixed_cam)
        if len(sample_viz_paths) < args.mlflow_max_artifacts:
            sample_viz_paths.append(save_path)

        logger.info(f'[{idx + 1}/{num_frames}] saved {save_path.name} '
                    f'({len(fused_boxes)} fused preds >= {args.score_thresh}, {len(gt_boxes)} gt boxes, '
                    f'match@IoU{args.match_iou_thresh}: tp={tp} fp={fp} fn={fn})')

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
    summary = {
        'combo': args.combo, 'sources': [s['label'] for s in sources], 'sequence': args.sequence,
        'score_thresh': args.score_thresh, 'match_iou_thresh': args.match_iou_thresh,
        'num_frames': len(per_frame_stats),
        'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn,
        'precision': precision, 'recall': recall,
        'note': 'Class-agnostic 3D-IoU sanity matching against WBF-fused, score-thresholded boxes, '
                'same convention as visualize_urbaning_v2x.py -- a localization check, not a '
                'class-aware benchmark (see eval_urbaning_v2x_late_fusion.py for that).',
        'per_frame': per_frame_stats,
    }
    summary_path = save_dir / 'match_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    run_name = args.run_name or f'{args.combo}__{args.sequence}'
    run = get_or_create_run(cfg.ROOT_DIR, args.mlflow_experiment, run_name, tags={
        'fusion_type': args.combo, 'sources': '+'.join(s['label'] for s in sources),
        'merge_strategy': 'late_wbf', 'sequence': args.sequence,
    })
    logger.info(f'MLflow run: {args.mlflow_experiment}/{run_name} ({run.info.run_id})')
    mlflow.log_params({'score_thresh': args.score_thresh})
    mlflow.log_metrics({'viz_precision': precision, 'viz_recall': recall,
                         'viz_tp': total_tp, 'viz_fp': total_fp, 'viz_fn': total_fn})
    mlflow.log_artifact(str(summary_path))
    for p in sample_viz_paths:
        mlflow.log_artifact(str(p), artifact_path='sample_visualizations')

    logger.info(f'Aggregate match@IoU{args.match_iou_thresh}: tp={total_tp} fp={total_fp} fn={total_fn} '
                f'precision={precision:.3f} recall={recall:.3f}')
    logger.info(f'Wrote match summary to: {summary_path}')
    logger.info('Done.')
    mlflow.end_run()


if __name__ == '__main__':
    main()
