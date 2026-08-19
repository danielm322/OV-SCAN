"""
Save per-sample prediction visualizations (point cloud + predicted/GT boxes) plus basic
class-agnostic 3D-IoU match statistics against GT, for a converted UrbanIng-V2X sequence
(see tools/urbaning_v2x/convert_to_nuscenes.py).

GT caveat: UrbanIng-V2X's own label->nuScenes category mapping is coarser than OV-SCAN's 10
classes (e.g. bus/truck/trailer/other-vehicle all collapse to "truck"), so GT boxes are matched
to predictions by 3D IoU only, ignoring class. This is a sanity check on localization, not a
class-aware detection benchmark.

Renders headlessly via open3d's legacy Visualizer(visible=False), which needs a (possibly
virtual) display. Run under Xvfb, e.g.:

    xvfb-run -a python visualize_urbaning_v2x.py \
        --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning.yaml \
        --ckpt ../../pretrained/ov_scan_lidar.pth
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils
from visual_utils.open3d_vis_utils import translate_boxes_to_open3d_instance

# index 0 unused (labels are 1-indexed); one color per CLASS_NAMES entry
PRED_COLOR_MAP = [
    (1.0, 1.0, 1.0),
    (0.89, 0.10, 0.11),  # car
    (0.22, 0.49, 0.72),  # truck
    (0.30, 0.69, 0.29),  # construction_vehicle
    (0.60, 0.31, 0.64),  # bus
    (1.00, 0.50, 0.00),  # trailer
    (1.00, 1.00, 0.20),  # barrier
    (0.65, 0.34, 0.16),  # motorcycle
    (0.97, 0.51, 0.75),  # bicycle
    (0.60, 0.60, 0.60),  # pedestrian
    (0.00, 0.80, 0.80),  # traffic_cone
]
GT_COLOR = (0.0, 0.4, 1.0)


def parse_config():
    parser = argparse.ArgumentParser(description='Save UrbanIng-V2X prediction visualizations')
    parser.add_argument('--cfg_file', type=str, required=True, help='dataset/model config')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--save_dir', type=str, default=None,
                         help='output dir for images (default: output/<exp>/<tag>/default/visualizations)')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='min score for a prediction to be drawn')
    parser.add_argument('--match_iou_thresh', type=float, default=0.25,
                         help='3D IoU threshold for greedy pred<->GT matching (class-agnostic)')
    parser.add_argument('--max_samples', type=int, default=None, help='limit number of samples (default: all)')
    parser.add_argument('--width', type=int, default=1280, help='render width in pixels')
    parser.add_argument('--height', type=int, default=960, help='render height in pixels')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    return args, cfg


def add_boxes(vis, boxes, color, labels=None):
    for i in range(boxes.shape[0]):
        line_set, _ = translate_boxes_to_open3d_instance(boxes[i])
        if labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(PRED_COLOR_MAP[int(labels[i]) % len(PRED_COLOR_MAP)])
        vis.add_geometry(line_set, reset_bounding_box=False)


def save_scene(points, gt_boxes, pred_boxes, pred_labels, save_path, width, height):
    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.get_render_option().point_size = 1.5
    vis.get_render_option().background_color = np.zeros(3)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)) * 0.6)
    vis.add_geometry(pts)

    if gt_boxes is not None and len(gt_boxes) > 0:
        add_boxes(vis, gt_boxes, GT_COLOR)
    if pred_boxes is not None and len(pred_boxes) > 0:
        add_boxes(vis, pred_boxes, None, labels=pred_labels)

    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, 1.0])
    ctr.set_up([0.0, 1.0, 0.0])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(save_path), do_render=True)
    vis.destroy_window()


def filter_valid_gt_boxes(gt_boxes_padded):
    valid = np.any(gt_boxes_padded[:, :6] != 0, axis=1)
    return gt_boxes_padded[valid]


def match_greedy(pred_boxes, pred_scores, gt_boxes, iou_thresh):
    """Class-agnostic greedy 3D-IoU matching. Returns (num_tp, num_fp, num_fn)."""
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    pred_t = torch.from_numpy(pred_boxes[:, :7]).float().cuda()
    gt_t = torch.from_numpy(gt_boxes[:, :7]).float().cuda()
    iou = iou3d_nms_utils.boxes_iou3d_gpu(pred_t, gt_t).cpu().numpy()  # (num_pred, num_gt)

    order = np.argsort(-pred_scores)
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    for i in order:
        candidate = np.where(~gt_used)[0]
        if len(candidate) == 0:
            break
        best = candidate[np.argmax(iou[i, candidate])]
        if iou[i, best] >= iou_thresh:
            gt_used[best] = True
            tp += 1
    fp = len(pred_boxes) - tp
    fn = int((~gt_used).sum())
    return tp, fp, fn


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- UrbanIng-V2X visualization -----------------')

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=2, logger=logger, training=False
    )
    logger.info(f'Total samples: {len(test_set)}')

    save_dir = Path(args.save_dir) if args.save_dir is not None else \
        cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'default' / 'visualizations'
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving visualizations to: {save_dir}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    num_samples = len(test_loader) if args.max_samples is None else min(args.max_samples, len(test_loader))

    total_tp, total_fp, total_fn = 0, 0, 0
    per_frame_stats = []

    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            if idx >= num_samples:
                break

            frame_id = data_dict['frame_id'][0]
            gt_boxes_padded = data_dict['gt_boxes'][0] if 'gt_boxes' in data_dict else None

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model(data_dict)

            pred = pred_dicts[0]
            keep = pred['pred_scores'].cpu().numpy() >= args.score_thresh
            pred_boxes = pred['pred_boxes'].cpu().numpy()[keep]
            pred_labels = pred['pred_labels'].cpu().numpy()[keep]
            pred_scores = pred['pred_scores'].cpu().numpy()[keep]

            points = data_dict['points'].cpu().numpy()[:, 1:]
            gt_boxes = filter_valid_gt_boxes(gt_boxes_padded) if gt_boxes_padded is not None else np.zeros((0, 7))

            tp, fp, fn = match_greedy(pred_boxes, pred_scores, gt_boxes, args.match_iou_thresh)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            per_frame_stats.append({'frame_id': frame_id, 'tp': tp, 'fp': fp, 'fn': fn,
                                     'num_preds': len(pred_boxes), 'num_gt': len(gt_boxes)})

            save_path = save_dir / f'{idx:04d}_{frame_id}.png'
            save_scene(points, gt_boxes, pred_boxes, pred_labels, save_path, args.width, args.height)

            logger.info(f'[{idx + 1}/{num_samples}] saved {save_path.name} '
                        f'({len(pred_boxes)} preds >= {args.score_thresh}, {len(gt_boxes)} gt boxes, '
                        f'match@IoU{args.match_iou_thresh}: tp={tp} fp={fp} fn={fn})')

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
    summary = {
        'score_thresh': args.score_thresh,
        'match_iou_thresh': args.match_iou_thresh,
        'num_frames': len(per_frame_stats),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'precision': precision,
        'recall': recall,
        'note': 'GT categories are coarser than OV-SCAN classes; matching is class-agnostic 3D IoU, '
                'a localization sanity check rather than a class-aware detection benchmark.',
        'per_frame': per_frame_stats,
    }
    summary_path = save_dir / 'match_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f'Aggregate match@IoU{args.match_iou_thresh}: tp={total_tp} fp={total_fp} fn={total_fn} '
                f'precision={precision:.3f} recall={recall:.3f}')
    logger.info(f'Wrote match summary to: {summary_path}')
    logger.info('Done.')


if __name__ == '__main__':
    main()
