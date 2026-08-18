"""
Save per-sample prediction visualizations (point cloud + predicted/GT boxes) for the
NuScenes v1.0-mini val split, using the real dataset pipeline (multi-sweep lidar,
camera-aware config, etc.) instead of the raw-.bin demo.py flow.

Renders headlessly via open3d's legacy Visualizer(visible=False), which needs a
(possibly virtual) display. Run under Xvfb, e.g.:

    xvfb-run -a python visualize_nuscenes_mini.py \
        --cfg_file cfgs/nuscenes_models/ov_scan_lidar.yaml \
        --ckpt ../../pretrained/ov_scan_lidar.pth
"""
import argparse
from pathlib import Path

import numpy as np
import open3d
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
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
    parser = argparse.ArgumentParser(description='Save NuScenes v1.0-mini prediction visualizations')
    parser.add_argument('--cfg_file', type=str, required=True, help='dataset/model config')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--save_dir', type=str, default=None,
                         help='output dir for images (default: output/<exp>/<tag>/default/visualizations)')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='min score for a prediction to be drawn')
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


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- NuScenes v1.0-mini visualization -----------------')

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

            points = data_dict['points'].cpu().numpy()[:, 1:]
            gt_boxes = filter_valid_gt_boxes(gt_boxes_padded) if gt_boxes_padded is not None else None

            save_path = save_dir / f'{idx:04d}_{frame_id}.png'
            save_scene(points, gt_boxes, pred_boxes, pred_labels, save_path, args.width, args.height)

            logger.info(f'[{idx + 1}/{num_samples}] saved {save_path.name} '
                        f'({len(pred_boxes)} preds >= {args.score_thresh}, '
                        f'{0 if gt_boxes is None else len(gt_boxes)} gt boxes)')

    logger.info('Done.')


if __name__ == '__main__':
    main()
