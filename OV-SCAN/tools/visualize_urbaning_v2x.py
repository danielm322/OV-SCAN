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

import mlflow
import numpy as np
import open3d
import torch
from PIL import Image, ImageDraw, ImageFont

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils
from visual_utils.open3d_vis_utils import translate_boxes_to_open3d_instance

from mlflow_logging import add_common_args, get_or_create_run, intersection_for, run_name_for

# Bundled with matplotlib, present in the OV-SCAN image; falls back to PIL's bitmap font if not.
_LABEL_FONT_PATH = '/opt/conda/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf'

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
    parser.add_argument('--show_labels', action='store_true',
                         help='overlay each box with a small class-id digit (same color as its wireframe) '
                              'plus a legend mapping digit -> class name, for both GT and predicted boxes')
    parser.add_argument('--mlflow_max_artifacts', type=int, default=5,
                         help='max sample visualization PNGs to upload as MLflow artifacts (0 disables)')
    parser.add_argument('--data_path', type=str, default=None,
                         help='override DATA_CONFIG.DATA_PATH from --cfg_file (lets one fixed set '
                              'of 6 model configs be reused across many converted sequence '
                              'datasets, instead of one dataset yaml per sequence)')
    add_common_args(parser)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.data_path is not None:
        cfg.DATA_CONFIG.DATA_PATH = args.data_path
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


def _class_color_255(label):
    r, g, b = PRED_COLOR_MAP[int(label) % len(PRED_COLOR_MAP)]
    return int(r * 255), int(g * 255), int(b * 255)


def _project_point(point_xyz, intrinsic, extrinsic):
    """World xyz -> (u, v) pixel coords, using the exact camera params the scene was rendered
    with (pinhole projection, OpenCV/Open3D convention: extrinsic is world->camera, camera looks
    down +Z). Returns None if the point is behind the camera."""
    p_cam = extrinsic @ np.append(point_xyz, 1.0)
    if p_cam[2] <= 1e-6:
        return None
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    return fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy


def _rects_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_label(u, v, tw, th, pad, base_dy, placed_rects):
    """Find a (cx, cy) center for a tw x th (+pad) label near (u, v + base_dy) that doesn't
    overlap any rect already in placed_rects. Tries the preferred spot first (base_dy keeps GT
    above / predictions below their box, so a matched pair -- which projects to nearly the same
    point -- doesn't collide), then searches outward in expanding rings so labels from *different*
    nearby boxes (e.g. two adjacent pedestrians) also separate instead of overlapping. Falls back
    to the preferred spot if no free ring position exists within the search radius, rather than
    drifting arbitrarily far from the box it labels."""
    step = th + 2 * pad + 1
    sign = 1 if base_dy >= 0 else -1
    candidates = [(0, base_dy)]
    for ring in range(1, 7):
        d = ring * step
        candidates += [(0, base_dy + sign * d), (d, base_dy), (-d, base_dy),
                        (d, base_dy + sign * d), (-d, base_dy + sign * d)]
    for dx, dy in candidates:
        cx, cy = u + dx, v + dy
        rect = (cx - tw / 2 - pad, cy - th / 2 - pad, cx + tw / 2 + pad, cy + th / 2 + pad)
        if not any(_rects_overlap(rect, p) for p in placed_rects):
            return cx, cy, rect
    cx, cy = u, v + base_dy
    return cx, cy, (cx - tw / 2 - pad, cy - th / 2 - pad, cx + tw / 2 + pad, cy + th / 2 + pad)


def draw_box_labels(save_path, entries, class_names, intrinsic, extrinsic, width, height):
    """Overlay a small "P:<id>"/"G:<id>" label at each box's projected center, plus a legend.
    entries: list of (center_xyz, label_id, is_gt). The P:/G: prefix makes GT vs. predicted
    explicit on the label itself (in addition to the wireframe color distinction: blue for GT,
    class-colored for predictions, both noted in the legend). A matched GT/prediction pair sits
    at nearly the same projected point (that's what makes it a match), so GT labels default above
    the box center and prediction labels below it; on top of that, _place_label pushes any label
    further away from ones already drawn, so unrelated nearby boxes (e.g. two adjacent
    pedestrians) don't collide either -- otherwise a later label just paints over an earlier one
    and only one is ever visible.
    """
    img = Image.open(save_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    try:
        font = ImageFont.truetype(_LABEL_FONT_PATH, 10)
        legend_font = ImageFont.truetype(_LABEL_FONT_PATH, 10)
    except OSError:
        font = ImageFont.load_default()
        legend_font = font

    present_labels = set()
    placed_rects = []
    for center_xyz, label, is_gt in entries:
        uv = _project_point(center_xyz, intrinsic, extrinsic)
        if uv is None:
            continue
        u, v = uv
        if not (0 <= u < width and 0 <= v < height):
            continue
        text = f'{"G" if is_gt else "P"}:{int(label)}'
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        pad = 2
        base_dy = -(th + 2 * pad + 1) if is_gt else (th + 2 * pad + 1)
        cx, cy, rect = _place_label(u, v, tw, th, pad, base_dy, placed_rects)
        placed_rects.append(rect)
        draw.rectangle(rect, fill=(0, 0, 0, 190))
        draw.text((cx - tw / 2 - l, cy - th / 2 - t), text, fill=_class_color_255(label), font=font)
        present_labels.add(int(label))

    if present_labels:
        rows = sorted(present_labels)
        line_h = 15
        legend_h = 10 + line_h * (len(rows) + 2)  # +2 for the P:/G: and outline notes
        legend_w = 150
        draw.rectangle([8, 8, 8 + legend_w, 8 + legend_h], fill=(0, 0, 0, 170))
        for i, label in enumerate(rows):
            y = 8 + 5 + i * line_h
            name = class_names[label - 1] if 1 <= label <= len(class_names) else '?'
            draw.text((14, y), str(label), fill=_class_color_255(label), font=legend_font)
            draw.text((34, y), name, fill=(255, 255, 255), font=legend_font)
        y = 8 + 5 + len(rows) * line_h
        draw.text((14, y), 'P: / G:', fill=(255, 255, 255), font=legend_font)
        draw.text((72, y), 'prediction / ground truth', fill=(255, 255, 255), font=legend_font)
        y += line_h
        draw.text((14, y), 'GT', fill=tuple(int(c * 255) for c in GT_COLOR), font=legend_font)
        draw.text((44, y), 'outline = ground truth', fill=(255, 255, 255), font=legend_font)

    img.save(save_path)


def save_scene(points, gt_boxes, pred_boxes, pred_labels, save_path, width, height,
               show_labels=False, class_names=None, point_size=1.5, fixed_cam_params=None):
    """fixed_cam_params (optional, open3d.camera.PinholeCameraParameters): when given, use this
    exact camera instead of auto-fitting to this frame's own geometry -- callers that render many
    frames of the same scene (e.g. a video sequence) can compute this once (see
    visualize_urbaning_v2x_late_fusion.py's compute_fixed_camera_params) so the zoom/center stays
    stable across frames instead of jumping around with each frame's own point/box extent."""
    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = np.zeros(3)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)) * 0.6)
    vis.add_geometry(pts)

    if gt_boxes is not None and len(gt_boxes) > 0:
        add_boxes(vis, gt_boxes, GT_COLOR)
    if pred_boxes is not None and len(pred_boxes) > 0:
        add_boxes(vis, pred_boxes, None, labels=pred_labels)

    ctr = vis.get_view_control()
    if fixed_cam_params is not None:
        ctr.convert_from_pinhole_camera_parameters(fixed_cam_params)
    else:
        vis.reset_view_point(True)
        ctr.set_front([0.0, 0.0, 1.0])
        ctr.set_up([0.0, 1.0, 0.0])

    vis.poll_events()
    vis.update_renderer()

    cam_params = None
    if show_labels:
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam_params = (np.asarray(cam.intrinsic.intrinsic_matrix), np.asarray(cam.extrinsic))

    vis.capture_screen_image(str(save_path), do_render=True)
    vis.destroy_window()

    if show_labels and cam_params is not None:
        intrinsic, extrinsic = cam_params
        entries = []
        if gt_boxes is not None:
            for box in gt_boxes:
                entries.append((box[:3], box[-1], True))
        if pred_boxes is not None:
            for box, label in zip(pred_boxes, pred_labels):
                entries.append((box[:3], label, False))
        draw_box_labels(save_path, entries, class_names, intrinsic, extrinsic, width, height)


def filter_valid_gt_boxes(gt_boxes_padded):
    valid = np.any(gt_boxes_padded[:, :6] != 0, axis=1)
    return gt_boxes_padded[valid]


def filter_gt_boxes_in_range(gt_boxes, point_cloud_range):
    """pcdet only strips out-of-range GT boxes when training=True (mask_points_and_boxes_outside_range
    in data_processor.py checks `and self.training`); in eval/inference mode gt_boxes keeps every
    labeled track for the whole scene, including objects far outside any sensor's actual range (e.g.
    vehicles on approach roads well beyond the infra LiDARs' coverage of the intersection). Matching
    against those inflates FN and deflates recall for something no input point cloud could ever
    contain, so apply the same x/y range filter used on points before scoring/drawing GT.
    """
    if len(gt_boxes) == 0:
        return gt_boxes
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    in_range = (gt_boxes[:, 0] >= x_min) & (gt_boxes[:, 0] <= x_max) & \
               (gt_boxes[:, 1] >= y_min) & (gt_boxes[:, 1] <= y_max)
    return gt_boxes[in_range]


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

    run_name = run_name_for(args)
    intersection = intersection_for(args)
    run = get_or_create_run(cfg.ROOT_DIR, args.mlflow_experiment, run_name, tags={
        'fusion_type': args.fusion_type, 'sources': args.sources, 'reference': args.reference,
        **({'sequence': args.sequence} if args.sequence else {}),
        **({'intersection': intersection} if intersection else {}),
    })
    logger.info(f'MLflow run: {args.mlflow_experiment}/{run_name} ({run.info.run_id})')
    mlflow.log_params({
        'fusion_type': args.fusion_type, 'sources': args.sources, 'reference': args.reference,
        'cfg_file': args.cfg_file, 'ckpt': args.ckpt, 'score_thresh': args.score_thresh,
        'match_iou_thresh': args.match_iou_thresh,
        **({'sequence': args.sequence} if args.sequence else {}),
        **({'intersection': intersection} if intersection else {}),
    })

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
    sample_viz_paths = []

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
            gt_boxes = filter_gt_boxes_in_range(gt_boxes, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)

            tp, fp, fn = match_greedy(pred_boxes, pred_scores, gt_boxes, args.match_iou_thresh)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            per_frame_stats.append({'frame_id': frame_id, 'tp': tp, 'fp': fp, 'fn': fn,
                                     'num_preds': len(pred_boxes), 'num_gt': len(gt_boxes)})

            save_path = save_dir / f'{idx:04d}_{frame_id}.png'
            save_scene(points, gt_boxes, pred_boxes, pred_labels, save_path, args.width, args.height,
                       show_labels=args.show_labels, class_names=cfg.CLASS_NAMES)
            if len(sample_viz_paths) < args.mlflow_max_artifacts:
                sample_viz_paths.append(save_path)

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

    mlflow.log_metrics({
        'precision': precision, 'recall': recall,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
    })
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
