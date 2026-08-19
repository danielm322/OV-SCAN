"""
Per-class AP / mAP evaluation for OV-SCAN inference on a converted UrbanIng-V2X sequence,
exploiting the model's open-vocabulary CLIP alignment head.

How the open-vocab exploitation works: OV-SCAN's final label for every detection comes from a
frozen-CLIP zero-shot classification step (see pcdet/models/dense_heads/ov_scan_head.py) against
whatever text vocabulary is listed in the model config's ALIGNMENT.NUSCENES_TO_OV_CLASSES. That
vocabulary has been swapped (cfgs/nuscenes_models/ov_scan_lidar_urbaning*.yaml) from the original
generic detection-style strings to UrbanIng-V2X's own native category names -- no retraining
needed, since CLIP's text encoder runs on whatever strings we give it at inference time. Each
prediction's `ov_str_labels` is therefore already a zero-shot guess at the dataset's own label
taxonomy, which this script scores directly against ground truth.

Ground truth's native category per box comes from a converter-written sidecar
(<dataset_root>/<version>/native_categories.json, keyed by sample token). Order/subset alignment
with data_dict['gt_boxes'] is guaranteed because:
  - nuscenes-devkit's reverse index builds each sample's `anns` list by scanning the
    sample_annotation table in file order and appending matches (nuscenes/nuscenes.py
    __make_reverse_index__), i.e. insertion order.
  - the converter appends to sample_annotation_table (and, in lockstep, to the native_categories
    sidecar) in a single fixed per-frame object loop -- same order for both.
  - pcdet's info-pkl builder (nuscenes_utils.fill_trainval_infos) keeps only annotations with
    num_lidar_pts + num_radar_pts > 0, preserving order; the converter mirrors that exact
    condition when appending to the sidecar.
  - at eval time (training=False), pcdet applies no further GT reordering/removal (verified: the
    outside-range box removal in mask_points_and_boxes_outside_range only fires when
    self.training=True, and FILTER_MIN_POINTS_IN_GT re-applies the same already-satisfied
    condition).
Empirically verified against both converted datasets: for all 200 samples in each, len(sidecar
list) == len(info['gt_boxes']), with matching per-box coarse category.

Per-class matching: for each class of interest (an UrbanIng-V2X native category, or one of the
paper's 4 label groups -- vehicle={Car,Van}, two_wheelers={Cyclist,Motorcycle,EScooter},
heavy_vehicle={Truck,Bus,Trailer,OtherVehicle}, pedestrian={Pedestrian,OtherPedestrians}), GT and
predictions of only that class are matched by greedy 3D IoU per frame (standard per-class
detection eval, as in nuScenes/COCO): a right-place-wrong-label detection is simultaneously a
miss (FN) for the true class and a false alarm (FP) for the predicted class. AP is the
all-points-interpolated area under the resulting precision-recall curve (Pascal VOC 2012 style),
using every raw proposal the model emits (no fixed score cutoff) so the full PR curve is covered.

Usage:
    xvfb-run -a python eval_urbaning_v2x_per_class.py \
        --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning.yaml \
        --ckpt ../../pretrained/ov_scan_lidar.pth
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils

# Raw UrbanIng-V2X object_type string (as it appears in the labels json / native_categories.json
# sidecar) -> canonical fine-class key.
GT_NATIVE_TO_CANONICAL = {
    'Car': 'car', 'Van': 'van', 'Bus': 'bus', 'Truck': 'truck', 'OtherVehicle': 'other_vehicle',
    'Trailer': 'trailer', 'EScooter': 'escooter', 'Motorcycle': 'motorcycle', 'Cyclist': 'cyclist',
    'Pedestrian': 'pedestrian', 'OtherPedestrians': 'other_pedestrian',
}
# Predicted ov_str_labels string (as defined in ALIGNMENT.NUSCENES_TO_OV_CLASSES in the model
# configs) -> the same canonical fine-class keys.
PRED_NATIVE_TO_CANONICAL = {
    'car': 'car', 'van': 'van', 'truck': 'truck', 'bus': 'bus', 'trailer': 'trailer',
    'other vehicle': 'other_vehicle', 'e-scooter': 'escooter', 'motorcycle': 'motorcycle',
    'cyclist': 'cyclist', 'pedestrian': 'pedestrian', 'other pedestrian': 'other_pedestrian',
}
# UrbanIng-V2X's own 4 label groups (as specified by the dataset authors).
CANONICAL_TO_GROUP = {
    'car': 'vehicle', 'van': 'vehicle',
    'cyclist': 'two_wheelers', 'motorcycle': 'two_wheelers', 'escooter': 'two_wheelers',
    'truck': 'heavy_vehicle', 'bus': 'heavy_vehicle', 'trailer': 'heavy_vehicle', 'other_vehicle': 'heavy_vehicle',
    'pedestrian': 'pedestrian_group', 'other_pedestrian': 'pedestrian_group',
}
FINE_CLASSES = sorted(set(GT_NATIVE_TO_CANONICAL.values()))
GROUPS = sorted(set(CANONICAL_TO_GROUP.values()))


def parse_config():
    parser = argparse.ArgumentParser(description='Per-class AP/mAP for UrbanIng-V2X inference')
    parser.add_argument('--cfg_file', type=str, required=True, help='dataset/model config')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint to load')
    parser.add_argument('--save_path', type=str, default=None,
                         help='output json path (default: output/<exp>/<tag>/default/per_class_ap.json)')
    parser.add_argument('--match_iou_thresh', type=float, default=0.25,
                         help='3D IoU threshold for greedy per-class pred<->GT matching')
    parser.add_argument('--max_samples', type=int, default=None, help='limit number of samples (default: all)')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    return args, cfg


def filter_valid_gt_boxes_with_categories(gt_boxes_padded, native_categories):
    valid = np.any(gt_boxes_padded[:, :6] != 0, axis=1)
    assert valid.sum() == len(native_categories), \
        f'gt_boxes/native_categories length mismatch: {valid.sum()} vs {len(native_categories)}'
    return gt_boxes_padded[valid], native_categories


def filter_in_range_with_categories(gt_boxes, native_categories, point_cloud_range):
    if len(gt_boxes) == 0:
        return gt_boxes, native_categories
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    in_range = (gt_boxes[:, 0] >= x_min) & (gt_boxes[:, 0] <= x_max) & \
               (gt_boxes[:, 1] >= y_min) & (gt_boxes[:, 1] <= y_max)
    kept_categories = [c for c, keep in zip(native_categories, in_range) if keep]
    return gt_boxes[in_range], kept_categories


def compute_iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    pred_t = torch.from_numpy(pred_boxes[:, :7]).float().cuda()
    gt_t = torch.from_numpy(gt_boxes[:, :7]).float().cuda()
    return iou3d_nms_utils.boxes_iou3d_gpu(pred_t, gt_t).cpu().numpy()


def voc_ap(recall, precision):
    """Pascal VOC 2012 all-points-interpolated average precision."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def class_ap(frames, class_key, iou_thresh, class_of):
    """frames: list of per-frame dicts with pred_boxes/pred_scores/pred_class, gt_boxes/gt_class.
    class_of(label) maps a raw pred/gt canonical label to this eval's class space (identity for
    fine-class eval, canonical->group for group eval)."""
    all_scores = []
    all_tp = []
    num_gt = 0
    for frame in frames:
        gt_mask = np.array([class_of(c) == class_key for c in frame['gt_class']], dtype=bool)
        pred_mask = np.array([class_of(c) == class_key for c in frame['pred_class']], dtype=bool)
        gt_boxes_c = frame['gt_boxes'][gt_mask]
        pred_boxes_c = frame['pred_boxes'][pred_mask]
        pred_scores_c = frame['pred_scores'][pred_mask]
        num_gt += len(gt_boxes_c)

        if len(pred_boxes_c) == 0:
            continue
        order = np.argsort(-pred_scores_c)
        if len(gt_boxes_c) == 0:
            all_scores.extend(pred_scores_c[order].tolist())
            all_tp.extend([0] * len(order))
            continue

        iou = compute_iou_matrix(pred_boxes_c, gt_boxes_c)
        gt_used = np.zeros(len(gt_boxes_c), dtype=bool)
        for i in order:
            candidate = np.where(~gt_used)[0]
            is_tp = 0
            if len(candidate) > 0:
                best = candidate[np.argmax(iou[i, candidate])]
                if iou[i, best] >= iou_thresh:
                    gt_used[best] = True
                    is_tp = 1
            all_scores.append(float(pred_scores_c[i]))
            all_tp.append(is_tp)

    if num_gt == 0:
        # No instances of this class exist in the (filtered) GT at all -- AP is undefined, not 0.
        return {'ap': float('nan'), 'num_gt': num_gt, 'num_pred': len(all_scores)}
    if len(all_scores) == 0:
        # GT exists but the model never predicts this class -- recall is 0 at every threshold,
        # so AP is well-defined and equal to 0 (not undefined).
        return {'ap': 0.0, 'num_gt': num_gt, 'num_pred': 0, 'final_precision': 0.0, 'final_recall': 0.0}

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = 1 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / num_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    ap = voc_ap(recall, precision)
    return {
        'ap': ap, 'num_gt': int(num_gt), 'num_pred': len(all_scores),
        'final_precision': float(precision[-1]), 'final_recall': float(recall[-1]),
    }


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('----------------- UrbanIng-V2X per-class AP/mAP -----------------')

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=2, logger=logger, training=False
    )
    logger.info(f'Total samples: {len(test_set)}')

    dataset_root = Path(cfg.ROOT_DIR) / 'tools' / cfg.DATA_CONFIG.DATA_PATH
    native_categories_path = (dataset_root / cfg.DATA_CONFIG.VERSION / 'native_categories.json').resolve()
    with open(native_categories_path) as f:
        native_categories_by_sample = json.load(f)
    logger.info(f'Loaded native categories sidecar: {native_categories_path}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    num_samples = len(test_loader) if args.max_samples is None else min(args.max_samples, len(test_loader))

    frames = []
    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            if idx >= num_samples:
                break

            sample_token = data_dict['metadata'][0]['token']
            gt_boxes_padded = data_dict['gt_boxes'][0] if 'gt_boxes' in data_dict else np.zeros((0, 10))

            load_data_to_gpu(data_dict)
            pred_dicts, _ = model(data_dict)
            pred = pred_dicts[0]

            pred_boxes = pred['pred_boxes'].cpu().numpy()
            pred_scores = pred['pred_scores'].cpu().numpy()
            pred_native = [PRED_NATIVE_TO_CANONICAL.get(s) for s in pred['ov_str_labels']]

            native_categories = native_categories_by_sample.get(sample_token, [])
            gt_boxes, gt_native_raw = filter_valid_gt_boxes_with_categories(gt_boxes_padded, native_categories)
            gt_boxes, gt_native_raw = filter_in_range_with_categories(
                gt_boxes, gt_native_raw, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
            gt_native = [GT_NATIVE_TO_CANONICAL.get(c) for c in gt_native_raw]

            frames.append({
                'pred_boxes': pred_boxes, 'pred_scores': pred_scores, 'pred_class': pred_native,
                'gt_boxes': gt_boxes, 'gt_class': gt_native,
            })
            logger.info(f'[{idx + 1}/{num_samples}] {len(pred_boxes)} raw preds, {len(gt_boxes)} gt boxes in range')

    logger.info('Computing per-class AP...')
    fine_results = {}
    for cls in FINE_CLASSES:
        fine_results[cls] = class_ap(frames, cls, args.match_iou_thresh, class_of=lambda c: c)

    group_results = {}
    canonical_to_group = lambda c: CANONICAL_TO_GROUP.get(c)
    for grp in GROUPS:
        group_results[grp] = class_ap(frames, grp, args.match_iou_thresh, class_of=canonical_to_group)

    valid_fine_aps = [r['ap'] for r in fine_results.values() if r['num_gt'] > 0]
    valid_group_aps = [r['ap'] for r in group_results.values() if r['num_gt'] > 0]
    mAP_fine = float(np.mean(valid_fine_aps)) if valid_fine_aps else float('nan')
    mAP_groups = float(np.mean(valid_group_aps)) if valid_group_aps else float('nan')

    summary = {
        'cfg_file': args.cfg_file,
        'match_iou_thresh': args.match_iou_thresh,
        'num_frames': len(frames),
        'fine_classes': fine_results,
        'groups': group_results,
        'mAP_fine_classes': mAP_fine,
        'mAP_groups': mAP_groups,
    }

    save_path = Path(args.save_path) if args.save_path is not None else \
        cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'default' / 'per_class_ap.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

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


if __name__ == '__main__':
    main()
