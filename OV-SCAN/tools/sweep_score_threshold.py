"""Sweeps a prediction-score threshold over a checkpoint's inference, to find a deployment-time
cutoff that fixes the `van` proposal-selectivity/hallucination problem found in
Urbaning_PHASE_7.md Sec 5.1/5.2 -- without any retraining.

Key fact this exploits: `OVScanHead.decode_bbox()`'s score filter reads
`DENSE_HEAD.POST_PROCESSING.SCORE_THRESH`, which every config in this project sets to 0.0 (not the
*different*, unused-by-this-path `0.1` in the top-level `MODEL.POST_PROCESSING` block) -- so every
frame always returns all NUM_PROPOSALS=200 raw proposals regardless of confidence, and
`eval_urbaning_v2x_per_class.py`'s `class_ap` (every phase's metric so far) computes AP over the
*full* PR curve, which is threshold-invariant by construction: it can never show what choosing an
actual deployment score cutoff would look like. This script instead reports precision/recall AT a
swept set of candidate thresholds -- the practical question Urbaning_PHASE_7.md Sec 5.1 already
flagged (~0.3 would filter ~92% of the van hallucination volume) but never quantified against
recall loss elsewhere.

Usage (inside the OV-SCAN container):
    xvfb-run -a python sweep_score_threshold.py \
        --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune.yaml \
        --ckpt /path/to/checkpoint_epoch_8.pth \
        --score_threshs 0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.7
"""
import argparse
import json

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from eval_urbaning_v2x_per_class import (
    PRED_NATIVE_TO_CANONICAL, GT_NATIVE_TO_CANONICAL, CANONICAL_TO_GROUP,
    FINE_CLASSES, GROUPS, filter_valid_gt_boxes_with_categories,
    filter_in_range_with_categories, compute_iou_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--score_threshs', type=str, default='0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.7')
    parser.add_argument('--match_iou_thresh', type=float, default=0.25)
    parser.add_argument('--max_samples', type=int, default=None)
    return parser.parse_args()


def class_pr_at_threshold(frames, class_key, iou_thresh, score_thresh, class_of):
    """Greedy per-class matching, same scheme as eval_urbaning_v2x_per_class.class_ap, but at a
    single fixed score threshold (one deployment operating point) instead of sweeping the full PR
    curve -- the two are not equivalent: AP marginalizes over every threshold, this answers what
    one specific cutoff actually does to precision/recall."""
    tp = fp = 0
    num_gt = 0
    for frame in frames:
        gt_mask = np.array([class_of(c) == class_key for c in frame['gt_class']], dtype=bool)
        gt_boxes_c = frame['gt_boxes'][gt_mask]
        num_gt += len(gt_boxes_c)

        pred_mask = np.array([
            class_of(c) == class_key and s >= score_thresh
            for c, s in zip(frame['pred_class'], frame['pred_scores'])
        ], dtype=bool)
        pred_boxes_c = frame['pred_boxes'][pred_mask]
        pred_scores_c = frame['pred_scores'][pred_mask]
        if len(pred_boxes_c) == 0:
            continue
        if len(gt_boxes_c) == 0:
            fp += len(pred_boxes_c)
            continue

        order = np.argsort(-pred_scores_c)
        iou = compute_iou_matrix(pred_boxes_c, gt_boxes_c)
        gt_used = np.zeros(len(gt_boxes_c), dtype=bool)
        for i in order:
            candidate = np.where(~gt_used)[0]
            matched = False
            if len(candidate) > 0:
                best = candidate[np.argmax(iou[i, candidate])]
                if iou[i, best] >= iou_thresh:
                    gt_used[best] = True
                    matched = True
            if matched:
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    recall = tp / num_gt if num_gt > 0 else float('nan')
    return {'tp': tp, 'fp': fp, 'num_gt': num_gt, 'num_pred': tp + fp,
            'precision': precision, 'recall': recall}


def main():
    args = parse_args()
    score_threshs = [float(s) for s in args.score_threshs.split(',')]

    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=4, logger=logger, training=False)
    logger.info(f'Total samples: {len(test_set)}')

    dataset_root = cfg.ROOT_DIR / 'tools' / cfg.DATA_CONFIG.DATA_PATH
    with open(dataset_root / cfg.DATA_CONFIG.VERSION / 'native_categories.json') as f:
        native_categories_by_sample = json.load(f)

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
            pred_class = [PRED_NATIVE_TO_CANONICAL.get(s) for s in pred['ov_str_labels']]

            native_categories = native_categories_by_sample.get(sample_token, [])
            gt_names = test_set.infos[idx]['gt_names']
            gt_boxes, gt_native_raw = filter_valid_gt_boxes_with_categories(
                gt_boxes_padded, native_categories, gt_names, cfg.CLASS_NAMES)
            gt_boxes, gt_native_raw = filter_in_range_with_categories(
                gt_boxes, gt_native_raw, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
            gt_class = [GT_NATIVE_TO_CANONICAL.get(c) for c in gt_native_raw]

            frames.append({
                'pred_boxes': pred_boxes, 'pred_scores': pred_scores, 'pred_class': pred_class,
                'gt_boxes': gt_boxes, 'gt_class': gt_class,
            })
            if (idx + 1) % 200 == 0:
                logger.info(f'[{idx + 1}/{num_samples}] inference done')

    logger.info(f'Collected {len(frames)} frames; sweeping {len(score_threshs)} score thresholds '
                f'at IoU>={args.match_iou_thresh}...')

    identity = lambda c: c
    to_group = lambda c: CANONICAL_TO_GROUP.get(c)

    for thresh in score_threshs:
        total_pred = int(sum((np.asarray(f['pred_scores']) >= thresh).sum() for f in frames))
        print(f'\n=== score_thresh >= {thresh}  (total surviving predictions, any class: {total_pred}) ===')
        print('  -- UrbanIng-V2X label groups --')
        for grp in GROUPS:
            res = class_pr_at_threshold(frames, grp, args.match_iou_thresh, thresh, to_group)
            if res['num_gt'] == 0 and res['num_pred'] == 0:
                continue
            print(f"  {grp:>16}: precision={res['precision']:.3f}  recall={res['recall']:.3f}  "
                  f"tp={res['tp']:5d}  fp={res['fp']:7d}  num_gt={res['num_gt']:5d}")
        print('  -- fine-grained native classes --')
        for cls in FINE_CLASSES:
            res = class_pr_at_threshold(frames, cls, args.match_iou_thresh, thresh, identity)
            if res['num_gt'] == 0 and res['num_pred'] == 0:
                continue
            print(f"  {cls:>16}: precision={res['precision']:.3f}  recall={res['recall']:.3f}  "
                  f"tp={res['tp']:5d}  fp={res['fp']:7d}  num_gt={res['num_gt']:5d}")


if __name__ == '__main__':
    main()
