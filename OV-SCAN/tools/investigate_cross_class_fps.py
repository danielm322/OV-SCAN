"""Diagnostic tool: for a given predicted class, finds the best-IoU match of every prediction of
that class against ALL GT boxes (any class, not just the one being investigated), and classifies
each as `matched_<same>` (correct), `matched_<other GT class>` (right place, wrong label),
`matched_other` (grouped), or `unmatched` (no GT anywhere nearby -- a genuine hallucination, not
a mislabel).

Built to investigate a specific false-positive explosion found in Phase 7 fine-tuning
(docs/Urbaning/Urbaning_PHASE_7.md Sec.5.1: `van` predictions grew 17x after fine-tuning
`v2i_vehicle1`, with AP dropping) -- the result there disproved the first guess (frozen
CLIP-alignment decision boundary shifting) in favor of a different mechanism (the car-bucket
heatmap becoming far less selective, flooding the fixed proposal budget with near-zero-confidence
noise that still gets CLIP-classified downstream) -- but the method generalizes to any
class/checkpoint comparison, so it's kept here rather than left as a throwaway script.

Usage (inside the OV-SCAN container):
    xvfb-run -a python investigate_cross_class_fps.py \
        --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune.yaml \
        --class_name van \
        --ckpt zero-shot=../../pretrained/ov_scan_lidar.pth \
        --ckpt fine-tuned=/path/to/checkpoint_epoch_8.pth
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
    PRED_NATIVE_TO_CANONICAL, GT_NATIVE_TO_CANONICAL, filter_valid_gt_boxes_with_categories,
    filter_in_range_with_categories, compute_iou_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--class_name', type=str, required=True,
                         help='canonical fine-class name to investigate, e.g. van')
    parser.add_argument('--ckpt', type=str, action='append', required=True,
                         help='label=path, repeatable, e.g. --ckpt zero-shot=... --ckpt fine-tuned=...')
    parser.add_argument('--match_iou_thresh', type=float, default=0.25)
    parser.add_argument('--min_score', type=float, default=0.0,
                         help='only consider --class_name predictions with score >= this (default: '
                              'all of them) -- e.g. to check whether high-confidence predictions are '
                              'still dominated by hallucination/mislabeling, not just the low-score bulk')
    parser.add_argument('--max_samples', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpts = dict(spec.split('=', 1) for spec in args.ckpt)

    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=4, logger=logger, training=False)
    dataset_root = cfg.ROOT_DIR / 'tools' / cfg.DATA_CONFIG.DATA_PATH
    with open(dataset_root / cfg.DATA_CONFIG.VERSION / 'native_categories.json') as f:
        native_categories_by_sample = json.load(f)

    num_samples = len(test_loader) if args.max_samples is None else min(args.max_samples, len(test_loader))

    for label, ckpt_path in ckpts.items():
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        counts = {}
        scores = []
        total = 0

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
                gt_names = test_set.infos[idx]['gt_names']
                gt_boxes, gt_native_raw = filter_valid_gt_boxes_with_categories(
                    gt_boxes_padded, native_categories, gt_names, cfg.CLASS_NAMES)
                gt_boxes, gt_native_raw = filter_in_range_with_categories(
                    gt_boxes, gt_native_raw, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
                gt_native = [GT_NATIVE_TO_CANONICAL.get(c) for c in gt_native_raw]

                mask = np.array([c == args.class_name and s >= args.min_score
                                  for c, s in zip(pred_native, pred_scores)], dtype=bool)
                boxes = pred_boxes[mask]
                total += len(boxes)
                scores.extend(pred_scores[mask].tolist())

                if len(boxes) == 0:
                    continue
                if len(gt_boxes) == 0:
                    counts['unmatched'] = counts.get('unmatched', 0) + len(boxes)
                    continue

                iou = compute_iou_matrix(boxes, gt_boxes)
                best_gt_idx = iou.argmax(axis=1)
                best_iou = iou.max(axis=1)
                for i in range(len(boxes)):
                    if best_iou[i] < args.match_iou_thresh:
                        key = 'unmatched'
                    else:
                        key = f'matched_{gt_native[best_gt_idx[i]]}'
                    counts[key] = counts.get(key, 0) + 1

                if (idx + 1) % 200 == 0:
                    logger.info(f'[{label}] [{idx + 1}/{num_samples}] total_{args.class_name}={total} counts={counts}')

        scores = np.array(scores)
        print(f'\n=== {label}: total {args.class_name} predictions = {total} ===')
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            pct = 100 * v / total if total > 0 else 0
            print(f'  {k}: {v} ({pct:.1f}%)')
        if total > 0:
            print(f'  score distribution: mean={scores.mean():.4f} median={np.median(scores):.4f} '
                  f'p90={np.percentile(scores, 90):.4f} max={scores.max():.4f}')
            print(f'  fraction score>0.1: {(scores > 0.1).mean() * 100:.1f}%  '
                  f'fraction score>0.3: {(scores > 0.3).mean() * 100:.1f}%')


if __name__ == '__main__':
    main()
