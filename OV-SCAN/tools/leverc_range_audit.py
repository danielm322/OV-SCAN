"""Phase 12 Lever C scoping audit (no retraining): for every GT box in the held-out val set --
not just the ones already inside the model's frozen point-cloud range, unlike every other eval
tool in this project -- checks whether it was matched by a same-canonical-class prediction within
--match_iou_thresh, then splits every miss into "out-of-range" (its own (x, y) falls outside
POINT_CLOUD_RANGE, so no proposal mechanism could ever reach it without widening the range) vs.
"in-range" (missed by a genuine model failure, not a range limitation).

Quantifies how much of the "van"/hallucination gap closed by Lever A/B is even addressable by a
future full-retrain range-widening phase, without paying for that retrain now (see
Urbaning_PHASE_12.md Design SS Lever C). Matching is per-canonical-class and lenient (any
same-class prediction within IoU, not a strict greedy 1-1 assignment) since this is a structural
miss/hit split, not an AP computation.

Usage (inside the OV-SCAN container):
    xvfb-run -a python leverc_range_audit.py \
        --cfg_file cfgs/nuscenes_models/ov_scan_lidar_urbaning_v2i_vehicle1_finetune_align.yaml \
        --ckpt /path/to/checkpoint_epoch_8.pth
"""
import argparse
from collections import defaultdict

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from eval_urbaning_v2x_per_class import (
    PRED_NATIVE_TO_CANONICAL, GT_NATIVE_TO_CANONICAL, filter_valid_gt_boxes_with_categories,
    compute_iou_matrix,
)
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--match_iou_thresh', type=float, default=0.25)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)
    x_min, y_min, _, x_max, y_max, _ = cfg.DATA_CONFIG.POINT_CLOUD_RANGE

    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=4, logger=logger, training=False)
    dataset_root = cfg.ROOT_DIR / 'tools' / cfg.DATA_CONFIG.DATA_PATH
    with open(dataset_root / cfg.DATA_CONFIG.VERSION / 'native_categories.json') as f:
        native_categories_by_sample = json.load(f)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    num_samples = len(test_loader) if args.max_samples is None else min(args.max_samples, len(test_loader))

    # Per-canonical-class and overall counts of {in_range_matched, in_range_missed, out_of_range}.
    counts = defaultdict(lambda: defaultdict(int))

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
            pred_native = np.array([PRED_NATIVE_TO_CANONICAL.get(s) for s in pred['ov_str_labels']])

            native_categories = native_categories_by_sample.get(sample_token, [])
            gt_names = test_set.infos[idx]['gt_names']
            # Deliberately NOT calling filter_in_range_with_categories -- the whole point of this
            # audit is to keep out-of-range GT boxes in the pool instead of discarding them.
            gt_boxes, gt_native_raw = filter_valid_gt_boxes_with_categories(
                gt_boxes_padded, native_categories, gt_names, cfg.CLASS_NAMES)
            gt_native = [GT_NATIVE_TO_CANONICAL.get(c) for c in gt_native_raw]

            for i, (gt_box, cls) in enumerate(zip(gt_boxes, gt_native)):
                if cls is None:
                    continue
                x, y = gt_box[0], gt_box[1]
                in_range = (x_min <= x <= x_max) and (y_min <= y <= y_max)
                if not in_range:
                    counts[cls]['out_of_range'] += 1
                    counts['__all__']['out_of_range'] += 1
                    continue
                same_class_mask = pred_native == cls
                if same_class_mask.any():
                    iou = compute_iou_matrix(pred_boxes[same_class_mask], gt_box[None, :7])
                    matched = bool((iou >= args.match_iou_thresh).any())
                else:
                    matched = False
                key = 'in_range_matched' if matched else 'in_range_missed'
                counts[cls][key] += 1
                counts['__all__'][key] += 1

            if (idx + 1) % 200 == 0:
                logger.info(f'[{idx + 1}/{num_samples}] running totals: {dict(counts["__all__"])}')

    print(f'\n=== Lever C range audit: {args.cfg_file} / {args.ckpt} ===')
    print(f'POINT_CLOUD_RANGE: x in [{x_min}, {x_max}], y in [{y_min}, {y_max}]')
    for cls in sorted(counts.keys(), key=lambda k: (k != '__all__', k)):
        c = counts[cls]
        total_missed = c['in_range_missed'] + c['out_of_range']
        total_gt = c['in_range_matched'] + total_missed
        if total_gt == 0:
            continue
        oor_share_of_misses = c['out_of_range'] / total_missed if total_missed > 0 else float('nan')
        print(f'{cls:>18s}: total_gt={total_gt:6d}  in_range_matched={c["in_range_matched"]:6d}  '
              f'in_range_missed={c["in_range_missed"]:6d}  out_of_range={c["out_of_range"]:6d}  '
              f'out_of_range_share_of_all_misses={oor_share_of_misses:.3f}')

    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_path, 'w') as f:
            json.dump({k: dict(v) for k, v in counts.items()}, f, indent=2)
        print(f'Wrote {args.save_path}')


if __name__ == '__main__':
    main()
