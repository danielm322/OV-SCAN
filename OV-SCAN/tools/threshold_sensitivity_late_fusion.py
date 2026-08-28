"""
Threshold sensitivity for the late-fusion baseline (see eval_urbaning_v2x_late_fusion.py).

Investigates two knobs, reusing each combo's WBF-fused predictions for every threshold combination
(inference + fusion run ONCE per combo, not once per threshold):

  --match_iou_thresholds  the 3D-IoU threshold used to decide a prediction<->GT match. Applied to
      BOTH per-class AP (class_ap -- rank-based over the full PR curve, so this is the only
      threshold that affects it) and the class-agnostic sanity precision/recall (match_greedy).

  --score_thresholds      the minimum fused score for a prediction to "count" at all. This has NO
      effect on AP (which sweeps every raw proposal by construction, no cutoff) -- it only affects
      the class-agnostic sanity precision/recall, matching what visualize_urbaning_v2x*.py's
      --score_thresh controls.

Note on MODEL.POST_PROCESSING.SCORE_THRESH (0.1 in every urbaning_v2x model config): that field is
NOT read anywhere in OVScanLidar.post_processing() (verified by reading
pcdet/models/detectors/ov_scan_lidar.py) -- only RECALL_THRESH_LIST is used, feeding a recall_dict
none of this project's eval/visualize scripts ever consult. It has zero effect on any metric this
codebase reports; the score threshold that actually matters is the one investigated here.

Usage (from tools/, matching every other urbaning_v2x eval/visualize script's cwd assumption):
    xvfb-run -a python threshold_sensitivity_late_fusion.py \
        --root_folder ../../datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --ckpt ../../pretrained/ov_scan_lidar.pth
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pcdet.config import cfg
from pcdet.utils import common_utils
from eval_urbaning_v2x_late_fusion import COMBOS, build_late_fusion_frames
from eval_urbaning_v2x_per_class import CANONICAL_TO_GROUP, FINE_CLASSES, GROUPS, class_ap
from visualize_urbaning_v2x import match_greedy


def sanity_stats_at(frames, score_thresh, iou_thresh):
    """Class-agnostic precision/recall/tp/fp/fn (visualize_urbaning_v2x.py's match_greedy
    convention), for fused predictions with score < score_thresh dropped first."""
    total_tp = total_fp = total_fn = 0
    for frame in frames:
        keep = frame['pred_scores'] >= score_thresh
        pred_boxes = frame['pred_boxes'][keep]
        pred_scores = frame['pred_scores'][keep]
        tp, fp, fn = match_greedy(pred_boxes, pred_scores, frame['gt_boxes'], iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
    return {'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'precision': precision, 'recall': recall}


def map_at(frames, iou_thresh):
    """mAP (fine classes, groups), same methodology as eval_urbaning_v2x_late_fusion.py."""
    fine_results = [class_ap(frames, cls, iou_thresh, class_of=lambda c: c) for cls in FINE_CLASSES]
    group_results = [class_ap(frames, grp, iou_thresh, class_of=lambda c: CANONICAL_TO_GROUP.get(c)) for grp in GROUPS]
    fine_aps = [r['ap'] for r in fine_results if r['num_gt'] > 0]
    group_aps = [r['ap'] for r in group_results if r['num_gt'] > 0]
    return float(np.mean(fine_aps)) if fine_aps else float('nan'), float(np.mean(group_aps)) if group_aps else float('nan')


def parse_args():
    parser = argparse.ArgumentParser(description='Score/IoU threshold sensitivity for the late-fusion baseline')
    parser.add_argument('--combos', type=str, default=','.join(COMBOS.keys()),
                         help='comma-separated combo names (default: all)')
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--wbf_iou_thresh', type=float, default=0.25,
                         help='IoU threshold for the WBF clustering step itself -- held fixed while '
                              'match_iou_thresholds below varies the separate matching/scoring threshold')
    parser.add_argument('--match_iou_thresholds', type=str, default='0.25,0.3,0.5')
    parser.add_argument('--score_thresholds', type=str, default='0.3,0.1,0.05')
    parser.add_argument('--sanity_iou_thresh', type=float, default=0.25,
                         help='IoU threshold held fixed while sweeping score_thresholds for the sanity precision/recall')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = common_utils.create_logger()
    combos = args.combos.split(',')
    iou_threshs = [float(x) for x in args.match_iou_thresholds.split(',')]
    score_threshs = [float(x) for x in args.score_thresholds.split(',')]

    results = {}
    for combo in combos:
        logger.info(f'=== {combo} ===')
        frames, sources = build_late_fusion_frames(
            combo, args.root_folder, args.sequence, args.ckpt, args.wbf_iou_thresh, logger, args.max_samples)

        map_by_iou = {iou: map_at(frames, iou) for iou in iou_threshs}
        sanity_by_score = {score: sanity_stats_at(frames, score, args.sanity_iou_thresh) for score in score_threshs}
        results[combo] = {
            'sources': [s['label'] for s in sources],
            'mAP_by_match_iou_thresh': {str(iou): {'mAP_fine': f, 'mAP_groups': g} for iou, (f, g) in map_by_iou.items()},
            'sanity_precision_recall_by_score_thresh': {str(s): v for s, v in sanity_by_score.items()},
        }

        logger.info(f'-- mAP vs match_iou_thresh (WBF clustering fixed at {args.wbf_iou_thresh}) --')
        for iou, (f, g) in map_by_iou.items():
            logger.info(f'  IoU={iou}: mAP_fine={f:.3f}  mAP_groups={g:.3f}')
        logger.info(f'-- class-agnostic sanity precision/recall vs score_thresh (IoU={args.sanity_iou_thresh}) --')
        for score, v in sanity_by_score.items():
            logger.info(f"  score>={score}: precision={v['precision']:.3f}  recall={v['recall']:.3f}  "
                        f"tp={v['tp']} fp={v['fp']} fn={v['fn']}")

    save_path = Path(args.save_path) if args.save_path is not None else \
        Path(cfg.ROOT_DIR) / 'output' / 'urbaning_v2x_late_fusion' / 'threshold_sensitivity.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({'sequence': args.sequence, 'wbf_iou_thresh': args.wbf_iou_thresh,
                   'sanity_iou_thresh': args.sanity_iou_thresh, 'results': results}, f, indent=2)
    logger.info(f'Wrote {save_path}')


if __name__ == '__main__':
    main()
