"""
Full-dataset-sweep variant of eval_urbaning_v2x_late_fusion.py: evaluates all 5 late-fusion combos
(i2i_late, v2v_late, v2i_v1_late, v2i_v2_late, full_late) for ONE sequence in a single process,
running each of that sequence's up to 6 unique sources' inference exactly once (vehicle1,
vehicle2, 4 infra channels) and reusing the cached per-source predictions across every combo that
uses them -- instead of invoking eval_urbaning_v2x_late_fusion.py once per combo, which would
re-run inference on shared sources up to ~3-4x redundantly (vehicle1/vehicle2 appear in 3 of 5
combos each; each infra channel appears in 4 of 5). See docs/Urbaning/Urbaning_PHASE_5.md.

Also sweeps match_iou_thresh (0.25 primary + extra thresholds, default 0.3/0.5) from the same
WBF-fused frames per combo -- no extra inference or WBF re-fusion -- mirroring
eval_urbaning_v2x_per_class.py's --extra_match_iou_threshs pattern already used in Phase 3's
early-fusion full-dataset sweep.

Logs one MLflow run per combo to --mlflow_experiment (default:
urbaning_v2x_late_fusion_full_dataset), resuming an existing run by name (get_or_create_run) so
re-invocations enrich rather than duplicate.

Usage (inside the OV-SCAN container):
    xvfb-run -a python eval_urbaning_v2x_late_fusion_batch.py \
        --root_folder ../datasets/urbaning_v2x_raw \
        --sequence 20241126_0001_crossing2_00 \
        --data_root ../datasets/urbaning_v2x_late_fusion_batch/20241126_0001_crossing2_00 \
        --ckpt ../../pretrained/ov_scan_lidar.pth
"""
import argparse
import json
import re
from pathlib import Path

import mlflow

from pcdet.config import cfg
from pcdet.utils import common_utils

from mlflow_logging import get_or_create_run
from eval_urbaning_v2x_late_fusion import (
    combo_sources_for, unique_sources, run_all_sources_inference, fuse_combo_frames, compute_ap_sweep,
)

COMBO_NAMES = ['i2i_late', 'v2v_late', 'v2i_v1_late', 'v2i_v2_late', 'full_late']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Full-dataset-sweep late-fusion (WBF) eval, all 5 combos from one process per sequence')
    parser.add_argument('--root_folder', type=str, required=True,
                         help='dir containing dataset/<sequence>/timesync_info.csv and labels/<sequence>.json')
    parser.add_argument('--sequence', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True,
                         help='dir containing <data_root>/vehicle1_solo, vehicle2_solo, '
                              'infra_solo_<channel> -- converted solo datasets for this sequence')
    parser.add_argument('--intersection', type=str, default=None,
                         help='default: derived from --sequence (crossing<N>)')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--wbf_iou_thresh', type=float, default=0.25,
                         help='3D IoU threshold for WBF clustering, held fixed while match_iou_thresh sweeps')
    parser.add_argument('--match_iou_thresh', type=float, default=0.25,
                         help='primary AP-matching IoU threshold (unsuffixed metric names)')
    parser.add_argument('--extra_match_iou_threshs', type=str, default='0.3,0.5',
                         help='comma-separated additional AP-matching IoU thresholds, swept from the '
                              'same WBF-fused frames. Empty string to disable.')
    parser.add_argument('--only_combo', type=str, default=None, choices=COMBO_NAMES,
                         help='restrict to one combo (for quick testing) -- inference still only '
                              'runs for that combo\'s own sources')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None,
                         help='default: output/urbaning_v2x_late_fusion_full_dataset/<sequence>/')
    parser.add_argument('--mlflow_experiment', type=str, default='urbaning_v2x_late_fusion_full_dataset')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = common_utils.create_logger()
    intersection = args.intersection or re.search(r'crossing\d+', args.sequence).group(0)
    logger.info(f'----------------- UrbanIng-V2X late fusion batch: {args.sequence} ({intersection}) -----------------')

    combos = combo_sources_for(intersection, args.data_root)
    if args.only_combo:
        combos = {args.only_combo: combos[args.only_combo]}
    sources = unique_sources(combos)
    logger.info(f'Running inference once for {len(sources)} unique source(s): '
                f'{[s["label"] for s in sources]} (covering {len(combos)} combo(s))')
    frame_predictions, point_cloud_range = run_all_sources_inference(sources, args.ckpt, logger, args.max_samples)

    extra_threshs = [float(x) for x in args.extra_match_iou_threshs.split(',') if x.strip()]
    save_dir = Path(args.save_dir) if args.save_dir is not None else \
        Path(cfg.ROOT_DIR) / 'output' / 'urbaning_v2x_late_fusion_full_dataset' / args.sequence

    for combo_name, combo_sources_list in combos.items():
        logger.info(f'--- combo: {combo_name} ({len(combo_sources_list)} sources) ---')
        frames = fuse_combo_frames(combo_sources_list, frame_predictions, point_cloud_range,
                                    args.root_folder, args.sequence, args.wbf_iou_thresh, logger, args.max_samples)
        summary, mlflow_metrics = compute_ap_sweep(frames, args.match_iou_thresh, extra_threshs, logger)
        summary['combo'] = combo_name
        summary['sources'] = [s['label'] for s in combo_sources_list]
        summary['sequence'] = args.sequence
        summary['wbf_iou_thresh'] = args.wbf_iou_thresh

        combo_save_dir = save_dir / combo_name
        combo_save_dir.mkdir(parents=True, exist_ok=True)
        save_path = combo_save_dir / 'per_class_ap.json'
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

        run_name = f'{combo_name}__{args.sequence}'
        run = get_or_create_run(cfg.ROOT_DIR, args.mlflow_experiment, run_name, tags={
            'fusion_type': combo_name, 'sources': '+'.join(summary['sources']),
            'merge_strategy': 'late_wbf', 'sequence': args.sequence, 'intersection': intersection,
        })
        logger.info(f'MLflow run: {args.mlflow_experiment}/{run_name} ({run.info.run_id})')
        mlflow.log_params({
            'combo': combo_name, 'sources': '+'.join(summary['sources']), 'merge_strategy': 'late_wbf',
            'sequence': args.sequence, 'intersection': intersection, 'ckpt': args.ckpt,
            'wbf_iou_thresh': args.wbf_iou_thresh, 'match_iou_thresh': args.match_iou_thresh,
            'extra_match_iou_threshs': args.extra_match_iou_threshs,
        })
        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(str(save_path))
        logger.info(f"[{combo_name}] mAP_fine={summary['mAP_fine_classes']:.3f} "
                    f"mAP_groups={summary['mAP_groups']:.3f} (wrote {save_path})")
        mlflow.end_run()

    logger.info('Done.')


if __name__ == '__main__':
    main()
