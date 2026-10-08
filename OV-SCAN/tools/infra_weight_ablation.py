#!/usr/bin/env python3
"""Ablation study: does down-weighting infra-source scores before WBF (relative to vehicle
sources, always at 1.0x) improve full_late's mAP? See Urbaning_PHASE_5.md Sec.7's hypothesis --
full_late (all 6 sources) placed second to v2v_late (2 vehicle sources only) at every intersection
and IoU threshold in the full 34-sequence sweep, suggesting the (individually weaker) infra
sources' detections may be dragging WBF's fused output down rather than purely adding coverage.

Runs entirely inside the OV-SCAN container. Reuses eval_urbaning_v2x_late_fusion.py's
run_all_sources_inference ONCE (6 unique sources, cached) and re-fuses via fuse_combo_frames'
new infra_weight parameter for each weight value swept -- no repeated inference, only the cheap
WBF/AP step reruns per weight.

Scope: full_late only, on one sequence first (20241126_0001_crossing2_00 -- same crossing2
reference sequence every other late-fusion result in this project is directly comparable against,
Phase 5's full_late=0.236 fine-mAP@0.25 included). Sweeps infra_weight in
{1.0 (baseline, sanity-checks against Phase 5's number), 0.75, 0.5, 0.25}, each with N_REPS
repeats of the fuse+AP step (inference itself is cached/shared, but WBF's own GPU IoU calls could
in principle jitter run to run -- repeats are the same noise-floor guard as every other ablation
in this project).

Logs to MLflow experiment 'urbaning_v2x_infra_weight_ablation', one run per (infra_weight, rep).

Usage (inside the OV-SCAN container):
    cd /OV-SCAN/OV-SCAN/tools
    xvfb-run -a python urbaning_v2x/infra_weight_ablation.py
"""
import json
from pathlib import Path

import mlflow

from pcdet.config import cfg
from pcdet.utils import common_utils

from mlflow_logging import get_or_create_run
from eval_urbaning_v2x_late_fusion import (
    combo_sources_for, unique_sources, run_all_sources_inference, fuse_combo_frames, compute_ap_sweep,
)

SEQUENCE = '20241126_0001_crossing2_00'
INTERSECTION = 'crossing2'
DATA_ROOT = '/OV-SCAN/datasets/urbaning_v2x_late_fusion_batch/' + SEQUENCE
ROOT_FOLDER = '/OV-SCAN/datasets/urbaning_v2x_raw'
CKPT = '../../pretrained/ov_scan_lidar.pth'
COMBO_NAME = 'full_late'
MLFLOW_EXPERIMENT = 'urbaning_v2x_infra_weight_ablation'
WBF_IOU_THRESH = 0.25
MATCH_IOU_THRESH = 0.25
EXTRA_MATCH_IOU_THRESHS = [0.3, 0.5]

INFRA_WEIGHTS = [1.0, 0.75, 0.5, 0.25]
N_REPS = 2


def main():
    logger = common_utils.create_logger()
    logger.info(f'----------------- infra_weight ablation: {COMBO_NAME} on {SEQUENCE} -----------------')

    combos = combo_sources_for(INTERSECTION, DATA_ROOT)
    combo_sources = combos[COMBO_NAME]
    sources = unique_sources({COMBO_NAME: combo_sources})
    logger.info(f'Running inference once for {len(sources)} unique source(s): {[s["label"] for s in sources]}')
    frame_predictions, point_cloud_range = run_all_sources_inference(sources, CKPT, logger)

    for infra_weight in INFRA_WEIGHTS:
        for rep in range(1, N_REPS + 1):
            logger.info(f'--- infra_weight={infra_weight} rep={rep} ---')
            frames = fuse_combo_frames(
                combo_sources, frame_predictions, point_cloud_range, ROOT_FOLDER, SEQUENCE,
                WBF_IOU_THRESH, logger, infra_weight=infra_weight)
            summary, mlflow_metrics = compute_ap_sweep(frames, MATCH_IOU_THRESH, EXTRA_MATCH_IOU_THRESHS, logger)
            summary['combo'] = COMBO_NAME
            summary['sequence'] = SEQUENCE
            summary['infra_weight'] = infra_weight
            summary['wbf_iou_thresh'] = WBF_IOU_THRESH

            save_dir = Path(cfg.ROOT_DIR) / 'output' / MLFLOW_EXPERIMENT / f'infraweight{infra_weight}' / f'rep{rep}'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / 'per_class_ap.json'
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2)

            run_name = f'{COMBO_NAME}__{SEQUENCE}__infraweight{infra_weight}__rep{rep}'
            run = get_or_create_run(cfg.ROOT_DIR, MLFLOW_EXPERIMENT, run_name, tags={
                'fusion_type': COMBO_NAME, 'sequence': SEQUENCE, 'intersection': INTERSECTION,
                'merge_strategy': 'late_wbf',
            })
            mlflow.log_params({
                'combo': COMBO_NAME, 'sequence': SEQUENCE, 'intersection': INTERSECTION,
                'infra_weight': infra_weight, 'wbf_iou_thresh': WBF_IOU_THRESH,
                'match_iou_thresh': MATCH_IOU_THRESH, 'rep': rep,
            })
            mlflow.log_metrics(mlflow_metrics)
            mlflow.log_artifact(str(save_path))
            logger.info(f"[infra_weight={infra_weight} rep={rep}] mAP_fine={summary['mAP_fine_classes']:.4f} "
                        f"mAP_groups={summary['mAP_groups']:.4f} (run {run.info.run_id})")
            mlflow.end_run()

    logger.info('Done.')


if __name__ == '__main__':
    main()
