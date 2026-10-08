"""Aggregate per-sequence metrics from the 'urbaning_v2x_late_fusion_full_dataset' MLflow
experiment (see run_full_dataset_eval_late_fusion.py / eval_urbaning_v2x_late_fusion_batch.py)
into per-intersection x per-combo mean/std tables -- the late-fusion analogue of
aggregate_full_dataset_results.py.

Each run carries mAP_fine_classes/mAP_groups at the primary IoU (0.25) plus
mAP_fine_classes_iou<T>/mAP_groups_iou<T> for each extra threshold swept (default 0.3, 0.5) --
one table per IoU threshold below.

Usage (inside the OV-SCAN container):
    python3 urbaning_v2x/aggregate_full_dataset_results_late_fusion.py [--format table|markdown]
"""
import argparse
from collections import defaultdict

import mlflow
import numpy as np

MLFLOW_DB = 'sqlite:////OV-SCAN/OV-SCAN/mlflow.db'
EXPERIMENT = 'urbaning_v2x_late_fusion_full_dataset'
# (label, metric key suffix) -- '' is the primary IoU=0.25 metrics (mAP_fine_classes, no suffix).
IOU_VARIANTS = [('0.25', ''), ('0.3', '_iou0.3'), ('0.5', '_iou0.5')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', choices=['table', 'markdown'], default='table')
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_DB)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)

    groups = defaultdict(list)
    for r in runs:
        key = (r.data.params.get('intersection'), r.data.params.get('combo'))
        groups[key].append(r)

    for iou_label, suffix in IOU_VARIANTS:
        rows = []
        for (inter, combo), rs in sorted(groups.items()):
            fine = [r.data.metrics[f'mAP_fine_classes{suffix}'] for r in rs if f'mAP_fine_classes{suffix}' in r.data.metrics]
            grp = [r.data.metrics[f'mAP_groups{suffix}'] for r in rs if f'mAP_groups{suffix}' in r.data.metrics]
            rows.append((inter, combo, len(rs),
                         np.mean(fine) if fine else None, np.std(fine) if fine else None,
                         np.mean(grp) if grp else None, np.std(grp) if grp else None))

        if args.format == 'markdown':
            print(f'\n### IoU = {iou_label}\n')
            print('| Intersection | Combo | N | mAP (fine) | mAP (groups) |')
            print('|---|---|---|---|---|')
            for inter, combo, n, fm, fs, gm, gs in rows:
                fine_s = f'{fm:.3f} ± {fs:.3f}' if fm is not None else 'N/A'
                grp_s = f'{gm:.3f} ± {gs:.3f}' if gm is not None else 'N/A'
                print(f'| {inter} | {combo} | {n} | {fine_s} | {grp_s} |')
        else:
            print(f'\n=== IoU = {iou_label} ===')
            print(f'{"intersection":<12}{"combo":<15}{"n":<4}'
                  f'{"mAP_fine (mean±std)":<24}{"mAP_groups (mean±std)":<24}')
            for inter, combo, n, fm, fs, gm, gs in rows:
                fine_s = f'{fm:.3f} ± {fs:.3f}' if fm is not None else 'N/A'
                grp_s = f'{gm:.3f} ± {gs:.3f}' if gm is not None else 'N/A'
                print(f'{inter:<12}{combo:<15}{n:<4}{fine_s:<24}{grp_s:<24}')


if __name__ == '__main__':
    main()
