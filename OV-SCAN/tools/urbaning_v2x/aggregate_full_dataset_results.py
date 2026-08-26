"""Aggregate per-sequence metrics from the 'urbaning_v2x_full_dataset' MLflow experiment
(see run_full_dataset_eval.py) into per-intersection x per-fusion-config mean/std tables.

Usage (inside the OV-SCAN container):
    python3 urbaning_v2x/aggregate_full_dataset_results.py [--format table|markdown]
"""
import argparse
from collections import defaultdict

import mlflow
import numpy as np

MLFLOW_DB = 'sqlite:////OV-SCAN/OV-SCAN/mlflow.db'
EXPERIMENT = 'urbaning_v2x_full_dataset'


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
        key = (r.data.params.get('intersection'), r.data.params.get('fusion_type'), r.data.params.get('reference'))
        groups[key].append(r)

    rows = []
    for (inter, ft, ref), rs in sorted(groups.items()):
        fine = [r.data.metrics.get('mAP_fine_classes') for r in rs if r.data.metrics.get('mAP_fine_classes') is not None]
        grp = [r.data.metrics.get('mAP_groups') for r in rs if r.data.metrics.get('mAP_groups') is not None]
        rows.append((inter, ft, ref, len(rs),
                     np.mean(fine) if fine else None, np.std(fine) if fine else None,
                     np.mean(grp) if grp else None, np.std(grp) if grp else None))

    if args.format == 'markdown':
        print('| Intersection | Fusion | Reference | N | mAP (fine) | mAP (groups) |')
        print('|---|---|---|---|---|---|')
        for inter, ft, ref, n, fm, fs, gm, gs in rows:
            fine_s = f'{fm:.3f} ± {fs:.3f}' if fm is not None else 'N/A'
            grp_s = f'{gm:.3f} ± {gs:.3f}' if gm is not None else 'N/A'
            print(f'| {inter} | {ft} | {ref} | {n} | {fine_s} | {grp_s} |')
    else:
        print(f'{"intersection":<12}{"fusion_type":<15}{"reference":<10}{"n":<4}'
              f'{"mAP_fine (mean±std)":<24}{"mAP_groups (mean±std)":<24}')
        for inter, ft, ref, n, fm, fs, gm, gs in rows:
            fine_s = f'{fm:.3f} ± {fs:.3f}' if fm is not None else 'N/A'
            grp_s = f'{gm:.3f} ± {gs:.3f}' if gm is not None else 'N/A'
            print(f'{inter:<12}{ft:<15}{ref:<10}{n:<4}{fine_s:<24}{grp_s:<24}')


if __name__ == '__main__':
    main()
