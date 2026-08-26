"""Shared MLflow experiment-tracking helper for the UrbanIng-V2X zero-shot inference scripts
(eval_urbaning_v2x_per_class.py, visualize_urbaning_v2x.py).

Backend store: local SQLite database at OV-SCAN/OV-SCAN/mlflow.db -- inside the docker
bind-mounted subtree, so it persists across container recreation and is visible on the host.
MLflow 3.x deprecated the plain filesystem backend (params/metrics/tags as meta.yaml files) in
favor of a DB backend; artifacts (json summaries, sample PNGs) still live as plain files under
OV-SCAN/OV-SCAN/mlruns/<experiment_id>/<run_id>/artifacts/, unaffected by the backend store choice
-- the two are decoupled, only the metadata store moved. One MLflow run per (fusion_type, sources)
experiment: the eval and visualize scripts are separate process invocations, but both log into the
*same* run (found by run_name) so AP metrics, class-agnostic precision/recall, and sample
visualizations for one experiment all land in one comparable row.
Browse with: mlflow ui --backend-store-uri sqlite:////OV-SCAN/OV-SCAN/mlflow.db -h 0.0.0.0
"""
import re
from pathlib import Path

import mlflow


def add_common_args(parser):
    parser.add_argument('--fusion_type', type=str, required=True,
                         choices=['single_vehicle', 'i2i', 'v2v', 'v2i'],
                         help='sensor-fusion configuration this run is, for MLflow comparison')
    parser.add_argument('--sources', type=str, required=True,
                         help='human-readable fused sources, e.g. "vehicle1+vehicle2" or '
                              '"vehicle1+crossing2_11,crossing2_12,crossing2_31,crossing2_32"')
    parser.add_argument('--reference', type=str, default='none',
                         choices=['vehicle1', 'vehicle2', 'none'],
                         help='reference/ego vehicle whose frame the fused cloud is expressed in '
                              '("none" for i2i, which uses a recentered global frame instead)')
    parser.add_argument('--sequence', type=str, default=None,
                         help='UrbanIng-V2X sequence name (e.g. "20241126_0001_crossing2_00"); '
                              'folded into the run name so multiple sequences of the same '
                              'fusion_type/sources stay distinguishable, and logged as a param/tag '
                              'for filtering. Omit for single-sequence, non-batch runs.')
    parser.add_argument('--intersection', type=str, default=None,
                         help='intersection id (e.g. "crossing1"); auto-derived from --sequence '
                              'if not given')
    parser.add_argument('--mlflow_experiment', type=str, default='urbaning_v2x',
                         help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                         help='MLflow run name (default: "<fusion_type>__<sources>[__<sequence>]")')


def intersection_for(args):
    if args.intersection:
        return args.intersection
    if args.sequence:
        m = re.search(r'crossing\d+', args.sequence)
        if m:
            return m.group(0)
    return None


def run_name_for(args):
    if args.run_name:
        return args.run_name
    slug = args.sources.replace('+', '_').replace(',', '-')
    base = f'{args.fusion_type}__{slug}'
    return f'{base}__{args.sequence}' if args.sequence else base


def get_or_create_run(root_dir, experiment_name, run_name, tags):
    """Sets the tracking URI + experiment, and resumes an existing run with this run_name if one
    already exists (so eval + visualize accumulate into the same row), else starts a new one."""
    db_path = Path(root_dir).resolve() / 'mlflow.db'
    mlflow.set_tracking_uri(f'sqlite:///{db_path}')
    experiment = mlflow.set_experiment(experiment_name)
    existing = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format='list',
    )
    run_id = existing[0].info.run_id if len(existing) > 0 else None
    return mlflow.start_run(run_id=run_id, run_name=run_name, tags=tags)
