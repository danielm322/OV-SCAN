#!/usr/bin/env python3
"""Ablation study: does convert_to_nuscenes.py's --voxel_size (mean-pool downsampling grid, in
meters, applied only when concatenating multiple raw point clouds for fusion -- i2i/v2v/v2i; see
that script's docstring and voxel_downsample()) affect early-fusion detection accuracy?

Phase 3's full-dataset sweep (run_full_dataset_eval.py) never overrode this -- every fusion config
there used the default 0.1m. A colleague suggested a coarser 0.5m grid might help (less redundant
density from overlapping sensors, closer to nuScenes' native single-LIDAR_TOP point density the
checkpoint was trained on). This script tests that on ONE sequence first.

Scope: the 5 point-cloud-FUSING configs only (i2i, v2v_v1ref, v2v_v2ref, v2i_vehicle1,
v2i_vehicle2) -- single_vehicle never concatenates multiple point clouds, so --voxel_size has no
effect on it (voxel_downsample is only called from InfraFusionConverter/FusedEgoConverter).

Reference sequence: 20241126_0001_crossing2_00 (Phase 3's designated crossing2 viz scene -- raw
data and the voxel_size=0.1 (default) converted datasets are already on disk permanently, so the
0.1 baseline is reused directly rather than reconverted).

N_REPS independent eval reruns per (config, voxel_size): Phase 3 Sec.9 found that a 2-sample A/B
comparison on this codebase's GPU-nondeterminism can look like a real, reproducible effect when it
isn't -- repeats are the guard against that, not a nice-to-have.

Logs to a new MLflow experiment, 'urbaning_v2x_voxel_size_ablation', one run per
(config, voxel_size, rep) -- each rep gets its own run_name (not resumed/enriched across reps),
since independent samples are the point.

Usage (from repo root or anywhere):
    python3 OV-SCAN/tools/urbaning_v2x/voxel_size_ablation.py
"""
import subprocess
import time
from pathlib import Path

CONTAINER = 'OV-SCAN'
SEQUENCE = '20241126_0001_crossing2_00'
INTERSECTION = 'crossing2'
CONTAINER_RAW_ROOT = '/OV-SCAN/datasets/urbaning_v2x_raw'
CONTAINER_TOOLS = '/OV-SCAN/OV-SCAN/tools'
HOST_DATASETS = Path('/home/daniel_montoya/Projects/OV-SCAN/datasets')
CKPT = '../../pretrained/ov_scan_lidar.pth'
MLFLOW_EXPERIMENT = 'urbaning_v2x_voxel_size_ablation'
LOG_PATH = Path(__file__).resolve().parent / 'voxel_size_ablation.log'

VOXEL_SIZES = [0.1, 0.15, 0.3, 0.5]
N_REPS = 2

# voxel_size=0.1 is the default every converted dataset in Phase 3's full sweep already used --
# reuse that existing conversion instead of redoing it.
BASELINE_VOXEL_SIZE = 0.1
BASELINE_CONTAINER_ROOT = f'/OV-SCAN/datasets/urbaning_v2x_batch/{SEQUENCE}'
BASELINE_HOST_ROOT = HOST_DATASETS / 'urbaning_v2x_batch' / SEQUENCE

ABLATION_CONTAINER_ROOT = f'/OV-SCAN/datasets/urbaning_v2x_voxel_ablation/{SEQUENCE}'
ABLATION_HOST_ROOT = HOST_DATASETS / 'urbaning_v2x_voxel_ablation' / SEQUENCE

INFRA_CHANNELS = ['crossing2_11_lidar', 'crossing2_12_lidar', 'crossing2_31_lidar', 'crossing2_32_lidar']
_infra_csv = ','.join(INFRA_CHANNELS)

CONFIGS = [
    dict(name='i2i', convert_mode=['--lidars', _infra_csv],
         model_cfg='ov_scan_lidar_urbaning_infra.yaml',
         fusion_type='i2i', sources=_infra_csv, reference='none'),
    dict(name='v2v_v1ref', convert_mode=['--fuse_ego', 'vehicle1', '--fuse_vehicle', 'vehicle2'],
         model_cfg='ov_scan_lidar_urbaning_v2v_v1ref.yaml',
         fusion_type='v2v', sources='vehicle1+vehicle2', reference='vehicle1'),
    dict(name='v2v_v2ref', convert_mode=['--fuse_ego', 'vehicle2', '--fuse_vehicle', 'vehicle1'],
         model_cfg='ov_scan_lidar_urbaning_v2v_v2ref.yaml',
         fusion_type='v2v', sources='vehicle2+vehicle1', reference='vehicle2'),
    dict(name='v2i_vehicle1', convert_mode=['--fuse_ego', 'vehicle1', '--fuse_lidars', _infra_csv],
         model_cfg='ov_scan_lidar_urbaning_v2i_vehicle1.yaml',
         fusion_type='v2i', sources=f'vehicle1+{_infra_csv}', reference='vehicle1'),
    dict(name='v2i_vehicle2', convert_mode=['--fuse_ego', 'vehicle2', '--fuse_lidars', _infra_csv],
         model_cfg='ov_scan_lidar_urbaning_v2i_vehicle2.yaml',
         fusion_type='v2i', sources=f'vehicle2+{_infra_csv}', reference='vehicle2'),
]


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def run(cmd, **kw):
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        raise RuntimeError(
            f'command failed ({result.returncode}): {" ".join(cmd)}\n'
            f'--- stdout (tail) ---\n{result.stdout[-3000:]}\n'
            f'--- stderr (tail) ---\n{result.stderr[-3000:]}'
        )
    return result


def docker_exec(args, use_xvfb=False):
    prefix = ['docker', 'exec', CONTAINER, 'bash', '-c']
    inner = ' '.join(args)
    if use_xvfb:
        inner = f'xvfb-run -a {inner}'
    return run(prefix + [f'cd {CONTAINER_TOOLS} && {inner}'])


def converted_dir_for(voxel_size, config_name):
    if voxel_size == BASELINE_VOXEL_SIZE:
        return f'{BASELINE_CONTAINER_ROOT}/{config_name}', BASELINE_HOST_ROOT / config_name
    tag = f'{config_name}_voxel{voxel_size}'
    return f'{ABLATION_CONTAINER_ROOT}/{tag}', ABLATION_HOST_ROOT / tag


def ensure_converted(voxel_size, config):
    container_dir, host_dir = converted_dir_for(voxel_size, config['name'])
    pkl = host_dir / 'v1.0-custom' / 'nuscenes_infos_1sweeps_val.pkl'
    if pkl.exists():
        log(f'  [{config["name"]} voxel={voxel_size}] already converted, skipping')
        return container_dir
    host_dir.parent.mkdir(parents=True, exist_ok=True)
    docker_exec(['python', 'urbaning_v2x/convert_to_nuscenes.py',
                 '--root_folder', CONTAINER_RAW_ROOT, '--sequence', SEQUENCE,
                 *config['convert_mode'], '--voxel_size', str(voxel_size),
                 '--out', container_dir, '--version', 'v1.0-custom'])
    docker_exec(['python', 'urbaning_v2x/build_infos.py',
                 '--data_path', container_dir, '--version', 'v1.0-custom', '--max_sweeps', '1'])
    log(f'  [{config["name"]} voxel={voxel_size}] converted')
    return container_dir


def marker_path(voxel_size, config_name, rep):
    container_dir, host_dir = converted_dir_for(voxel_size, config_name)
    name = f'.eval_voxel{voxel_size}_rep{rep}_done'
    return f'{container_dir}/{name}', host_dir / name


def run_eval(container_dir, config, voxel_size, rep):
    container_marker, host_marker = marker_path(voxel_size, config['name'], rep)
    if host_marker.exists():
        log(f'  [{config["name"]} voxel={voxel_size} rep={rep}] already evaluated, skipping')
        return
    run_name = f'{config["name"]}__{SEQUENCE}__voxel{voxel_size}__rep{rep}'
    save_path = f'{container_dir}/per_class_ap_voxel{voxel_size}_rep{rep}.json'
    cmd = ['python', 'eval_urbaning_v2x_per_class.py',
           '--cfg_file', f'cfgs/nuscenes_models/{config["model_cfg"]}', '--ckpt', CKPT,
           '--data_path', container_dir,
           '--fusion_type', config['fusion_type'], '--sources', f'"{config["sources"]}"',
           '--reference', config['reference'], '--sequence', SEQUENCE,
           '--intersection', INTERSECTION, '--mlflow_experiment', MLFLOW_EXPERIMENT,
           '--run_name', run_name, '--save_path', save_path]
    docker_exec(cmd, use_xvfb=True)
    # host_dir is root-owned (written by docker exec, same as every converted dataset dir in this
    # pipeline -- see run_full_dataset_eval.py's comments) -- the marker must be written from
    # inside the container too, a host-side write hits PermissionError.
    docker_exec(['touch', container_marker])
    log(f'  [{config["name"]} voxel={voxel_size} rep={rep}] eval done ({run_name})')


def main():
    log(f'Starting voxel_size ablation: {len(VOXEL_SIZES)} voxel sizes x {len(CONFIGS)} configs x {N_REPS} reps')
    for voxel_size in VOXEL_SIZES:
        for config in CONFIGS:
            container_dir = ensure_converted(voxel_size, config)
            for rep in range(1, N_REPS + 1):
                run_eval(container_dir, config, voxel_size, rep)
    log('Ablation sweep complete.')


if __name__ == '__main__':
    main()
