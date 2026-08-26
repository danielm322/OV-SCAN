#!/usr/bin/env python3
"""Host-side orchestrator for the full-dataset OV-SCAN x UrbanIng-V2X eval sweep.

Runs all 6 fusion configs (single_vehicle, i2i, v2v x2 directions, v2i x2 vehicles) against every
sequence of the UrbanIng-V2X dataset (34 sequences across 3 intersections: crossing1/2/3), storage-
efficiently: extract -> convert -> build infos -> eval, then DELETE the raw+converted data before
moving to the next sequence, keeping only what MLflow already captured (params/metrics/artifacts).
The one designated "visualization scene" per intersection is the exception: its raw+converted data
is kept on disk, and visualize_urbaning_v2x.py additionally runs for it (all 6 configs).

Must run on the HOST, not inside the OV-SCAN container: it shells out to `7z` against the source
archives (which live outside the container's bind mounts, under CVDatasets/) for extraction, and to
`docker exec` into the running OV-SCAN container for everything that needs pcdet/CUDA/mlflow
(conversion, info-pkl building, eval, visualize).

Resumable: each (sequence, variant) pair writes a JSON marker file on success; a marker already
present is skipped on the next invocation, so an interrupted run can just be re-launched.

Usage (from repo root or anywhere):
    python3 OV-SCAN/tools/urbaning_v2x/run_full_dataset_eval.py [--only-sequence SEQ]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONTAINER = 'OV-SCAN'
ARCHIVE_DIR = Path('/home/daniel_montoya/Projects/CVDatasets/UrbanIng-V2X/dataset')
LABELS_SRC_DIR = Path('/home/daniel_montoya/Projects/CVDatasets/UrbanIng-V2X/labels')

HOST_DATASETS = Path('/home/daniel_montoya/Projects/OV-SCAN/datasets')
HOST_RAW_ROOT = HOST_DATASETS / 'urbaning_v2x_raw'
HOST_BATCH_ROOT = HOST_DATASETS / 'urbaning_v2x_batch'
CONTAINER_RAW_ROOT = '/OV-SCAN/datasets/urbaning_v2x_raw'
CONTAINER_BATCH_ROOT = '/OV-SCAN/datasets/urbaning_v2x_batch'
CONTAINER_TOOLS = '/OV-SCAN/OV-SCAN/tools'
CKPT = '../../pretrained/ov_scan_lidar.pth'
MLFLOW_EXPERIMENT = 'urbaning_v2x_full_dataset'
LOG_PATH = Path(__file__).resolve().parent / 'full_dataset_eval.log'

# One designated visualization scene per intersection (first sequence chronologically).
VIZ_SCENES = {
    'crossing1': '20241126_0008_crossing1_00',
    'crossing2': '20241126_0001_crossing2_00',
    'crossing3': '20241127_0009_crossing3_00',
}

INFRA_CHANNELS = {
    'crossing1': ['crossing1_11_lidar', 'crossing1_12_lidar', 'crossing1_31_lidar', 'crossing1_32_lidar'],
    'crossing2': ['crossing2_11_lidar', 'crossing2_12_lidar', 'crossing2_31_lidar', 'crossing2_32_lidar'],
    'crossing3': ['crossing3_11_lidar', 'crossing3_12_lidar', 'crossing3_21_lidar', 'crossing3_22_lidar'],
}


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def intersection_of(sequence):
    return re.search(r'crossing\d+', sequence).group(0)


def list_sequences():
    seqs = []
    for f in sorted(ARCHIVE_DIR.glob('*.7z.001')):
        seqs.append(f.name[:-len('.7z.001')])
    return seqs


def variants_for(intersection):
    infra = INFRA_CHANNELS[intersection]
    infra_csv = ','.join(infra)
    infra_sources = ','.join(infra)
    return [
        dict(name='single_vehicle', convert_mode=['--ego', 'vehicle1'],
             model_cfg='ov_scan_lidar_urbaning.yaml',
             fusion_type='single_vehicle', sources='vehicle1', reference='vehicle1'),
        dict(name='i2i', convert_mode=['--lidars', infra_csv],
             model_cfg='ov_scan_lidar_urbaning_infra.yaml',
             fusion_type='i2i', sources=infra_sources, reference='none'),
        dict(name='v2v_v1ref', convert_mode=['--fuse_ego', 'vehicle1', '--fuse_vehicle', 'vehicle2'],
             model_cfg='ov_scan_lidar_urbaning_v2v_v1ref.yaml',
             fusion_type='v2v', sources='vehicle1+vehicle2', reference='vehicle1'),
        dict(name='v2v_v2ref', convert_mode=['--fuse_ego', 'vehicle2', '--fuse_vehicle', 'vehicle1'],
             model_cfg='ov_scan_lidar_urbaning_v2v_v2ref.yaml',
             fusion_type='v2v', sources='vehicle2+vehicle1', reference='vehicle2'),
        dict(name='v2i_vehicle1', convert_mode=['--fuse_ego', 'vehicle1', '--fuse_lidars', infra_csv],
             model_cfg='ov_scan_lidar_urbaning_v2i_vehicle1.yaml',
             fusion_type='v2i', sources=f'vehicle1+{infra_sources}', reference='vehicle1'),
        dict(name='v2i_vehicle2', convert_mode=['--fuse_ego', 'vehicle2', '--fuse_lidars', infra_csv],
             model_cfg='ov_scan_lidar_urbaning_v2i_vehicle2.yaml',
             fusion_type='v2i', sources=f'vehicle2+{infra_sources}', reference='vehicle2'),
    ]


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


def extract_sequence(sequence, intersection):
    dest = HOST_RAW_ROOT / 'dataset' / sequence
    if dest.exists() and any(dest.iterdir()):
        log(f'  [extract] {sequence}: already extracted, skipping')
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE_DIR / f'{sequence}.7z.001'
    patterns = [
        'vehicle1_middle_lidar/*', 'vehicle1_state/*',
        'vehicle2_middle_lidar/*', 'vehicle2_state/*',
        'timesync_info.csv', 'calibration.json',
    ] + [f'{c}/*' for c in INFRA_CHANNELS[intersection]]
    run(['7z', 'x', '-y', f'-o{dest}', str(archive)] + patterns)
    log(f'  [extract] {sequence}: done')

    labels_dest = HOST_RAW_ROOT / 'labels' / f'{sequence}.json'
    if not labels_dest.exists():
        labels_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(LABELS_SRC_DIR / f'{sequence}.json', labels_dest)


def marker_path(sequence, variant_name, kind):
    return HOST_BATCH_ROOT / sequence / f'.{variant_name}_{kind}_done'


def process_variant(sequence, intersection, variant, is_viz_scene):
    name = variant['name']
    eval_marker = marker_path(sequence, name, 'eval')
    viz_marker = marker_path(sequence, name, 'viz')
    container_data_dir = f'{CONTAINER_BATCH_ROOT}/{sequence}/{name}'
    host_data_dir = HOST_BATCH_ROOT / sequence / name

    need_eval = not eval_marker.exists()
    need_viz = is_viz_scene and not viz_marker.exists()
    if not need_eval and not need_viz:
        log(f'  [{name}] already done, skipping')
        return

    if need_eval or need_viz:
        host_data_dir.parent.mkdir(parents=True, exist_ok=True)
        converted_ok = (host_data_dir / 'v1.0-custom' / 'nuscenes_infos_1sweeps_val.pkl').exists() and \
            (host_data_dir / 'v1.0-custom' / 'native_categories.json').exists()
        if not converted_ok:
            if (host_data_dir / 'v1.0-custom').exists():
                # root-owned (written by docker exec) -- must be removed from inside the
                # container, a host-side rmtree hits PermissionError.
                run(['docker', 'exec', CONTAINER, 'rm', '-rf', container_data_dir])
            docker_exec(['python', 'urbaning_v2x/convert_to_nuscenes.py',
                         '--root_folder', CONTAINER_RAW_ROOT, '--sequence', sequence,
                         *variant['convert_mode'], '--out', container_data_dir,
                         '--version', 'v1.0-custom'])
            docker_exec(['python', 'urbaning_v2x/build_infos.py',
                         '--data_path', container_data_dir, '--version', 'v1.0-custom',
                         '--max_sweeps', '1'])
        log(f'  [{name}] converted')

    common = ['--cfg_file', f'cfgs/nuscenes_models/{variant["model_cfg"]}', '--ckpt', CKPT,
              '--data_path', container_data_dir,
              '--fusion_type', variant['fusion_type'], '--sources', f'"{variant["sources"]}"',
              '--reference', variant['reference'], '--sequence', sequence,
              '--intersection', intersection, '--mlflow_experiment', MLFLOW_EXPERIMENT]

    if need_eval:
        docker_exec(['python', 'eval_urbaning_v2x_per_class.py', *common,
                     '--save_path', f'{container_data_dir}/per_class_ap.json'])
        eval_marker.write_text(json.dumps({'ts': time.time()}))
        log(f'  [{name}] eval done')

    if need_viz:
        docker_exec(['python', 'visualize_urbaning_v2x.py', *common, '--show_labels',
                     '--save_dir', f'{container_data_dir}/visualizations'],
                    use_xvfb=True)
        viz_marker.write_text(json.dumps({'ts': time.time()}))
        log(f'  [{name}] viz done')


def process_sequence(sequence, only_variant=None):
    intersection = intersection_of(sequence)
    is_viz_scene = VIZ_SCENES.get(intersection) == sequence
    log(f'=== {sequence} ({intersection}, viz_scene={is_viz_scene}) ===')

    extract_sequence(sequence, intersection)

    variants = variants_for(intersection)
    if only_variant:
        variants = [v for v in variants if v['name'] == only_variant]

    failures = []
    for variant in variants:
        try:
            process_variant(sequence, intersection, variant, is_viz_scene)
        except Exception as e:
            log(f'  [{variant["name"]}] FAILED: {e}')
            failures.append((sequence, variant['name'], str(e)))

    if not is_viz_scene:
        try:
            raw_dir = HOST_RAW_ROOT / 'dataset' / sequence
            if raw_dir.exists():
                shutil.rmtree(raw_dir)  # host-owned (extracted by host-side 7z)
            batch_dir_container = f'{CONTAINER_BATCH_ROOT}/{sequence}'
            if (HOST_BATCH_ROOT / sequence).exists():
                # root-owned (written by docker exec inside the container) -- must be removed
                # from inside the container too, a host-side rmtree hits PermissionError.
                run(['docker', 'exec', CONTAINER, 'rm', '-rf', batch_dir_container])
            log(f'  cleaned up raw+converted data for {sequence}')
        except Exception as e:
            log(f'  WARNING: cleanup failed for {sequence}: {e}')
            failures.append((sequence, 'cleanup', str(e)))
    else:
        log(f'  kept raw+converted data for {sequence} (designated viz scene)')

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-sequence', type=str, default=None)
    parser.add_argument('--only-variant', type=str, default=None,
                         help='restrict to one variant name, e.g. single_vehicle (for quick testing)')
    args = parser.parse_args()

    sequences = [args.only_sequence] if args.only_sequence else list_sequences()
    log(f'Starting full-dataset sweep: {len(sequences)} sequence(s)')

    all_failures = []
    for i, seq in enumerate(sequences, 1):
        log(f'--- sequence {i}/{len(sequences)}: {seq} ---')
        all_failures += process_sequence(seq, only_variant=args.only_variant)

    log(f'Sweep complete. {len(all_failures)} failure(s).')
    for seq, variant, err in all_failures:
        log(f'  FAILURE {seq}/{variant}: {err.splitlines()[0] if err else err}')

    if all_failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
