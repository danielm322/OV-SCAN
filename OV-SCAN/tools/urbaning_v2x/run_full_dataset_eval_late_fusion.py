#!/usr/bin/env python3
"""Host-side orchestrator for the full-dataset late-fusion eval sweep -- the late-fusion analogue
of run_full_dataset_eval.py (which does this for early/input-level fusion, see
docs/Urbaning/Urbaning_PHASE_3.md). See docs/Urbaning/Urbaning_PHASE_5.md for the full design.

For every sequence: extract raw data (host-side 7z), convert up to 6 solo-source datasets
(vehicle1, vehicle2, each infra LiDAR channel individually -- late fusion needs independent
per-agent detections, unlike early fusion's per-combo fused point clouds), then run
eval_urbaning_v2x_late_fusion_batch.py ONCE for the sequence (all 5 combos, each unique source's
inference running exactly once and reused across every combo that needs it, IoU-thresholds swept
from the same WBF-fused frames), then DELETE the raw+converted data -- storage-efficient, same
pattern as Phase 3. The 3 designated per-intersection viz-scene sequences (same ones Phase 3 used)
are the exception: their raw+converted data is kept, and visualize_urbaning_v2x_late_fusion.py
additionally runs there for all 5 combos.

Must run on the HOST, not inside the OV-SCAN container: shells out to `7z` against the source
archives and to `docker exec` into the running OV-SCAN container for conversion/eval/visualize.

Resumable: each (sequence, step) pair writes a JSON marker file on success; a marker already
present is skipped on the next invocation, so an interrupted run can just be re-launched.

Usage (from repo root or anywhere):
    python3 OV-SCAN/tools/urbaning_v2x/run_full_dataset_eval_late_fusion.py [--only-sequence SEQ]
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
HOST_BATCH_ROOT = HOST_DATASETS / 'urbaning_v2x_late_fusion_batch'
CONTAINER_RAW_ROOT = '/OV-SCAN/datasets/urbaning_v2x_raw'
CONTAINER_BATCH_ROOT = '/OV-SCAN/datasets/urbaning_v2x_late_fusion_batch'
CONTAINER_TOOLS = '/OV-SCAN/OV-SCAN/tools'
CKPT = '../../pretrained/ov_scan_lidar.pth'
MLFLOW_EXPERIMENT = 'urbaning_v2x_late_fusion_full_dataset'
LOG_PATH = Path(__file__).resolve().parent / 'full_dataset_eval_late_fusion.log'

# Same 3 designated visualization scenes as run_full_dataset_eval.py (Phase 3), one per intersection.
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

COMBO_NAMES = ['i2i_late', 'v2v_late', 'v2i_v1_late', 'v2i_v2_late', 'full_late']


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


def solo_sources_for(intersection):
    """The up-to-6 solo-source converter invocations needed for every late-fusion combo at this
    intersection: vehicle1 alone, vehicle2 alone, and each infra LiDAR channel alone (crossing3
    uses channel suffixes 21/22 instead of 31/32 -- see INFRA_CHANNELS)."""
    sources = [
        dict(name='vehicle1_solo', convert_mode=['--ego', 'vehicle1']),
        dict(name='vehicle2_solo', convert_mode=['--ego', 'vehicle2']),
    ]
    for ch in INFRA_CHANNELS[intersection]:
        suffix = ch.split('_')[1]
        sources.append(dict(name=f'infra_solo_{suffix}', convert_mode=['--lidars', ch]))
    return sources


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


def marker_path(sequence, name, kind):
    return HOST_BATCH_ROOT / sequence / f'.{name}_{kind}_done'


def convert_solo_source(source, container_data_dir, host_data_dir, sequence):
    converted_ok = (host_data_dir / 'v1.0-custom' / 'nuscenes_infos_1sweeps_val.pkl').exists()
    if converted_ok:
        return
    if (host_data_dir / 'v1.0-custom').exists():
        # root-owned (written by docker exec) -- must be removed from inside the container, a
        # host-side rmtree hits PermissionError.
        run(['docker', 'exec', CONTAINER, 'rm', '-rf', container_data_dir])
    docker_exec(['python', 'urbaning_v2x/convert_to_nuscenes.py',
                 '--root_folder', CONTAINER_RAW_ROOT, '--sequence', sequence,
                 *source['convert_mode'], '--out', container_data_dir, '--version', 'v1.0-custom'])
    docker_exec(['python', 'urbaning_v2x/build_infos.py',
                 '--data_path', container_data_dir, '--version', 'v1.0-custom', '--max_sweeps', '1'])


def process_sequence(sequence, only_combo=None):
    intersection = intersection_of(sequence)
    is_viz_scene = VIZ_SCENES.get(intersection) == sequence
    log(f'=== {sequence} ({intersection}, viz_scene={is_viz_scene}) ===')

    extract_sequence(sequence, intersection)

    container_seq_dir = f'{CONTAINER_BATCH_ROOT}/{sequence}'
    host_seq_dir = HOST_BATCH_ROOT / sequence
    host_seq_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    try:
        for source in solo_sources_for(intersection):
            convert_solo_source(source, f'{container_seq_dir}/{source["name"]}',
                                 host_seq_dir / source['name'], sequence)
        log('  all solo sources converted')
    except Exception as e:
        log(f'  FAILED converting solo sources: {e}')
        failures.append((sequence, 'convert', str(e)))
        return failures  # nothing to eval/viz without converted data

    eval_marker_name = only_combo if only_combo else 'all'
    eval_marker = marker_path(sequence, eval_marker_name, 'eval')
    if not eval_marker.exists():
        try:
            cmd = ['python', 'eval_urbaning_v2x_late_fusion_batch.py',
                   '--root_folder', CONTAINER_RAW_ROOT, '--sequence', sequence,
                   '--data_root', container_seq_dir, '--intersection', intersection,
                   '--ckpt', CKPT, '--mlflow_experiment', MLFLOW_EXPERIMENT,
                   '--save_dir', f'{container_seq_dir}/eval_output']
            if only_combo:
                cmd += ['--only_combo', only_combo]
            docker_exec(cmd, use_xvfb=True)
            eval_marker.write_text(json.dumps({'ts': time.time()}))
            log(f'  eval done ({eval_marker_name})')
        except Exception as e:
            log(f'  FAILED eval: {e}')
            failures.append((sequence, 'eval', str(e)))
    else:
        log(f'  eval ({eval_marker_name}) already done, skipping')

    if is_viz_scene:
        combo_names = [only_combo] if only_combo else COMBO_NAMES
        for combo_name in combo_names:
            viz_marker = marker_path(sequence, combo_name, 'viz')
            if viz_marker.exists():
                log(f'  [{combo_name}] viz already done, skipping')
                continue
            try:
                docker_exec(['python', 'visualize_urbaning_v2x_late_fusion.py',
                             '--combo', combo_name, '--root_folder', CONTAINER_RAW_ROOT,
                             '--sequence', sequence, '--ckpt', CKPT, '--data_root', container_seq_dir,
                             '--intersection', intersection, '--show_labels',
                             '--save_dir', f'{container_seq_dir}/visualizations/{combo_name}',
                             '--mlflow_experiment', MLFLOW_EXPERIMENT], use_xvfb=True)
                viz_marker.write_text(json.dumps({'ts': time.time()}))
                log(f'  [{combo_name}] viz done')
            except Exception as e:
                log(f'  [{combo_name}] viz FAILED: {e}')
                failures.append((sequence, f'viz_{combo_name}', str(e)))

    if not is_viz_scene:
        try:
            raw_dir = HOST_RAW_ROOT / 'dataset' / sequence
            if raw_dir.exists():
                shutil.rmtree(raw_dir)  # host-owned (extracted by host-side 7z)
            if host_seq_dir.exists():
                # root-owned (written by docker exec inside the container) -- must be removed from
                # inside the container too, a host-side rmtree hits PermissionError.
                run(['docker', 'exec', CONTAINER, 'rm', '-rf', container_seq_dir])
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
    parser.add_argument('--only-combo', type=str, default=None, choices=COMBO_NAMES,
                         help='restrict to one combo (for quick testing)')
    args = parser.parse_args()

    sequences = [args.only_sequence] if args.only_sequence else list_sequences()
    log(f'Starting full-dataset late-fusion sweep: {len(sequences)} sequence(s)')

    all_failures = []
    for i, seq in enumerate(sequences, 1):
        log(f'--- sequence {i}/{len(sequences)}: {seq} ---')
        all_failures += process_sequence(seq, only_combo=args.only_combo)

    log(f'Sweep complete. {len(all_failures)} failure(s).')
    for seq, step, err in all_failures:
        log(f'  FAILURE {seq}/{step}: {err.splitlines()[0] if err else err}')

    if all_failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
