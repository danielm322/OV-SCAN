#!/usr/bin/env python3
"""Host-side orchestrator for Phase 10's alignment-head fine-tune: builds real `gt_grounding_sam`
CLIP crop-embedding supervision for the v2i_vehicle1 train split (crossing1+crossing2), which
build_v2i_vehicle1_multiseq_dataset.py originally left as an all-None placeholder (UrbanIng-V2X
has no SAM-grounding/camera pipeline wired up -- see Urbaning_PHASE_7.md/_PHASE_10.md).

For each train-split sequence (read directly off the existing train pkl's `lidar_path` fields,
not re-derived from TRAIN_INTERSECTIONS, so this always matches whatever's actually in the pkl):
  1. extracts ONLY vehicle1's 6 camera folders + calibration.json + timesync_info.csv from that
     sequence's .7z archive (NOT the full `extract_sequence()` allow-list in
     build_v2i_vehicle1_multiseq_dataset.py, which deliberately excludes cameras -- this is a
     separate, narrower extraction, left untouched to avoid risking that already-working script);
  2. runs build_alignment_crops_for_sequence.py inside the OV-SCAN container, which projects every
     GT box into those 6 cameras, crops the best-visible one, and encodes+saves a CLIP embedding
     per matched box;
  3. deletes the extracted camera data immediately (same streaming extract->process->delete
     pattern as the main pipeline, bounding peak extra disk use to ~1 sequence, ~700MB, rather
     than the ~19GB the full train split's camera data would take extracted all at once).

After every sequence is processed, runs merge_alignment_crops.py once (also inside the container)
to fold all per-sequence results into a NEW pkl (`..._full_ov_task.pkl`), leaving the original
train pkl untouched.

Resumable: a sequence already in the progress marker file is skipped on rerun (mirrors
build_v2i_vehicle1_multiseq_dataset.py's convention).

Must run on the HOST: shells out to `7z` for extraction and `docker exec` for the GPU-side
projection/crop/CLIP-encoding work (same split as build_v2i_vehicle1_multiseq_dataset.py).

Usage (from repo root or anywhere):
    python3 OV-SCAN/tools/urbaning_v2x/build_alignment_crops.py [--only-sequence SEQ]
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

CONTAINER = 'OV-SCAN'
ARCHIVE_DIR = Path('/home/daniel_montoya/Projects/CVDatasets/UrbanIng-V2X/dataset')

HOST_DATASETS = Path('/home/daniel_montoya/Projects/OV-SCAN/datasets')
HOST_RAW_ROOT = HOST_DATASETS / 'urbaning_v2x_raw'
HOST_MERGED_ROOT = HOST_DATASETS / 'urbaning_v2x_v2i_vehicle1_multiseq'
CONTAINER_RAW_ROOT = '/OV-SCAN/datasets/urbaning_v2x_raw'
CONTAINER_MERGED_ROOT = '/OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1_multiseq'
CONTAINER_TOOLS = '/OV-SCAN/OV-SCAN/tools'

LOG_PATH = Path(__file__).resolve().parent / 'build_alignment_crops.log'
PROGRESS_PATH = Path(__file__).resolve().parent / '.build_alignment_crops_progress.json'

VERSION = 'v1.0-custom'
TRAIN_PKL_NAME = 'nuscenes_infos_1sweeps_train.pkl'
OUT_PKL_NAME = 'nuscenes_infos_1sweeps_train_full_ov_task.pkl'
CLIP_MODEL_TAG = 'DFN5B-CLIP-ViT-H-14'

CAMERAS = ['front_left', 'front_right', 'left', 'right', 'back_left', 'back_right']
LIDAR_PATH_RE = re.compile(r'^samples/LIDAR_TOP/(.+)_fused_vehicle1_\d+\.pcd\.bin$')


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


def docker_exec(args):
    prefix = ['docker', 'exec', CONTAINER, 'bash', '-c']
    inner = ' '.join(args)
    result = run(prefix + [f'cd {CONTAINER_TOOLS} && {inner}'])
    if result.stdout.strip():
        log(result.stdout.strip())
    return result


def list_train_sequences():
    """Unpickling the pkl needs numpy (gt_boxes is a numpy array) -- the bare host python3 doesn't
    have it, so this has to run inside the container, same as every other pkl read in this
    script. One-shot helper script, same ephemeral docker-cp pattern as
    build_v2i_vehicle1_multiseq_dataset.py's build_infos_with_split()."""
    helper_script = Path(__file__).resolve().parent / '_list_sequences_helper.py'
    helper_script.write_text(
        "import json, pickle, re, sys\n"
        "with open(sys.argv[1], 'rb') as f:\n"
        "    infos = pickle.load(f)\n"
        "pattern = re.compile(r'^samples/LIDAR_TOP/(.+)_fused_vehicle1_\\d+\\.pcd\\.bin$')\n"
        "seqs = sorted({m.group(1) for info in infos for m in [pattern.match(info['lidar_path'])] if m})\n"
        "print(json.dumps(seqs))\n"
    )
    run(['docker', 'cp', str(helper_script), f'{CONTAINER}:{CONTAINER_TOOLS}/urbaning_v2x/_list_sequences_helper.py'])
    pkl_container_path = f'{CONTAINER_MERGED_ROOT}/{VERSION}/{TRAIN_PKL_NAME}'
    result = run(['docker', 'exec', CONTAINER, 'python', f'{CONTAINER_TOOLS}/urbaning_v2x/_list_sequences_helper.py',
                  pkl_container_path])
    helper_script.unlink()
    run(['docker', 'exec', CONTAINER, 'rm', '-f', f'{CONTAINER_TOOLS}/urbaning_v2x/_list_sequences_helper.py'])
    return json.loads(result.stdout.strip())


def extract_camera_data(sequence):
    dest = HOST_RAW_ROOT / 'dataset' / sequence
    if dest.exists() and any(dest.iterdir()):
        log(f'  [extract] {sequence}: already extracted, skipping')
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE_DIR / f'{sequence}.7z.001'
    patterns = ['calibration.json', 'timesync_info.csv'] + [f'vehicle1_{c}_camera/*' for c in CAMERAS]
    run(['7z', 'x', '-y', f'-o{dest}', str(archive)] + patterns)
    log(f'  [extract] {sequence}: camera data extracted')


def process_sequence(sequence):
    save_root_container = (f'{CONTAINER_MERGED_ROOT}/ov_alignment/{CLIP_MODEL_TAG}/'
                            f'object_crops/train')
    pkl_container_path = f'{CONTAINER_MERGED_ROOT}/{VERSION}/{TRAIN_PKL_NAME}'
    out_path_container = f'{CONTAINER_RAW_ROOT}/{sequence}_gt_grounding_sam.pkl'
    docker_exec(['python', 'urbaning_v2x/build_alignment_crops_for_sequence.py',
                 '--seq_dir', f'{CONTAINER_RAW_ROOT}/dataset/{sequence}',
                 '--sequence', sequence,
                 '--pkl_path', pkl_container_path,
                 '--save_root', save_root_container,
                 '--out_path', out_path_container])


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {'done_sequences': []}


def save_progress(progress):
    PROGRESS_PATH.write_text(json.dumps(progress))


def merge_all(sequences):
    pkl_container_path = f'{CONTAINER_MERGED_ROOT}/{VERSION}/{TRAIN_PKL_NAME}'
    out_path_container = f'{CONTAINER_MERGED_ROOT}/{VERSION}/{OUT_PKL_NAME}'
    docker_exec(['python', 'urbaning_v2x/merge_alignment_crops.py',
                 '--pkl_path', pkl_container_path,
                 '--results_dir', CONTAINER_RAW_ROOT,
                 '--out_path', out_path_container,
                 '--sequences', ','.join(sequences)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-sequence', type=str, default=None)
    parser.add_argument('--skip-merge', action='store_true',
                         help='process sequences but skip the final merge step (e.g. for a '
                              'single-sequence smoke test)')
    args = parser.parse_args()

    all_sequences = list_train_sequences()
    sequences = [args.only_sequence] if args.only_sequence else all_sequences
    log(f'Starting alignment-crop build: {len(sequences)} train sequence(s)')

    progress = load_progress()
    done = set(progress['done_sequences'])

    for i, sequence in enumerate(sequences, 1):
        if sequence in done:
            log(f'--- sequence {i}/{len(sequences)}: {sequence} (already processed, skipping) ---')
            continue
        log(f'--- sequence {i}/{len(sequences)}: {sequence} ---')

        extract_camera_data(sequence)
        process_sequence(sequence)

        raw_dir = HOST_RAW_ROOT / 'dataset' / sequence
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        log(f'  [{sequence}] cleaned up extracted camera data')

        done.add(sequence)
        progress['done_sequences'] = sorted(done)
        save_progress(progress)

    if args.skip_merge:
        log('--skip-merge set, not merging into a full pkl yet.')
        return

    log(f'All {len(sequences)} sequence(s) processed. Merging into {OUT_PKL_NAME}...')
    merge_all(sorted(done) if not args.only_sequence else sequences)
    log('Done.')


if __name__ == '__main__':
    main()
