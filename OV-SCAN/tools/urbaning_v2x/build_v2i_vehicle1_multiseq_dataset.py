#!/usr/bin/env python3
"""Host-side orchestrator for Phase 7's fine-tuning dataset: builds ONE combined, multi-sequence
`v2i_vehicle1` nuScenes-format dataset spanning all 34 UrbanIng-V2X sequences, with a REAL
train/val split by intersection (crossing1 + crossing2 -> train, crossing3 -> held-out val) --
unlike every prior phase, which only ever needed one sequence (or one combo) at a time.

Why a merge is needed (not just pointing build_infos.py at a bigger directory):
convert_to_nuscenes.py's write_tables() always OVERWRITES its whole output directory from one
converter instance's own in-memory tables -- there's no built-in multi-sequence append. But every
token it writes is `uuid.uuid4().hex` (create_token(), confirmed globally unique across
independent runs) and every point-cloud filename is prefixed with the sequence name (confirmed in
FusedEgoConverter: `f'{self.sequence}_fused_{self.reference}_{ts_ms:06d}.pcd.bin'`) -- so merging
is safe as a plain concatenation of each sequence's table JSON lists plus moving each sequence's
point-cloud files into one shared samples/ tree, no remapping or deduplication needed.

Storage-efficient like run_full_dataset_eval.py: only ONE sequence's raw archive + per-sequence
converted output is ever on disk at a time (a fixed, reused temp directory, overwritten and then
deleted each iteration) -- the only thing that grows across the whole run is the final merged
output itself (the thing this script exists to produce), plus small in-memory table lists (a few
thousand dicts of scalars/short lists, not raw tensors).

Must run on the HOST, not inside the OV-SCAN container: shells out to `7z` for extraction and to
`docker exec` for conversion (same split as run_full_dataset_eval.py, for the same reason).

Resumable: a sequence already present in the merge-progress marker file is skipped on rerun. NOT
safely resumable mid-merge-write (the final create_nuscenes_info/gt_grounding_sam-injection step is
idempotent to rerun in full, just not checkpointed internally) -- acceptable for a one-shot dataset
build, unlike the per-sequence eval sweeps this mirrors.

Usage (from repo root or anywhere):
    python3 OV-SCAN/tools/urbaning_v2x/build_v2i_vehicle1_multiseq_dataset.py
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
HOST_PERSEQ_ROOT = HOST_DATASETS / 'urbaning_v2x_v2i_vehicle1_perseq_tmp'
HOST_MERGED_ROOT = HOST_DATASETS / 'urbaning_v2x_v2i_vehicle1_multiseq'
CONTAINER_RAW_ROOT = '/OV-SCAN/datasets/urbaning_v2x_raw'
CONTAINER_PERSEQ_ROOT = '/OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1_perseq_tmp'
CONTAINER_MERGED_ROOT = '/OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1_multiseq'
CONTAINER_TOOLS = '/OV-SCAN/OV-SCAN/tools'
LOG_PATH = Path(__file__).resolve().parent / 'build_v2i_vehicle1_multiseq_dataset.log'
PROGRESS_PATH = Path(__file__).resolve().parent / '.build_v2i_vehicle1_multiseq_progress.json'

VERSION = 'v1.0-custom'
MAX_SWEEPS = 1
TABLE_NAMES = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
               'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

INFRA_CHANNELS = {
    'crossing1': ['crossing1_11_lidar', 'crossing1_12_lidar', 'crossing1_31_lidar', 'crossing1_32_lidar'],
    'crossing2': ['crossing2_11_lidar', 'crossing2_12_lidar', 'crossing2_31_lidar', 'crossing2_32_lidar'],
    'crossing3': ['crossing3_11_lidar', 'crossing3_12_lidar', 'crossing3_21_lidar', 'crossing3_22_lidar'],
}
TRAIN_INTERSECTIONS = {'crossing1', 'crossing2'}
VAL_INTERSECTIONS = {'crossing3'}


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


def convert_sequence(sequence, intersection):
    infra_csv = ','.join(INFRA_CHANNELS[intersection])
    container_dir = f'{CONTAINER_PERSEQ_ROOT}/{sequence}'
    host_dir = HOST_PERSEQ_ROOT / sequence
    if host_dir.exists():
        run(['docker', 'exec', CONTAINER, 'rm', '-rf', container_dir])
    docker_exec(['python', 'urbaning_v2x/convert_to_nuscenes.py',
                 '--root_folder', CONTAINER_RAW_ROOT, '--sequence', sequence,
                 '--fuse_ego', 'vehicle1', '--fuse_lidars', infra_csv,
                 '--out', container_dir, '--version', VERSION])
    log(f'  [{sequence}] converted')


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {'done_sequences': [], 'train_scene_names': []}


def save_progress(progress):
    PROGRESS_PATH.write_text(json.dumps(progress))


def merge_sequence_into(sequence, intersection, merged_version_dir, merged_samples_dir):
    """Appends one sequence's already-converted tables into the merged JSON files on disk
    (read-modify-write per table -- simpler and less error-prone than holding all 34 sequences'
    tables in Python memory across the whole run, and still cheap: each table is a small list of
    dicts, not raw tensors) and moves its point-cloud files into the shared samples tree. Returns
    this sequence's scene name (for the train/val split) and its sample/annotation counts."""
    seq_version_dir = HOST_PERSEQ_ROOT / sequence / VERSION
    for name in TABLE_NAMES:
        src = seq_version_dir / f'{name}.json'
        with open(src) as f:
            rows = json.load(f)
        dst = merged_version_dir / f'{name}.json'
        existing = json.loads(dst.read_text()) if dst.exists() else []
        existing.extend(rows)
        dst.write_text(json.dumps(existing))

    nat_cat_src = seq_version_dir / 'native_categories.json'
    nat_cat_dst = merged_version_dir / 'native_categories.json'
    with open(nat_cat_src) as f:
        seq_nat_cat = json.load(f)
    merged_nat_cat = json.loads(nat_cat_dst.read_text()) if nat_cat_dst.exists() else {}
    merged_nat_cat.update(seq_nat_cat)
    nat_cat_dst.write_text(json.dumps(merged_nat_cat))

    with open(seq_version_dir / 'scene.json') as f:
        scene = json.load(f)[0]
    with open(seq_version_dir / 'sample.json') as f:
        num_samples = len(json.load(f))
    with open(seq_version_dir / 'sample_annotation.json') as f:
        num_anns = len(json.load(f))

    # Point-cloud files were written by docker exec (root-owned on the host) -- a host-side
    # shutil.move's os.rename/os.unlink hits PermissionError, same recurring pattern as every
    # other root-owned converted-dataset path in this project (see run_full_dataset_eval.py's
    # comments). Move them from inside the container instead, where the owning process can.
    container_seq_samples_dir = f'{CONTAINER_PERSEQ_ROOT}/{sequence}/samples/LIDAR_TOP'
    run(['docker', 'exec', CONTAINER, 'bash', '-c',
         f'mv {container_seq_samples_dir}/*.pcd.bin {CONTAINER_MERGED_ROOT}/samples/LIDAR_TOP/'])

    return scene['name'], num_samples, num_anns


def build_infos_with_split(train_scene_names):
    """Runs create_nuscenes_info (with the real train/val split) AND injects gt_grounding_sam=None
    into both resulting pkls, all inside the container -- both steps need numpy/pcdet to even
    unpickle the info records (gt_boxes etc. are numpy arrays), so none of this can run on the
    host. OVScanHead's training-time __getitem__ requires info['gt_grounding_sam'] to exist --
    UrbanIng-V2X has no SAM-grounding data at all, so every box gets None (a path the loss/dataset
    code already tolerates safely: the alignment loss degenerates to its zero-dummy branch when
    every entry in a batch is None, and align_weight is set to 0.0 in the fine-tuning config
    anyway -- see Urbaning_PHASE_7.md)."""
    helper_script = Path(__file__).resolve().parent / '_build_infos_with_split_helper.py'
    helper_script.write_text(
        "import json, pickle, sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from pcdet.datasets.nuscenes.nuscenes_dataset import create_nuscenes_info\n"
        "data_path, version, max_sweeps, train_scenes_path = sys.argv[1:5]\n"
        "with open(train_scenes_path) as f:\n"
        "    train_scene_names = json.load(f)\n"
        "create_nuscenes_info(version=version, data_path=Path(data_path), save_path=Path(data_path),\n"
        "                      max_sweeps=int(max_sweeps), train_scene_names=train_scene_names)\n"
        "version_dir = Path(data_path) / version\n"
        "for split in ['train', 'val']:\n"
        "    pkl_path = version_dir / f'nuscenes_infos_{max_sweeps}sweeps_{split}.pkl'\n"
        "    with open(pkl_path, 'rb') as f:\n"
        "        infos = pickle.load(f)\n"
        "    for info in infos:\n"
        "        info['gt_grounding_sam'] = np.array([None] * len(info['gt_boxes']), dtype=object)\n"
        "    with open(pkl_path, 'wb') as f:\n"
        "        pickle.dump(infos, f)\n"
        "    print(f'{split}: {len(infos)} info records, gt_grounding_sam injected')\n"
    )
    train_scenes_json_path = Path(__file__).resolve().parent / '_train_scene_names.json'
    train_scenes_json_path.write_text(json.dumps(train_scene_names))
    run(['docker', 'cp', str(helper_script), f'{CONTAINER}:{CONTAINER_TOOLS}/urbaning_v2x/_build_infos_with_split_helper.py'])
    run(['docker', 'cp', str(train_scenes_json_path), f'{CONTAINER}:{CONTAINER_TOOLS}/urbaning_v2x/_train_scene_names.json'])
    result = docker_exec(['python', 'urbaning_v2x/_build_infos_with_split_helper.py',
                           CONTAINER_MERGED_ROOT, VERSION, str(MAX_SWEEPS), 'urbaning_v2x/_train_scene_names.json'])
    log(result.stdout.strip())
    helper_script.unlink()
    train_scenes_json_path.unlink()
    run(['docker', 'exec', CONTAINER, 'rm', '-f',
         f'{CONTAINER_TOOLS}/urbaning_v2x/_build_infos_with_split_helper.py',
         f'{CONTAINER_TOOLS}/urbaning_v2x/_train_scene_names.json'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-sequence', type=str, default=None)
    args = parser.parse_args()

    sequences = [args.only_sequence] if args.only_sequence else list_sequences()
    log(f'Starting multi-sequence v2i_vehicle1 dataset build: {len(sequences)} sequence(s)')

    merged_version_dir = HOST_MERGED_ROOT / VERSION
    merged_samples_dir = HOST_MERGED_ROOT / 'samples' / 'LIDAR_TOP'
    merged_version_dir.mkdir(parents=True, exist_ok=True)
    merged_samples_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress()
    done = set(progress['done_sequences'])
    train_scene_names = progress['train_scene_names']

    for i, sequence in enumerate(sequences, 1):
        if sequence in done:
            log(f'--- sequence {i}/{len(sequences)}: {sequence} (already merged, skipping) ---')
            continue
        intersection = intersection_of(sequence)
        log(f'--- sequence {i}/{len(sequences)}: {sequence} ({intersection}) ---')

        extract_sequence(sequence, intersection)
        convert_sequence(sequence, intersection)
        scene_name, num_samples, num_anns = merge_sequence_into(
            sequence, intersection, merged_version_dir, merged_samples_dir)
        log(f'  [{sequence}] merged: {num_samples} samples, {num_anns} annotations, scene={scene_name}')

        if intersection in TRAIN_INTERSECTIONS:
            train_scene_names.append(scene_name)

        raw_dir = HOST_RAW_ROOT / 'dataset' / sequence
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        perseq_dir = HOST_PERSEQ_ROOT / sequence
        if perseq_dir.exists():
            run(['docker', 'exec', CONTAINER, 'rm', '-rf', f'{CONTAINER_PERSEQ_ROOT}/{sequence}'])
        log(f'  [{sequence}] cleaned up raw+per-sequence temp data')

        done.add(sequence)
        progress['done_sequences'] = sorted(done)
        progress['train_scene_names'] = train_scene_names
        save_progress(progress)

    log(f'All {len(sequences)} sequences merged. {len(train_scene_names)} train scenes '
        f'(crossing1+crossing2), {len(sequences) - len(train_scene_names)} held-out val scenes (crossing3).')

    log('Running create_nuscenes_info with the real train/val split, and injecting gt_grounding_sam...')
    build_infos_with_split(train_scene_names)

    # NuScenesOVDataset.__init__ (pcdet/datasets/nuscenes/nuscenes_ov_dataset.py) resolves
    # self.root_path as DATA_PATH/VERSION, and get_lidar_with_sweeps() loads points from
    # self.root_path / info['lidar_path'] -- i.e. it expects samples/LIDAR_TOP nested INSIDE the
    # version directory. But info['lidar_path'] itself is computed (in
    # nuscenes_utils.fill_trainval_infos) relative to the UNVERSIONED data_path, matching where
    # convert_to_nuscenes.py and create_nuscenes_info's own NuScenes(dataroot=...) call actually
    # need the files to be during the BUILD step above. Every single-sequence dataset this project
    # has used until now happened to have the point clouds duplicated in both locations already
    # (a historical artifact, confirmed by inspecting an existing converted dataset directly) --
    # this merged dataset is built cleanly and only has the one (sibling, build-step-correct)
    # copy, so the runtime (training) resolution needs it moved into the versioned location too.
    # A move, not a copy: nothing else needs the sibling location once the pkls are built.
    log('Moving samples/ under the version directory (see comment above for why)...')
    run(['docker', 'exec', CONTAINER, 'mv',
         f'{CONTAINER_MERGED_ROOT}/samples', f'{CONTAINER_MERGED_ROOT}/{VERSION}/samples'])
    log('Done.')


if __name__ == '__main__':
    main()
