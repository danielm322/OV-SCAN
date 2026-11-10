"""Runs INSIDE the OV-SCAN container (needs numpy to unpickle gt_boxes arrays). Merges the
per-sequence `gt_grounding_sam` results produced by build_alignment_crops_for_sequence.py (one
pickle per sequence, {token: [entry_or_None, ...]}) into a COPY of the train info pkl, replacing
its placeholder `gt_grounding_sam = np.array([None]*len(gt_boxes))` with the real per-box CLIP
crop-embedding entries, and writes it out as a new pkl -- the original is left untouched.

Usage (inside the container):
    python urbaning_v2x/merge_alignment_crops.py \
        --pkl_path /OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1_multiseq/v1.0-custom/nuscenes_infos_1sweeps_train.pkl \
        --results_dir /OV-SCAN/datasets/urbaning_v2x_raw \
        --out_path /OV-SCAN/datasets/urbaning_v2x_v2i_vehicle1_multiseq/v1.0-custom/nuscenes_infos_1sweeps_train_full_ov_task.pkl \
        --sequences seq1,seq2,...
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--sequences', required=True, help='comma-separated sequence names')
    parser.add_argument('--result_suffix', default='',
                         help='matches build_alignment_crops_for_sequence.py\'s per-sequence '
                              'output filename when it was run as a non-default variant (e.g. '
                              '"_occfilt"), so this merge reads the right intermediate files')
    args = parser.parse_args()

    with open(args.pkl_path, 'rb') as f:
        infos = pickle.load(f)

    sequences = args.sequences.split(',')
    token_to_entries = {}
    for sequence in sequences:
        result_path = Path(args.results_dir) / f'{sequence}_gt_grounding_sam{args.result_suffix}.pkl'
        with open(result_path, 'rb') as f:
            token_to_entries.update(pickle.load(f))

    num_total, num_matched, num_missing_token = 0, 0, 0
    for info in infos:
        entries = token_to_entries.get(info['token'])
        if entries is None:
            num_missing_token += 1
            continue
        assert len(entries) == len(info['gt_boxes']), (
            f"length mismatch for token {info['token']}: {len(entries)} entries vs "
            f"{len(info['gt_boxes'])} gt_boxes")
        info['gt_grounding_sam'] = np.array(entries, dtype=object)
        num_total += len(entries)
        num_matched += sum(1 for e in entries if e is not None)

    with open(args.out_path, 'wb') as f:
        pickle.dump(infos, f)

    coverage = num_matched / num_total if num_total else 0.0
    print(f'Merged {len(sequences)} sequences into {args.out_path}')
    print(f'{num_matched}/{num_total} GT boxes have a real CLIP crop embedding ({coverage:.1%} coverage)')
    if num_missing_token:
        print(f'WARNING: {num_missing_token} info records had no matching per-sequence result '
              f'(left with their original gt_grounding_sam, likely still all-None)')


if __name__ == '__main__':
    main()
