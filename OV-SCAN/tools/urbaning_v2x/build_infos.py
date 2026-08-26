"""Build nuscenes_infos_1sweeps_{train,val}.pkl for a converted UrbanIng-V2X dataset directory
(see convert_to_nuscenes.py), without going through pcdet's nuscenes_dataset.py CLI.

Why not the CLI: nuscenes_dataset.py's own __main__ block hardcodes
data_path=ROOT_DIR/'data'/'nuscenes', ignoring whatever DATA_PATH a dataset yaml specifies -- a
pre-existing pcdet bug (documented in docs/Urbaning/Urbaning_PHASE_1.md). Calling
create_nuscenes_info() directly, with an explicit --data_path, sidesteps it. This also means no
per-sequence dataset yaml is needed at all: eval/visualize scripts take a --data_path CLI override,
and this script takes the same path directly -- one fixed set of 6 model configs covers every
converted sequence.

Usage:
    python build_infos.py --data_path ../../datasets/urbaning_v2x_batch/<sequence>/<variant>
"""
import argparse
from pathlib import Path

from pcdet.datasets.nuscenes.nuscenes_dataset import create_nuscenes_info


def main():
    parser = argparse.ArgumentParser(description='Build nuScenes info pkls for a converted UrbanIng-V2X dataset')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-custom')
    parser.add_argument('--max_sweeps', type=int, default=1)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    create_nuscenes_info(version=args.version, data_path=data_path, save_path=data_path,
                          max_sweeps=args.max_sweeps)


if __name__ == '__main__':
    main()
