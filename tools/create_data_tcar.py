# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from tools.data_converter import tcar_converter


def tcar_data_prep(root_path,
                   out_dir,
                   can_bus_root_path,
                   info_prefix,
                   version,
                   max_sweeps=10):
    """Prepare TCAR info files in nuScenes format."""
    tcar_converter.create_nuscenes_infos(
        root_path,
        out_dir,
        can_bus_root_path,
        info_prefix,
        version=version,
        max_sweeps=max_sweeps)


def parse_args():
    parser = argparse.ArgumentParser(description='Create TCAR info files')
    parser.add_argument('--root-path', required=True, help='TCAR dataset root')
    parser.add_argument('--out-dir', required=True, help='Output directory for info files')
    parser.add_argument('--can-bus-root-path', default='', help='CAN bus root path (optional)')
    parser.add_argument('--info-prefix', default='tcar', help='Prefix for info files')
    parser.add_argument('--version', default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--max-sweeps', type=int, default=10, help='Max number of sweeps')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tcar_data_prep(
        root_path=args.root_path,
        out_dir=args.out_dir,
        can_bus_root_path=args.can_bus_root_path,
        info_prefix=args.info_prefix,
        version=args.version,
        max_sweeps=args.max_sweeps)
