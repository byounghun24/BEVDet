#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare valid BEVDet TensorRT input binaries for trtexec.')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='optional checkpoint (needed for some model variants)')
    parser.add_argument(
        '--out-dir',
        default='tensorrt/trtexec_inputs',
        help='output directory for .bin files')
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='dataset sample index to export')
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=2,
        help='workers per gpu for dataloader')
    return parser.parse_args()


def _save_bin(path, arr):
    arr = np.ascontiguousarray(arr)
    arr.tofile(path)


def _shape_str(shape):
    return 'x'.join(str(int(v)) for v in shape)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg.model.train_cfg = None

    if not hasattr(cfg.data, 'test'):
        raise RuntimeError('Config has no test dataset')
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()

    data_iter = iter(data_loader)
    for _ in range(args.index + 1):
        data = next(data_iter)

    with torch.no_grad():
        inputs = [t.cuda() for t in data['img_inputs'][0]]
        metas = model.get_bev_pool_input(inputs)

        img = inputs[0].squeeze(0).contiguous().float().cpu().numpy()
        ranks_depth = metas[1].contiguous().int().cpu().numpy()
        ranks_feat = metas[2].contiguous().int().cpu().numpy()
        ranks_bev = metas[0].contiguous().int().cpu().numpy()
        interval_starts = metas[3].contiguous().int().cpu().numpy()
        interval_lengths = metas[4].contiguous().int().cpu().numpy()

    tensors = {
        'img': img,
        'ranks_depth': ranks_depth,
        'ranks_feat': ranks_feat,
        'ranks_bev': ranks_bev,
        'interval_starts': interval_starts,
        'interval_lengths': interval_lengths
    }

    meta = {'shapes': {}, 'dtypes': {}, 'files': {}}
    for name, arr in tensors.items():
        path = os.path.join(args.out_dir, f'{name}.bin')
        _save_bin(path, arr)
        meta['shapes'][name] = list(arr.shape)
        meta['dtypes'][name] = str(arr.dtype)
        meta['files'][name] = path

    meta_path = os.path.join(args.out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    shape_spec = ','.join(
        f'{k}:{_shape_str(v)}' for k, v in meta['shapes'].items())
    input_spec = ','.join(f'{k}:{meta["files"][k]}' for k in tensors.keys())

    print(f'Saved input bins to: {args.out_dir}')
    print(f'Meta: {meta_path}')
    print('\nUse this with trtexec:')
    print(
        f'tools/trtexec_861.sh --loadEngine=<engine> --shapes={shape_spec} '
        f'--loadInputs={input_spec} --separateProfileRun --dumpProfile')


if __name__ == '__main__':
    main()
