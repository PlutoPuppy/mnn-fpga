#!/usr/bin/env python3
"""Measure per-channel FP32 output ranges for selected pointwise projections."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--allowlist', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--python-path', type=Path)
    parser.add_argument('--samples', type=int, default=128)
    args = parser.parse_args()
    paths = sorted(args.images.glob('*.png'))
    if not 0 < args.samples <= len(paths) or args.output.exists():
        parser.error('Use an available sample count and a new output path')
    paths = [paths[i * len(paths) // args.samples] for i in range(args.samples)]
    names = {s.strip() for s in args.allowlist.read_text().splitlines()
             if s.strip() and not s.lstrip().startswith('#')}
    if not names:
        parser.error('Empty allowlist')
    if args.python_path:
        sys.path.insert(0, str(args.python_path.resolve()))
    import MNN

    model = MNN.Interpreter(str(args.model.resolve()))
    model.setSessionMode(7)
    session = model.createSession({'backend': 'CPU', 'numThread': 2, 'precision': 'high'})
    inp = model.getSessionInput(session)
    model.resizeTensor(inp, (1, 3, 224, 224))
    model.resizeSession(session)
    limits = {}
    seen = set()

    def channel_rows(tensors, name):
        tensor, = tensors
        shape = tensor.getShape()
        if len(shape) != 4:
            raise ValueError(f'Unsupported projection tensor {name}: {shape}')
        host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.zeros(shape, np.float32),
                          MNN.Tensor_DimensionType_Caffe)
        tensor.copyToHostTensor(host)
        data = host.getNumpyData().transpose(1, 0, 2, 3).reshape(shape[1], -1)
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite calibration tensor: ' + name)
        return data

    def collect(tensors, info):
        name = info.getName()
        if name not in names:
            return True
        if name in seen or info.getType() != 'Convolution':
            raise ValueError('Unsupported or duplicate projection output: ' + name)
        data = channel_rows(tensors,name)
        low, high = data.min(axis=1), data.max(axis=1)
        if name in limits:
            low, high = np.minimum(low, limits[name][0]), np.maximum(high, limits[name][1])
        limits[name] = (low, high)
        seen.add(name)
        return True

    for i, path in enumerate(paths, start=1):
        with Image.open(path) as original:
            image = original.convert('RGB')
        if image.size != (224, 224):
            raise ValueError('Expected pre-resized 224x224 calibration images')
        pixels = np.asarray(image, dtype=np.float32) / 255
        data = np.ascontiguousarray(((pixels - np.array([.485, .456, .406], np.float32)) /
                                    np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1)[None])
        inp.copyFrom(MNN.Tensor(data.shape, MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe))
        seen.clear()
        model.runSessionWithCallBackInfo(session, lambda tensors, info: True, collect)
        if seen != names:
            raise ValueError('Missing calibration outputs: ' + ', '.join(sorted(names-seen)))
        if i % 16 == 0:
            print(f'Calibrated {i}/{len(paths)} images', flush=True)
    with args.output.open('x') as output:
        for name, (low, high) in sorted(limits.items()):
            record = f'{name}\t{len(low)}\t' + ' '.join(f'{x:.9g} {y:.9g}' for x,y in zip(low,high))
            output.write(record+'\n')
    identity = {'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                'images': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'operations': sorted(names), 'method': 'per-channel min/max over the independent calibration split'}
    args.output.with_suffix('.json').write_text(json.dumps(identity, indent=2) + '\n')


if __name__ == '__main__':
    main()
