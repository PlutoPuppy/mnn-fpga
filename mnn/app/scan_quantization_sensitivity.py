#!/usr/bin/env python3
"""Quantize each convolution separately and rank its effect on final logits.

Use a diagnostic set, then freeze the allowlist before testing a disjoint
holdout. The scan's scores are for precision selection, not reported accuracy.
"""
import argparse
import gc
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=pathlib.Path, required=True)
    p.add_argument('--calibrated', type=pathlib.Path, required=True)
    p.add_argument('--builder', type=pathlib.Path, required=True)
    p.add_argument('--images', type=pathlib.Path, required=True)
    p.add_argument('--output-dir', type=pathlib.Path, required=True)
    p.add_argument('--python-path', type=pathlib.Path)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--threshold', type=float, default=.9998)
    p.add_argument('--max-ops', type=int, default=0, help='Limit the scan for a tool smoke check')
    a = p.parse_args()
    paths = sorted(a.images.glob('*.png'))
    if a.output_dir.exists() or not 0 < a.samples <= len(paths) or not 0 <= a.threshold <= 1 or a.max_ops < 0:
        p.error('Use a new directory, available positive sample count and valid limits')
    paths = [paths[i * len(paths) // a.samples] for i in range(a.samples)]
    a.output_dir.mkdir(parents=True)
    if a.python_path:
        sys.path.insert(0, str(a.python_path.resolve()))
    import MNN

    inputs = []
    for path in paths:
        image = Image.open(path).convert('RGB')
        if image.size != (224, 224):
            raise ValueError('Expected pre-resized 224x224 PNGs')
        pixels = np.asarray(image, np.float32) / 255
        inputs.append(np.ascontiguousarray(((pixels - np.array([.485, .456, .406], np.float32)) /
                      np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1)[None]))
    names = []

    def run(model, collect_names=False):
        net = MNN.Interpreter(str(model.resolve()))
        net.setSessionMode(7)
        session = net.createSession({'backend': 'CPU', 'numThread': 2, 'precision': 'high'})
        inp = net.getSessionInput(session)
        net.resizeTensor(inp, inputs[0].shape)
        net.resizeSession(session)
        outputs = []

        def end(tensors, info):
            if info.getType().split()[0] in ('Convolution', 'ConvolutionDepthwise') and info.getName() not in names:
                names.append(info.getName())
            return True

        for i, data in enumerate(inputs):
            inp.copyFrom(MNN.Tensor(data.shape, MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe))
            if collect_names and i == 0:
                net.runSessionWithCallBackInfo(session, lambda tensors, info: True, end)
            else:
                net.runSession(session)
            out = net.getSessionOutput(session)
            host = MNN.Tensor(out.getShape(), MNN.Halide_Type_Float, np.zeros(out.getShape(), np.float32),
                              out.getDimensionType())
            out.copyToHostTensor(host)
            values = host.getNumpyData().reshape(-1).copy().astype(np.float64)
            if values.size != 1000 or not np.isfinite(values).all():
                raise ValueError('Invalid logits')
            outputs.append(values)
        return outputs

    reference = run(a.reference, collect_names=True)
    total = len(names)
    if a.max_ops:
        names = names[:a.max_ops]
    rows = []
    for i, name in enumerate(names):
        policy = a.output_dir / 'single-layer.txt'
        policy.write_text(name + '\n')
        model = a.output_dir / ('layer-%03d.mnn' % i)
        subprocess.run([str(a.builder.resolve()), str(a.reference.resolve()), str(a.calibrated.resolve()),
                        str(policy.resolve()), str(model.resolve())], check=True, stdout=subprocess.DEVNULL)
        outputs = run(model)
        cosines = [float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(reference, outputs)]
        row = dict(index=i, name=name, agreement=int(sum(x.argmax() == y.argmax() for x, y in zip(reference, outputs))),
                   mean_cosine=float(np.mean(cosines)), min_cosine=min(cosines))
        rows.append(row)
        print(json.dumps(row), flush=True)
        gc.collect()
        model.unlink()
        result = dict(images=[p.name for p in paths], threshold=a.threshold, total_convolutions=total,
                      complete=len(rows) == total, rows=rows,
                      reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
                      calibrated_sha256=hashlib.sha256(a.calibrated.read_bytes()).hexdigest())
        (a.output_dir / 'sensitivity.json').write_text(json.dumps(result, indent=2) + '\n')
    if len(rows) == total:
        selected = [r['name'] for r in rows if r['mean_cosine'] >= a.threshold]
        (a.output_dir / 'allowlist.txt').write_text('\n'.join(selected) + '\n')
        print('Selected %d/%d convolutions' % (len(selected), total))


if __name__ == '__main__':
    main()
