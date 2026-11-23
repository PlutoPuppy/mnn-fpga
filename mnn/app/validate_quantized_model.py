#!/usr/bin/env python3
"""Compare a quantized candidate with FP32 on pre-resized, disjoint RGB PNGs.

Reports prediction agreement separately from labeled accuracy. Host latency is
diagnostic only; it is not an HPS benchmark. Run this before deploying a model.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--candidate', type=pathlib.Path, required=True)
    parser.add_argument('--images', type=pathlib.Path, required=True)
    parser.add_argument('--manifest', type=pathlib.Path, help='JSON containing images: [{file, label}]')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--python-path', type=pathlib.Path)
    parser.add_argument('--memory-low', action='store_true', help='Use dynamic CPU quantization for the candidate')
    parser.add_argument('--min-agreement', type=float, default=.95)
    parser.add_argument('--min-cosine', type=float, default=.995)
    parser.add_argument('--max-accuracy-drop', type=float, default=.01)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path to preserve earlier validation evidence')
    for value in (args.min_agreement, args.min_cosine, args.max_accuracy_drop):
        if not 0 <= value <= 1:
            parser.error('Acceptance thresholds must be between 0 and 1')
    paths = sorted(args.images.glob('*.png'))
    if not paths:
        parser.error('No validation PNGs found')
    labels = None
    if args.manifest:
        entries = json.loads(args.manifest.read_text())['images']
        labels = {x['file']: int(x['label']) for x in entries}
        if len(labels) != len(entries) or set(labels) != {p.name for p in paths}:
            parser.error('Label manifest must match the validation images exactly')
        for entry in entries:
            if 'sha256' in entry and hashlib.sha256((args.images / entry['file']).read_bytes()).hexdigest() != entry['sha256']:
                parser.error('Validation image hash mismatch')
    if args.python_path:
        sys.path.insert(0, str(args.python_path.resolve()))
    import MNN

    models = []
    for path in (args.reference, args.candidate):
        net = MNN.Interpreter(str(path.resolve()))
        net.setSessionMode(7)  # Session_Resize_Defer for this dynamic graph.
        config = {'backend': 'CPU', 'numThread': 2, 'precision': 'high'}
        if args.memory_low and len(models) == 1:
            config['memory'] = 2
        session = net.createSession(config)
        inp = net.getSessionInput(session)
        net.resizeTensor(inp, (1, 3, 224, 224))
        net.resizeSession(session)
        models.append((net, session, inp))
    rows = []
    for i, path in enumerate(paths):
        image = Image.open(path).convert('RGB')
        if image.size != (224, 224):
            raise ValueError('Use the same pre-resized 224x224 PNGs for calibration and validation')
        pixels = np.asarray(image, np.float32) / 255
        data = np.ascontiguousarray(((pixels - np.array([.485, .456, .406], np.float32)) /
                                   np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1)[None])
        outputs = []
        for net, session, inp in models:
            inp.copyFrom(MNN.Tensor(data.shape, MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe))
            net.runSession(session)
            out = net.getSessionOutput(session)
            host = MNN.Tensor(out.getShape(), MNN.Halide_Type_Float, np.zeros(out.getShape(), np.float32),
                              out.getDimensionType())
            out.copyToHostTensor(host)
            values = host.getNumpyData().reshape(-1).copy().astype(np.float64)
            if values.size != 1000 or not np.isfinite(values).all() or np.linalg.norm(values) == 0:
                raise ValueError('Invalid model logits')
            outputs.append(values)
        ref, candidate = outputs
        row = dict(image=path.name, reference_top1=int(ref.argmax()), candidate_top1=int(candidate.argmax()),
                   agrees=bool(ref.argmax() == candidate.argmax()),
                   cosine=float(np.dot(ref, candidate) / (np.linalg.norm(ref) * np.linalg.norm(candidate))),
                   relative_l2=float(np.linalg.norm(ref - candidate) / np.linalg.norm(ref)))
        if labels is not None:
            row.update(label=labels[path.name], reference_correct=row['reference_top1'] == labels[path.name],
                       candidate_correct=row['candidate_top1'] == labels[path.name])
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f'Validated {i + 1}/{len(paths)}', flush=True)
    agreement = sum(r['agrees'] for r in rows) / len(rows)
    cosine = float(np.mean([r['cosine'] for r in rows]))
    summary = dict(images=len(rows), top1_agreement=agreement, mean_cosine=cosine,
                   mean_relative_l2=float(np.mean([r['relative_l2'] for r in rows])))
    passed = agreement >= args.min_agreement and cosine >= args.min_cosine
    if labels is not None:
        ref_acc = sum(r['reference_correct'] for r in rows) / len(rows)
        candidate_acc = sum(r['candidate_correct'] for r in rows) / len(rows)
        summary.update(reference_accuracy=ref_acc, candidate_accuracy=candidate_acc,
                       accuracy_drop=ref_acc - candidate_acc)
        passed = passed and ref_acc - candidate_acc <= args.max_accuracy_drop + 1e-12
    result = dict(passed=passed, summary=summary, thresholds=dict(min_agreement=args.min_agreement,
                  min_cosine=args.min_cosine, max_accuracy_drop=args.max_accuracy_drop),
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  mnn_version=getattr(MNN, '__version__', 'unknown'), candidate_memory_low=args.memory_low, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=passed, **summary), indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
