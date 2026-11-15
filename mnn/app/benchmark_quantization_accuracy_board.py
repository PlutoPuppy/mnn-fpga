#!/usr/bin/env python3
"""Compare FP32 and repaired mixed precision on 20 real images on HPS Linux.

Uses fresh processes, alternating paired order, two threads and no FPGA.
Compatible with the board's Python 3.5 standard library.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board-dir', default='/home/ubuntu/tinyvit')
    parser.add_argument('--images', default='accuracy-real-20')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference', default='tiny-vit.mnn')
    parser.add_argument('--candidate', default='tiny-vit-mixed-w8a8-accuracy.mnn')
    parser.add_argument('--runner', default='tinyvit_fc1_w8a8_v4')
    args = parser.parse_args()
    board = pathlib.Path(args.board_dir).resolve()
    images = sorted((board / args.images).glob('*.png'))
    if len(images) != 20:
        parser.error('Expected exactly 20 PNGs')
    output = board / args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('MNN_FPGA', 'MNN_MATMUL_SOC', 'MNN_INT8_TRACE')):
            env.pop(key)
    metadata = dict(runner_sha256=sha(board / args.runner), threads=2, fpga_enabled=False,
                    fresh_process_per_inference=True, measured_inferences=40, warmup_inferences=2,
                    models={name: dict(file=str(path), sha256=sha(path), bytes=path.stat().st_size)
                            for name, path in [('fp32', board / args.reference), ('corrected', board / args.candidate)]},
                    images=[dict(file=p.name, sha256=sha(p)) for p in images],
                    timing='runSession; excludes load, session setup, image preprocessing and saving outputs')
    freq = pathlib.Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
    metadata['cpu_frequency_khz_before'] = freq.read_text().strip() if freq.exists() else None
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    def run(name, image, folder):
        target = output / folder / name
        target.mkdir(parents=True)
        command = [str(board / args.runner), '--model', metadata['models'][name]['file'],
                   '--image', str(image), '--mean', '0.485,0.456,0.406', '--std', '0.229,0.224,0.225',
                   '--height', '224', '--width', '224', '--threads', '2', '--runs', '1',
                   '--output-dir', str(target)]
        with (target / 'run.log').open('w') as log:
            start = time.perf_counter()
            proc = subprocess.run(command, cwd=str(board), env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = 1000 * (time.perf_counter() - start)
        text = (target / 'run.log').read_text()
        timing = re.search(r'^Inference 1: ([0-9.]+) ms$', text, re.M)
        if proc.returncode or not timing or not re.search(r'^FPGA submissions: 0$', text, re.M):
            raise RuntimeError('Failed CPU inference: ' + str(target))
        payload = (target / 'output_0.f32').read_bytes()
        if len(payload) != 4000:
            raise RuntimeError('Invalid output size')
        scores = struct.unpack('<1000f', payload)
        if not all(math.isfinite(x) for x in scores):
            raise RuntimeError('Non-finite logits')
        result = dict(inference_ms=float(timing.group(1)), process_ms=elapsed,
                      output_sha256=hashlib.sha256(payload).hexdigest(), top1=max(range(1000), key=lambda j: scores[j]))
        return result, scores

    for name in ('fp32', 'corrected'):
        run(name, images[0], 'warmup')
    rows = []
    for i, image in enumerate(images):
        order = ('fp32', 'corrected') if i % 2 == 0 else ('corrected', 'fp32')
        runs, values = {}, {}
        for name in order:
            runs[name], values[name] = run(name, image, image.stem)
        a, b = values['fp32'], values['corrected']
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        rows.append(dict(image=image.name, order=list(order), runs=runs,
                         agrees=runs['fp32']['top1'] == runs['corrected']['top1'],
                         cosine=sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b),
                         relative_l2=math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / norm_a))
        (output / 'results.json').write_text(json.dumps(rows, indent=2))
        print('IMAGE %d/20 top1 %d/%d; inference %.3f/%.3f ms' %
              (i + 1, runs['fp32']['top1'], runs['corrected']['top1'],
               runs['fp32']['inference_ms'], runs['corrected']['inference_ms']), flush=True)
    metadata['cpu_frequency_khz_after'] = freq.read_text().strip() if freq.exists() else None
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('COMPLETE %d/20 top1 agreement' % sum(row['agrees'] for row in rows), flush=True)


if __name__ == '__main__':
    main()
