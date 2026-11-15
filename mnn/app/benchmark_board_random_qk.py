#!/usr/bin/env python3
"""Paired random-RGB TinyViT benchmark; run on HPS after loading the right FPGA image.

Uses fresh processes because session reuse currently changes model outputs.
Runner inference timing and full child-process wall timing are both recorded.
Compatible with the board's Python 3.5; no third-party Python dependencies.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import struct
import subprocess
import time

OP = '/layers.1/blocks.0/attn/MatMul_output_0'
EXPECTED_JOBS = 2048
MEAN = '0.485,0.456,0.406'
STD = '0.229,0.224,0.225'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_state():
    result = {}
    for cpu in range(2):
        base = pathlib.Path('/sys/devices/system/cpu/cpu' + str(cpu) + '/cpufreq')
        for name in ('scaling_governor', 'scaling_cur_freq'):
            try:
                result['cpu' + str(cpu) + '_' + name] = (base / name).read_text().strip()
            except OSError:
                pass
    return result


def scores(path):
    data = path.read_bytes()
    if len(data) != 4000:
        raise RuntimeError('Expected 1000 FP32 output scores: ' + str(path))
    values = struct.unpack('<1000f', data)
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError('Nonfinite scores: ' + str(path))
    return values


def compare(a, b):
    errors = [x-y for x, y in zip(a, b)]
    aa = sum(x*x for x in a)
    bb = sum(x*x for x in b)
    ee = sum(x*x for x in errors)
    topa = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:5]
    topb = sorted(range(len(b)), key=lambda i: b[i], reverse=True)[:5]
    return {'max_abs_error': max(abs(x) for x in errors),
            'rmse': math.sqrt(ee/len(a)),
            'relative_l2': math.sqrt(ee/aa) if aa else None,
            'cosine_similarity': sum(x*y for x, y in zip(a, b))/math.sqrt(aa*bb) if aa*bb else None,
            'cpu_top5': topa, 'fpga_top5': topb,
            'top1_agrees': topa[0] == topb[0], 'top5_ranking_agrees': topa == topb}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board-dir', default='/home/ubuntu/tinyvit')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pairs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026111100)
    parser.add_argument('--bitstream-sha256', required=True)
    parser.add_argument('--runner', default='tinyvit_offload')
    parser.add_argument('--expected-fpga-jobs', type=int, default=EXPECTED_JOBS)
    parser.add_argument('--dma', action='store_true')
    args = parser.parse_args()
    if args.pairs <= 0 or os.geteuid() != 0:
        parser.error('Run as root with a positive pair count; both modes use the same privilege level')
    board = pathlib.Path(args.board_dir).resolve()
    # Python 3.5 Path.resolve() requires the destination to exist.
    out = pathlib.Path(os.path.abspath(args.output_dir))
    out.mkdir(parents=True, exist_ok=False)
    images = out / 'images'
    images.mkdir()
    env = dict(os.environ)
    for key in ('MNN_FPGA_UIO', 'MNN_MATMUL_SOC_UIO', 'MNN_MATMUL_SOC_DEVMEM', 'MNN_MATMUL_SOC_DMA'):
        env.pop(key, None)
    metadata = {'pairs': args.pairs, 'seed_base': args.seed, 'image_shape_hwc': [224,224,3],
                'image_distribution': 'independent uniform uint8 RGB channels, 0..255',
                'mean': MEAN, 'std': STD, 'operation': OP,
                'operand_shapes': [[16,4,49,32],[16,4,32,49]],
                'expected_jobs_per_fpga_inference': args.expected_fpga_jobs,
                'transport': 'DMA' if args.dma else 'PIO',
                'threads': 2, 'fresh_process_per_inference': True,
                'session_warmup': 0, 'unmeasured_fresh_process_warmup_pairs': 2,
                'cpu_reference': 'default CPU runSession path; device variables removed',
                'fpga_mode': 'selected first QK, explicit /dev/mem, require-fpga',
                'image_generation_excluded_from_both_timers': True,
                'inference_timer': 'runner runSession duration, excluding load/preprocessing/output copy',
                'process_timer': 'subprocess.run wall time including model load, image decode/preprocessing, inference, output files and process teardown',
                'fpga_clock_mhz': 90, 'programmed_bitstream_sha256': args.bitstream_sha256,
                'runner_sha256': sha(board/args.runner), 'model_sha256': sha(board/'tiny-vit.mnn'),
                'platform': platform.platform(), 'cpu_state_before': cpu_state(),
                'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'board_wall_clock_may_be_incorrect': True}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))

    def make_image(index):
        seed = args.seed + index
        rng = random.Random(seed)
        pixels = bytes(rng.getrandbits(8) for _ in range(224*224*3))
        image = images / ('random-%02d.ppm' % index)
        image.write_bytes(b'P6\n224 224\n255\n'+pixels)
        return image, seed

    def execute(mode, image, label):
        target = out / label / mode
        target.mkdir(parents=True)
        command = [str(board/args.runner), '--model', str(board/'tiny-vit.mnn'),
                   '--height','224','--width','224','--image',str(image),
                   '--mean',MEAN,'--std',STD,'--threads','2','--runs','1',
                   '--output-dir',str(target)]
        runenv = dict(env)
        if mode == 'fpga':
            runenv['MNN_MATMUL_SOC_DEVMEM'] = '1'
            if args.dma:
                runenv['MNN_MATMUL_SOC_DMA'] = '1'
            command += ['--fpga-op',OP,'--require-fpga']
        state_before = cpu_state()
        with (target/'run.log').open('w') as log:
            start = time.perf_counter()
            # Blocking wait avoids timeout polling adding up to 50 ms of
            # harness delay to the process-wall measurement on Python 3.5.
            completed = subprocess.run(command, cwd=str(board), env=runenv,
                                       stdout=log, stderr=subprocess.STDOUT)
            process_ms = (time.perf_counter()-start)*1000
        text = (target/'run.log').read_text()
        timing = re.search(r'^Inference 1: ([0-9.]+) ms$',text,re.M)
        jobs = re.search(r'^FPGA submissions: (\d+)$',text,re.M)
        if completed.returncode or not timing or not jobs:
            raise RuntimeError('Run failed: '+str(target))
        jobs = int(jobs.group(1))
        if jobs != (args.expected_fpga_jobs if mode == 'fpga' else 0):
            raise RuntimeError('Unexpected submission count: '+str(target)+' '+str(jobs))
        values = scores(target/'output_0.f32')
        return {'inference_ms':float(timing.group(1)), 'process_ms':process_ms,
                'fpga_jobs':jobs, 'exit_code':completed.returncode,
                'output_sha256':sha(target/'output_0.f32'),
                'cpu_state_before':state_before, 'cpu_state_after':cpu_state()}, values

    # Warm caches and CPU scheduling with separate processes, never session reuse.
    warm_image, _ = make_image(0)
    warmups = []
    for index, order in enumerate((('cpu','fpga'),('fpga','cpu'))):
        for mode in order:
            record, _ = execute(mode,warm_image,'warmup-%d'%index)
            warmups.append(dict(record,mode=mode))
    (out/'warmups.json').write_text(json.dumps(warmups,indent=2))
    rows = []
    for index in range(1,args.pairs+1):
        image, seed = make_image(index)
        order = ('cpu','fpga') if index % 2 else ('fpga','cpu')
        results, outputs = {}, {}
        for mode in order:
            results[mode], outputs[mode] = execute(mode,image,'pair-%02d'%index)
        row = {'pair':index, 'seed':seed, 'order':list(order), 'image_sha256':sha(image),
               'cpu':results['cpu'], 'fpga':results['fpga'],
               'scores':compare(outputs['cpu'],outputs['fpga'])}
        for metric in ('inference_ms','process_ms'):
            cpu, fpga = results['cpu'][metric], results['fpga'][metric]
            row[metric] = {'saved_ms':cpu-fpga, 'reduction_pct':100*(cpu-fpga)/cpu,
                           'speedup':cpu/fpga}
        rows.append(row)
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        print('PAIR %02d order=%s CPU=%.2f FPGA=%.2f ms inference reduction=%+.2f%% process CPU=%.2f FPGA=%.2f ms jobs=%d' %
              (index,','.join(order),results['cpu']['inference_ms'],results['fpga']['inference_ms'],
               row['inference_ms']['reduction_pct'],results['cpu']['process_ms'],
               results['fpga']['process_ms'],results['fpga']['fpga_jobs']),flush=True)
    with (out/'pairs.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair','seed','order','cpu_inference_ms','fpga_inference_ms',
                         'inference_saved_ms','inference_reduction_pct','cpu_process_ms',
                         'fpga_process_ms','process_saved_ms','process_reduction_pct','fpga_jobs',
                         'max_score_error','score_cosine','top1_agrees'])
        for row in rows:
            writer.writerow([row['pair'],row['seed'],','.join(row['order']),
                             row['cpu']['inference_ms'],row['fpga']['inference_ms'],
                             row['inference_ms']['saved_ms'],row['inference_ms']['reduction_pct'],
                             row['cpu']['process_ms'],row['fpga']['process_ms'],
                             row['process_ms']['saved_ms'],row['process_ms']['reduction_pct'],
                             row['fpga']['fpga_jobs'],row['scores']['max_abs_error'],
                             row['scores']['cosine_similarity'],row['scores']['top1_agrees']])
    metadata['cpu_state_after'] = cpu_state()
    metadata['utc_end'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('COMPLETE %d pairs; %d measured FPGA submissions' %
          (len(rows),sum(row['fpga']['fpga_jobs'] for row in rows)),flush=True)


if __name__ == '__main__':
    main()
