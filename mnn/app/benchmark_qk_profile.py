#!/usr/bin/env python3
"""Twenty inputs/repetitions, four fresh-process modes; Python 3.5 on the HPS.

CPU/DMA controls measure latency without phase timers. CPU/DMA profiles measure
the same paths with timers. Rotating orders balance position across 20 inputs.
"""
import argparse
import json
import os
import pathlib
import platform
import random
import re
import subprocess
import time
from benchmark_board_random_qk import OP, MEAN, STD, sha, scores, compare, cpu_state

MODES = ('cpu_control', 'dma_control', 'cpu_profile', 'dma_profile')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-dir', default='/home/ubuntu/tinyvit')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--runner', default='tinyvit_offload_profile')
    p.add_argument('--bitstream-sha256', required=True)
    p.add_argument('--inputs', type=int, default=20)
    p.add_argument('--seed', type=int, default=2026111100)
    p.add_argument('--zeros', action='store_true', help='Repeat an all-zero NCHW FP32 model input; not a normalized black image')
    p.add_argument('--order-design', choices=('cyclic','williams'), default='cyclic')
    a = p.parse_args()
    if os.geteuid() != 0 or a.inputs <= 0:
        p.error('Run as root with a positive input count')
    board = pathlib.Path(a.board_dir).resolve()
    out = pathlib.Path(os.path.abspath(a.output_dir))
    out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir()
    env = dict(os.environ)
    for key in ('MNN_FPGA_UIO','MNN_MATMUL_SOC_UIO','MNN_MATMUL_SOC_DEVMEM','MNN_MATMUL_SOC_DMA'):
        env.pop(key, None)
    metadata = dict(inputs=a.inputs, seed_base=None if a.zeros else a.seed, modes=list(MODES),
                    operation=OP, operand_shapes=[[16,4,49,32],[16,4,32,49]],
                    input_mode='zero_fp32' if a.zeros else 'random_rgb',
                    image_distribution='all-zero model input, 1x3x224x224 FP32' if a.zeros else 'independent uniform uint8 RGB 0..255, 224x224',
                    mean=None if a.zeros else MEAN, std=None if a.zeros else STD, threads=2, session_warmup=0,
                    fresh_process_per_inference=True, warmup_inferences=4,
                    order_design=a.order_design,
                    order='Williams design balances positions, pair precedence and directed neighbors' if a.order_design=='williams' else 'four cyclic rotations, balanced positions for 20 inputs',
                    fpga_clock_mhz=90, programmed_bitstream_sha256=a.bitstream_sha256,
                    runner_sha256=sha(board/a.runner), model_sha256=sha(board/'tiny-vit.mnn'),
                    platform=platform.platform(), cpu_state_before=cpu_state(),
                    board_wall_clock_may_be_incorrect=True,
                    compute_timer='host start-to-done wall time includes MMIO polling and scheduling; no fabric cycle counter',
                    actual_experiment_date='2026-11-11')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))

    def image(index):
        if a.zeros:
            path=out/'images'/'zeros.f32'
            if not path.exists(): path.write_bytes(bytes(3*224*224*4))
            return path
        rng = random.Random(a.seed+index)
        path = out/'images'/('random-%02d.ppm'%index)
        path.write_bytes(b'P6\n224 224\n255\n'+bytes(rng.getrandbits(8) for _ in range(224*224*3)))
        return path

    def execute(mode, img, label):
        target = out/label/mode
        target.mkdir(parents=True)
        runenv = dict(env)
        command = [str(board/a.runner),'--model',str(board/'tiny-vit.mnn'),
                   '--height','224','--width','224','--threads','2','--runs','1',
                   '--output-dir',str(target)]
        command += ['--zeros'] if a.zeros else ['--image',str(img),'--mean',MEAN,'--std',STD]
        dma = mode.startswith('dma')
        profiled = mode.endswith('profile')
        if dma:
            runenv.update(MNN_MATMUL_SOC_DEVMEM='1', MNN_MATMUL_SOC_DMA='1')
            command += ['--fpga-op',OP,'--require-fpga']
        if profiled:
            command += ['--profile']
            if not dma:
                command += ['--fpga-op',OP]
        with (target/'run.log').open('w') as log:
            start = time.perf_counter()
            process = subprocess.run(command,cwd=str(board),env=runenv,stdout=log,stderr=subprocess.STDOUT)
            process_ms = (time.perf_counter()-start)*1000
        log = (target/'run.log').read_text()
        inference = re.search(r'^Inference 1: ([0-9.]+) ms$',log,re.M)
        jobs = re.search(r'^FPGA submissions: (\d+)$',log,re.M)
        if process.returncode or not inference or not jobs or int(jobs.group(1)) != (64 if dma else 0):
            raise RuntimeError('Failed inference: '+str(target))
        result = dict(inference_ms=float(inference.group(1)),process_ms=process_ms,
                      fpga_jobs=int(jobs.group(1)),output_sha256=sha(target/'output_0.f32'),exit_code=process.returncode)
        if profiled:
            match = re.search(r'^PROFILE_JSON (.+)$',log,re.M)
            if not match:
                raise RuntimeError('Missing profile: '+str(target))
            result['profile'] = json.loads(match.group(1))
            if result['profile']['selected_qk_calls'] != 1:
                raise RuntimeError('Unexpected QK callback count: '+str(target))
        return result, scores(target/'output_0.f32')

    warm = image(0)
    warmups = {}
    for mode in MODES:
        warmups[mode], _ = execute(mode,warm,'warmup')
    (out/'warmups.json').write_text(json.dumps(warmups,indent=2))
    rows = []
    for index in range(1,a.inputs+1):
        img = image(index)
        offset = (index-1)%4
        permutations=((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1))
        order = tuple(MODES[i] for i in permutations[offset]) if a.order_design=='williams' else MODES[offset:]+MODES[:offset]
        runs, outputs = {}, {}
        for mode in order:
            runs[mode], outputs[mode] = execute(mode,img,'input-%02d'%index)
        for base in ('cpu','dma'):
            if runs[base+'_control']['output_sha256'] != runs[base+'_profile']['output_sha256']:
                raise RuntimeError('Profiling changed output: '+base+' '+str(index))
        row = dict(input=index,seed=None if a.zeros else a.seed+index,image_sha256=sha(img),order=list(order),runs=runs,
                   scores=compare(outputs['cpu_control'],outputs['dma_control']))
        rows.append(row)
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        print('INPUT %02d CPU %.3f DMA %.3f ms; QK CPU %.3f DMA %.3f; fabric+poll %.3f ms'%(
            index,runs['cpu_control']['inference_ms'],runs['dma_control']['inference_ms'],
            runs['cpu_profile']['profile']['selected_qk_ms'],runs['dma_profile']['profile']['selected_qk_ms'],
            runs['dma_profile']['profile']['offload']['ms']['completion_wait']),flush=True)
    metadata['cpu_state_after'] = cpu_state()
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('COMPLETE %d %s, %d measured inferences'%(len(rows),'zero-input repetitions' if a.zeros else 'random inputs',4*len(rows)),flush=True)


if __name__ == '__main__':
    main()
