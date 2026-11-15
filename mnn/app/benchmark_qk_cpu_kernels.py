#!/usr/bin/env python3
"""CPU-only kernel follow-up on the same 20 images as the DMA experiment."""
import argparse
import json
import os
import pathlib
import re
import subprocess
import time
from benchmark_board_random_qk import OP, MEAN, STD, sha, scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-dir',default='/home/ubuntu/tinyvit')
    p.add_argument('--source-results',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--runner',default='tinyvit_cpu_compute_profile')
    a=p.parse_args()
    board=pathlib.Path(a.board_dir).resolve();source=pathlib.Path(a.source_results).resolve()
    out=pathlib.Path(os.path.abspath(a.output_dir));out.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/'results.json').read_text())
    source_metadata=json.loads((source/'metadata.json').read_text())
    zeros=source_metadata.get('input_mode')=='zero_fp32'
    assert len(original)==20
    env=dict(os.environ)
    for k in ('MNN_FPGA_UIO','MNN_MATMUL_SOC_UIO','MNN_MATMUL_SOC_DEVMEM','MNN_MATMUL_SOC_DMA'):
        env.pop(k,None)
    meta=dict(inputs=20,runner_sha256=sha(board/a.runner),model_sha256=sha(board/'tiny-vit.mnn'),
              original_metadata=source_metadata,threads=2,
              kernel_timer='only MNNPackedMatMul / MNNPackedMatMulRemain calls, FP32; union of overlapping intervals across threads',
              cpu_execute_timer='operand packing, kernels, output unpack, thread synchronization; union of overlapping intervals',
              fresh_process_per_inference=True,session_warmup=0,warmup_inferences=2)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))

    def execute(mode,img,label):
        target=out/label/mode;target.mkdir(parents=True)
        command=[str(board/a.runner),'--model',str(board/'tiny-vit.mnn'),'--height','224','--width','224',
                 '--threads','2','--runs','1','--output-dir',str(target)]
        command+=['--zeros'] if zeros else ['--image',str(img),'--mean',MEAN,'--std',STD]
        if mode=='profile':command+=['--profile','--fpga-op',OP]
        with (target/'run.log').open('w') as log:
            start=time.perf_counter()
            proc=subprocess.run(command,cwd=str(board),env=env,stdout=log,stderr=subprocess.STDOUT)
            wall=(time.perf_counter()-start)*1000
        log=(target/'run.log').read_text()
        timing=re.search(r'^Inference 1: ([0-9.]+) ms$',log,re.M)
        assert proc.returncode==0 and timing and 'FPGA submissions: 0' in log
        scores(target/'output_0.f32')
        record=dict(inference_ms=float(timing.group(1)),process_ms=wall,output_sha256=sha(target/'output_0.f32'))
        if mode=='profile':
            record['profile']=json.loads(re.search(r'^PROFILE_JSON (.+)$',log,re.M).group(1))
            prof=record['profile'];off=prof['offload']
            assert prof['selected_qk_calls']==1 and off['cpu_execute_calls']==64
            assert off['cpu_kernel_calls']>=64 and off['cpu_interval_overflow']==0
            assert 0<off['cpu_kernel_union_ms']<=off['cpu_execute_union_ms']<=prof['selected_qk_ms']
        return record

    for mode in ('control','profile'):
        execute(mode,source/'images'/('zeros.f32' if zeros else 'random-00.ppm'),'warmup')
    rows=[]
    for row in original:
        i=row['input'];img=source/'images'/('zeros.f32' if zeros else 'random-%02d.ppm'%i)
        assert sha(img)==row['image_sha256']
        order=('control','profile') if i%2 else ('profile','control')
        runs={}
        for mode in order:runs[mode]=execute(mode,img,'input-%02d'%i)
        assert runs['control']['output_sha256']==runs['profile']['output_sha256']==row['runs']['cpu_control']['output_sha256']
        rows.append(dict(input=i,seed=row['seed'],image_sha256=row['image_sha256'],order=list(order),runs=runs))
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        prof=runs['profile']['profile'];off=prof['offload']
        print('INPUT %02d CPU QK %.6f ms execute-union %.6f kernel-union %.6f thread-kernel %.6f calls=%d'%(
              i,prof['selected_qk_ms'],off['cpu_execute_union_ms'],off['cpu_kernel_union_ms'],off['cpu_kernel_thread_ms'],off['cpu_kernel_calls']),flush=True)
    print('COMPLETE 20 CPU-only pairs; all outputs match original CPU baseline',flush=True)


if __name__=='__main__':
    main()
