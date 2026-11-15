#!/usr/bin/env python3
"""Whole-model HPS CPU profiling on the prior experiment's random RGB images.

Two fresh processes per image: an unprofiled CPU control and callback profile.
No FPGA environment variables or operation selection are enabled.
Python 3.5 compatible; no third-party modules needed on the board.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
from benchmark_board_random_qk import MEAN, STD, sha, scores, cpu_state


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-dir',default='/home/ubuntu/tinyvit')
    p.add_argument('--source-results',default='/home/ubuntu/tinyvit/benchmark-qk-profile-90mhz-20')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--runner',default='tinyvit_operator_profile')
    p.add_argument('--inputs',type=int,default=20)
    a=p.parse_args()
    board=pathlib.Path(a.board_dir).resolve();source=pathlib.Path(a.source_results).resolve()
    original=json.loads((source/'results.json').read_text())[:a.inputs]
    assert len(original)==a.inputs and a.inputs>0
    out=pathlib.Path(os.path.abspath(a.output_dir));out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    for key in ('MNN_FPGA_UIO','MNN_MATMUL_SOC_UIO','MNN_MATMUL_SOC_DEVMEM','MNN_MATMUL_SOC_DMA'):
        env.pop(key,None)
    meta=dict(inputs=a.inputs,runner_sha256=sha(board/a.runner),model_sha256=sha(board/'tiny-vit.mnn'),
              source_results=str(source),source_metadata=json.loads((source/'metadata.json').read_text()),
              threads=2,fresh_process_per_inference=True,session_warmup=0,warmup_inferences=2,
              execution='CPU only, all FPGA environment variables removed, no selected operation',
              order='alternating CPU control/profile, balanced 10/10 for 20 inputs',
              operator_timer='CPU synchronous before/after callback wall interval; descriptor capture excluded from op intervals',
              process_timer='complete subprocess wall time including profile JSON serialization after inference',
              actual_experiment_date='2026-11-11',cpu_state_before=cpu_state())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    catalog=None;catalog_hash=None

    def execute(mode,img,label):
        nonlocal catalog,catalog_hash
        target=out/label/mode;target.mkdir(parents=True)
        command=[str(board/a.runner),'--model',str(board/'tiny-vit.mnn'),'--height','224','--width','224',
                 '--image',str(img),'--mean',MEAN,'--std',STD,'--threads','2','--runs','1','--output-dir',str(target)]
        if mode=='profile':command+=['--profile-ops',str(target/'operator-profile.json')]
        with (target/'run.log').open('w') as log:
            start=time.perf_counter()
            proc=subprocess.run(command,cwd=str(board),env=env,stdout=log,stderr=subprocess.STDOUT)
            process_ms=(time.perf_counter()-start)*1000
        log=(target/'run.log').read_text()
        match=re.search(r'^Inference 1: ([0-9.]+) ms$',log,re.M)
        if proc.returncode or not match or not re.search(r'^FPGA submissions: 0$',log,re.M):
            raise RuntimeError('CPU run failed: '+str(target))
        scores(target/'output_0.f32')
        record=dict(inference_ms=float(match.group(1)),process_ms=process_ms,
                    output_sha256=sha(target/'output_0.f32'),exit_code=proc.returncode,fpga_jobs=0)
        if mode=='profile':
            profile=json.loads((target/'operator-profile.json').read_text())
            assert profile['runs']==1 and len(profile['records'])>0
            assert abs(sum(op['ms'] for op in profile['records'])-profile['sum_operator_ms'])<1e-5
            assert 0<=profile['outside_operator_ms']<profile['inference_ms']
            description=dict(records=[{k:v for k,v in op.items() if k!='ms'} for op in profile['records']],model_nodes=profile['model_nodes'])
            digest=hashlib.sha256(canonical(description)).hexdigest()
            if catalog is None:
                catalog=description;catalog_hash=digest
                (out/'operator-catalog.json').write_text(json.dumps(catalog,indent=2))
            elif digest!=catalog_hash:raise RuntimeError('Operator metadata changed: '+str(target))
            record.update(operator_ms=[op['ms'] for op in profile['records']],
                          exact_inference_ms=profile['inference_ms'],sum_operator_ms=profile['sum_operator_ms'],
                          outside_operator_ms=profile['outside_operator_ms'],operator_count=len(profile['records']),
                          catalog_sha256=digest,raw_profile_sha256=sha(target/'operator-profile.json'))
        return record

    for mode in ('control','profile'):execute(mode,source/'images'/'random-00.ppm','warmup')
    rows=[]
    for original_row in original:
        i=original_row['input'];img=source/'images'/('random-%02d.ppm'%i)
        assert sha(img)==original_row['image_sha256']
        order=('control','profile') if i%2 else ('profile','control')
        runs={}
        for mode in order:runs[mode]=execute(mode,img,'input-%02d'%i)
        assert runs['control']['output_sha256']==runs['profile']['output_sha256']==original_row['runs']['cpu_control']['output_sha256']
        rows.append(dict(input=i,seed=original_row['seed'],image_sha256=original_row['image_sha256'],order=list(order),runs=runs))
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        print('INPUT %02d CPU %.3f profile %.3f ms; %d operators sum %.3f outside %.3f ms'%(
              i,runs['control']['inference_ms'],runs['profile']['inference_ms'],runs['profile']['operator_count'],
              runs['profile']['sum_operator_ms'],runs['profile']['outside_operator_ms']),flush=True)
    meta.update(cpu_state_after=cpu_state(),operator_count=len(catalog['records']),catalog_sha256=catalog_hash)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('COMPLETE %d inputs, %d CPU inferences; all outputs match original CPU baseline'%(len(rows),2*len(rows)),flush=True)


if __name__=='__main__':
    main()
