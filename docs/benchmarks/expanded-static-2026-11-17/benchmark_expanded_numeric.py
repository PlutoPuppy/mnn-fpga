#!/usr/bin/env python3
"""Compare each expanded CPU/FPGA INT8 operator on identical real inputs.

Diagnostic runs execute CPU operators, mirror them on FPGA, and preserve CPU
graph outputs. Times from this mode are not an offload performance benchmark.
Compatible with the HPS Python 3.5 environment.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
from benchmark_board_random_qk import MEAN, STD, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-dir',default='/home/ubuntu/tinyvit')
    p.add_argument('--runner',required=True)
    p.add_argument('--images',default='/home/ubuntu/tinyvit/accuracy-real-20')
    p.add_argument('--inputs',type=int,default=20)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--bitstream-sha256',required=True)
    a=p.parse_args()
    if os.geteuid()!=0 or not 1<=a.inputs<=20:p.error('Root and 1..20 images required')
    if not re.fullmatch('[0-9a-f]{64}',a.bitstream_sha256):p.error('Qualified SOF SHA256 required')
    board=pathlib.Path(a.board_dir).resolve();images=sorted(pathlib.Path(a.images).glob('*.png'))[:a.inputs]
    assert len(images)==a.inputs
    out=pathlib.Path(os.path.abspath(a.output_dir));out.mkdir(exist_ok=False)
    env=dict(os.environ)
    for key in list(env):
        if key.startswith(('MNN_FPGA','MNN_MATMUL_SOC','MNN_INT8_TRACE')):env.pop(key)
    env.update(MNN_MATMUL_SOC_DEVMEM='1',MNN_MATMUL_SOC_DMA='1')
    models={name:'tiny-vit-mixed-w8a8-all-'+name+'.mnn' for name in ('transformer','static')}
    metadata=dict(runner_sha256=sha(board/a.runner),models={k:sha(board/v) for k,v in models.items()},
        abi='NI83',clock_mhz=90,bitstream_sha256=a.bitstream_sha256,inputs=a.inputs,
        comparison='Same-input per-operator FPGA vs MNN CPU, output code tolerance1',
        cpu_graph_outputs_preserved=True,timing_is_diagnostic=True,
        images=[dict(file=x.name,sha256=sha(x)) for x in images])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[];failed=False
    for i,image in enumerate(images):
        for scope in ('transformer','static') if i%2==0 else ('static','transformer'):
            target=out/('input-%02d'%i)/scope;target.mkdir(parents=True)
            command=[str(board/a.runner),'--model',str(board/models[scope]),'--height','224','--width','224',
                '--image',str(image),'--mean',MEAN,'--std',STD,'--threads','2','--runs','1',
                '--fpga-native-int8','--native-verify','--require-fpga','--output-dir',str(target)]
            with (target/'run.log').open('w') as log:
                result=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            text=(target/'run.log').read_text();match=re.findall(r'^NATIVE_JSON (.+)$',text,re.M)
            if len(match)!=1:raise RuntimeError('No completed native diagnostic: '+text[-1500:])
            native=json.loads(match[0]);checks=native['verification'];expected=40 if scope=='transformer' else 51
            assert native['abi']=='NI83' and native['operations']==expected and len(checks)==expected
            assert native['pending']==-1 and native['fused_pairs']==0
            total=sum(v['elements'] for v in checks);over=sum(v['over_one'] for v in checks)
            maximum=max(v['maximum'] for v in checks)
            if result and 'Native verification exceeded one INT8 output code' not in text:
                raise RuntimeError('Native diagnostic execution failed: '+text[-1500:])
            assert bool(result)==bool(over)
            record=dict(input=i,image=image.name,image_sha256=sha(image),scope=scope,returncode=result,
                total_elements=total,mismatches=sum(v['mismatches'] for v in checks),over_one=over,
                maximum=maximum,native=native,command=command)
            output=target/'output_0.f32'
            if output.exists():record['output_sha256']=sha(output)
            rows.append(record);failed=failed or bool(over)
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            print('CHECK input=%02d %s ops=%d elements=%d max_code_error=%d over_one=%d'%
                (i,scope,expected,total,maximum,over),flush=True)
    print('COMPLETE',len(rows),'same-input numeric runs; gate_passed=',not failed,flush=True)
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
