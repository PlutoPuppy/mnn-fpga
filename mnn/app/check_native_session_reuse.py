#!/usr/bin/env python3
"""Check repeated CPU/native sessions against saved fresh-process outputs.

Run as root on HPS after the paired benchmark. Python 3.5 compatible. The
runner exports the final iteration, so use independent runs=1,2,3 sessions
to observe each iteration's result without changing benchmark timing code.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',required=True,type=pathlib.Path)
    p.add_argument('--reference-results',required=True,type=pathlib.Path)
    p.add_argument('--output-dir',required=True,type=pathlib.Path)
    a=p.parse_args()
    root=a.reference_results.resolve()
    row=json.loads((root/'results.json').read_text())[0]
    out=pathlib.Path(os.path.abspath(str(a.output_dir)));out.mkdir(exist_ok=False)
    env=dict(os.environ)
    for key in list(env):
        if key.startswith(('MNN_FPGA','MNN_MATMUL_SOC','MNN_INT8_TRACE')):env.pop(key)
    checks=[]
    for mode in ('cpu','pairs'):
        reference=row['runs'][mode+'_control']
        expected=(root/('input-%02d'%row['input'])/(mode+'_control')/'output_0.f32').read_bytes()
        assert hashlib.sha256(expected).hexdigest()==reference['output_sha256']
        for count in (1,2,3):
            folder=out/('%s-runs%d'%(mode,count));folder.mkdir()
            command=list(reference['command']);command[0]=str(a.runner.resolve())
            command[command.index('--runs')+1]=str(count)
            command[command.index('--output-dir')+1]=str(folder)
            environment=dict(env)
            if mode=='pairs':environment.update(MNN_MATMUL_SOC_DEVMEM='1',MNN_MATMUL_SOC_DMA='1')
            with (folder/'run.log').open('w') as log:
                rc=subprocess.call(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
            assert rc==0,'Runner failed: '+str(folder)
            text=(folder/'run.log').read_text()
            jobs=int(re.findall(r'^FPGA submissions: (\d+)$',text,re.M)[0])
            assert jobs==reference['jobs']*count
            assert (folder/'output_0.f32').read_bytes()==expected,'Session reuse changed scores: '+str(folder)
            if mode=='pairs':
                native=json.loads(re.findall(r'^NATIVE_JSON (.+)$',text,re.M)[0])
                assert native['pending']==-1 and native['fused_pairs']==4*count
                assert native['skipped_runtime_operations']==16*count
            checks.append(dict(mode=mode,runs=count,jobs=jobs,output_sha256=reference['output_sha256']))
            (out/'checks.json').write_text(json.dumps(checks,indent=2))
            print('PASS %s runs=%d jobs=%d; final output matches fresh reference'%(mode,count,jobs),flush=True)
    print('PASS six session-reuse cases, twelve inferences',flush=True)


if __name__=='__main__':main()
