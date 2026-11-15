#!/usr/bin/env python3
"""Paired CPU/fixed-panel/capacity-panel fc1 experiment, Python 3.5 on HPS."""
import argparse,json,os,pathlib,re,subprocess,time
from benchmark_board_random_qk import MEAN,STD,sha,scores,compare,cpu_state

OP='/layers.1/blocks.0/mlp/fc1/Add_output_0__matmul_converted'
MODES=('cpu_control','fixed_control','capacity_control','cpu_profile','fixed_profile','capacity_profile')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-dir',default='/home/ubuntu/tinyvit')
    p.add_argument('--source-results',default='/home/ubuntu/tinyvit/benchmark-qk-profile-90mhz-20')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--runner',default='tinyvit_fc1_capacity_v2')
    p.add_argument('--inputs',type=int,default=20)
    p.add_argument('--bitstream-sha256',required=True)
    a=p.parse_args()
    if os.geteuid()!=0:p.error('Run as root; all modes use the same privileges')
    board=pathlib.Path(a.board_dir).resolve();source=pathlib.Path(a.source_results).resolve()
    original=json.loads((source/'results.json').read_text())[:a.inputs]
    assert len(original)==a.inputs and a.inputs>0
    out=pathlib.Path(os.path.abspath(a.output_dir));out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    for key in ('MNN_FPGA_UIO','MNN_MATMUL_SOC_UIO','MNN_MATMUL_SOC_DEVMEM','MNN_MATMUL_SOC_DMA','MNN_MATMUL_SOC_TILE_POLICY'):
        env.pop(key,None)
    metadata=dict(inputs=a.inputs,modes=list(MODES),operation=OP,M=784,K=128,N=512,threads=2,
                  runner_sha256=sha(board/a.runner),model_sha256=sha(board/'tiny-vit.mnn'),
                  fpga_clock_mhz=90,programmed_bitstream_sha256=a.bitstream_sha256,
                  source_results=str(source),source_metadata=json.loads((source/'metadata.json').read_text()),
                  actual_experiment_date='2026-11-12',fresh_process_per_inference=True,session_warmup=0,
                  warmup_inferences=6,one_time_weights='FP32 OI model weights transposed/quantized once before inference; setup included in process time',
                  replacement='before callback skips selected native Convolution; after callback fills its output with FPGA result',
                  tensor_layout='Explicit MNN CAFFE host conversions adapt packed NC4HW4 input/output; included in inference',
                  order='10 Williams-design rows paired with their reversals; each pair of modes precedes the other on 10 of 20 inputs',
                  inference_timer='runSession only; excludes static weight/plan setup, model/session load and output save',
                  compute_timer='host FPGA start-to-done includes polling, not a hardware cycle counter',
                  cpu_state_before=cpu_state())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))

    def execute(mode,image,label):
        target=out/label/mode;target.mkdir(parents=True)
        family=mode.split('_')[0];profiled=mode.endswith('_profile');runenv=dict(env)
        command=[str(board/a.runner),'--model',str(board/'tiny-vit.mnn'),'--height','224','--width','224',
                 '--image',str(image),'--mean',MEAN,'--std',STD,'--threads','2','--runs','1','--output-dir',str(target)]
        if family!='cpu':
            runenv.update(MNN_MATMUL_SOC_DEVMEM='1',MNN_MATMUL_SOC_DMA='1',
                          MNN_MATMUL_SOC_TILE_POLICY='fixed64' if family=='fixed' else 'capacity')
            command+=['--fpga-op',OP,'--fpga-linear','--require-fpga']
        if profiled:
            command+=['--profile']
            if family=='cpu':command+=['--fpga-op',OP]
        with (target/'run.log').open('w') as log:
            start=time.perf_counter()
            proc=subprocess.run(command,cwd=str(board),env=runenv,stdout=log,stderr=subprocess.STDOUT)
            process_ms=(time.perf_counter()-start)*1000
        log=(target/'run.log').read_text()
        inference=re.search(r'^Inference 1: ([0-9.]+) ms$',log,re.M)
        jobs=re.search(r'^FPGA submissions: (\d+)$',log,re.M)
        expected_jobs=0 if family=='cpu' else 16 if family=='fixed' else 28
        if proc.returncode or not inference or not jobs or int(jobs.group(1))!=expected_jobs:
            raise RuntimeError('Inference failed or unexpected FPGA job count: '+str(target))
        values=scores(target/'output_0.f32')
        record=dict(inference_ms=float(inference.group(1)),process_ms=process_ms,fpga_jobs=expected_jobs,
                    output_sha256=sha(target/'output_0.f32'),exit_code=proc.returncode)
        if family!='cpu':
            assert re.search(r'^Linear replacements: 1$',log,re.M)
            setup=json.loads(re.search(r'^LINEAR_SETUP_JSON (.+)$',log,re.M).group(1))
            assert setup['ready'] and setup['jobs']==expected_jobs and setup['weight_values']==65536
            assert setup['M']==784 and setup['N']==512 and setup['K']==128
            assert (setup['tile_m'],setup['tile_k'],setup['tile_n'])==((256,64,256) if family=='fixed' else (128,128,128))
            record['linear_setup']=setup
        if profiled:
            profile=json.loads(re.search(r'^PROFILE_JSON (.+)$',log,re.M).group(1));record['profile']=profile
            assert profile['selected_qk_calls']==1 and profile['selected_operation_name']==OP
            assert profile['linear_replacements']==(0 if family=='cpu' else 1)
            off=profile['offload']
            assert off['jobs']==expected_jobs
            if family!='cpu':
                assert off['input_bytes']==setup['operand_bytes'] and off['output_bytes']==setup['result_bytes']
                assert off['quantized_values']==100352
                assert off['input_descriptors']==(60 if family=='fixed' else 108)
                assert off['output_descriptors']==(100 if family=='fixed' else 52)
        return record,values

    warm=source/'images/random-00.ppm';warmups={}
    for mode in MODES:warmups[mode],_=execute(mode,warm,'warmup')
    (out/'warmups.json').write_text(json.dumps(warmups,indent=2))
    base=(0,1,5,2,4,3);rows=[]
    for row in original:
        i=row['input'];image=source/'images'/('random-%02d.ppm'%i)
        assert sha(image)==row['image_sha256']
        shift=((i-1)//2)%6;order=tuple(MODES[(x+shift)%6] for x in base)
        if i%2==0:order=tuple(reversed(order))
        runs,outputs={},{}
        for mode in order:runs[mode],outputs[mode]=execute(mode,image,'input-%02d'%i)
        for family in ('cpu','fixed','capacity'):
            assert runs[family+'_control']['output_sha256']==runs[family+'_profile']['output_sha256']
        assert runs['cpu_control']['output_sha256']==row['runs']['cpu_control']['output_sha256']
        record=dict(input=i,seed=row['seed'],image_sha256=row['image_sha256'],order=list(order),runs=runs,
                    scores={family:compare(outputs['cpu_control'],outputs[family+'_control']) for family in ('fixed','capacity')})
        rows.append(record);(out/'results.json').write_text(json.dumps(rows,indent=2))
        print('INPUT %02d CPU %.3f fixed %.3f capacity %.3f ms; fc1 CPU %.3f fixed %.3f capacity %.3f; fabric %.3f/%.3f ms'%(
              i,runs['cpu_control']['inference_ms'],runs['fixed_control']['inference_ms'],runs['capacity_control']['inference_ms'],
              runs['cpu_profile']['profile']['selected_operation_ms'],runs['fixed_profile']['profile']['selected_operation_ms'],
              runs['capacity_profile']['profile']['selected_operation_ms'],runs['fixed_profile']['profile']['offload']['ms']['completion_wait'],
              runs['capacity_profile']['profile']['offload']['ms']['completion_wait']),flush=True)
    metadata['cpu_state_after']=cpu_state();(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('COMPLETE %d images, %d measured inferences; only one fc1 replaced per FPGA inference'%(len(rows),len(rows)*6),flush=True)


if __name__=='__main__':main()
