#!/usr/bin/env python3
"""Paired FP32 / broad W8A8 / fc1-only W8A8 CPU and FPGA experiment on HPS."""
import argparse,json,os,pathlib,re,subprocess,time
from benchmark_board_random_qk import MEAN,STD,sha,scores,compare,cpu_state

OP='/layers.1/blocks.0/mlp/fc1/Add_output_0__matmul_converted'
FAMILIES=('fp32','full_cpu','full_fpga','hybrid_cpu','hybrid_fpga')
MODES=tuple(f+'_control' for f in FAMILIES)+tuple(f+'_profile' for f in FAMILIES)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-dir',default='/home/ubuntu/tinyvit')
    p.add_argument('--source-results',default='/home/ubuntu/tinyvit/benchmark-qk-profile-90mhz-20')
    p.add_argument('--output-dir',required=True);p.add_argument('--inputs',type=int,default=20)
    p.add_argument('--runner',default='tinyvit_fc1_w8a8_v4');p.add_argument('--bitstream-sha256',required=True)
    p.add_argument('--full-model',default='tiny-vit-w8a8-kl.mnn',help='Broad or repaired mixed-precision model filename')
    p.add_argument('--operator-details',action='store_true',help='Profile explicit conversion nodes and all CPU operators in separate profile runs')
    a=p.parse_args()
    if os.geteuid()!=0 or a.inputs<=0:p.error('Run as root with positive input count')
    board=pathlib.Path(a.board_dir).resolve();source=pathlib.Path(a.source_results).resolve()
    original=json.loads((source/'results.json').read_text())[:a.inputs];assert len(original)==a.inputs
    out=pathlib.Path(os.path.abspath(a.output_dir));out.mkdir(parents=True,exist_ok=False)
    model_names=dict(fp32='tiny-vit.mnn',full_cpu=a.full_model,full_fpga=a.full_model,
                     hybrid_cpu='tiny-vit-fc1-w8a8.mnn',hybrid_fpga='tiny-vit-fc1-w8a8.mnn')
    env=dict(os.environ)
    for key in ('MNN_FPGA_UIO','MNN_MATMUL_SOC_UIO','MNN_MATMUL_SOC_DEVMEM','MNN_MATMUL_SOC_DMA','MNN_MATMUL_SOC_TILE_POLICY','MNN_INT8_TRACE'):
        env.pop(key,None)
    metadata=dict(inputs=a.inputs,modes=list(MODES),families=list(FAMILIES),operation=OP,M=784,K=128,N=512,
        threads=2,fpga_clock_mhz=90,programmed_bitstream_sha256=a.bitstream_sha256,
        runner_sha256=sha(board/a.runner),models={f:dict(file=name,sha256=sha(board/name),bytes=(board/name).stat().st_size) for f,name in model_names.items()},
        source_results=str(source),source_metadata=json.loads((source/'metadata.json').read_text()),
        actual_experiment_date='2026-11-12',board_wall_clock_may_be_incorrect=True,fresh_process_per_inference=True,
        warmup_inferences=10,measured_inferences=a.inputs*10,session_warmup=0,cpu_state_before=cpu_state(),
        order='Ten cyclic Williams rows paired with reversals; 20 images balance pair precedence10/10 and each mode position2/20',
        full_quantization='MNN3.6.2 KL,128 natural Imagenette calibration images,68 quantized convolutions; other operators may remain FP32',
        hybrid_quantization='Only selected fc1 and its two activation quantization records copied from calibrated model; other operators retain FP32 parameters',
        replacement='INT8 convolution executor hook after MNN runtime casts; one selected fc1 only',
        inference_timer='runSession; excludes model/session load, weight decode/transpose/plan setup, preprocessing and output save',
        compute_timer='Host FPGA start-to-done includes polling; no hardware cycle counter')
    metadata['operator_details']=a.operator_details
    if a.full_model=='tiny-vit-mixed-w8a8-accuracy.mnn':
        metadata['full_quantization']='Corrected mixed precision:37 W8A8 convolutions,31 FP32 convolutions;37 explicit quantize/dequantize pairs; non-convolution arithmetic FP32'
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))

    def execute(mode,image,label):
        family,kind=mode.rsplit('_',1);profiled=kind=='profile';fpga=family.endswith('_fpga')
        target=out/label/mode;target.mkdir(parents=True);runenv=dict(env)
        command=[str(board/a.runner),'--model',str(board/model_names[family]),'--height','224','--width','224',
            '--image',str(image),'--mean',MEAN,'--std',STD,'--threads','2','--runs','1','--output-dir',str(target)]
        if fpga:
            runenv.update(MNN_MATMUL_SOC_DEVMEM='1',MNN_MATMUL_SOC_DMA='1',MNN_MATMUL_SOC_TILE_POLICY='capacity')
            command+=['--fpga-op',OP,'--fpga-linear-int8','--require-fpga']
        if profiled:
            command+=['--profile']
            if not fpga:command+=['--fpga-op',OP]
            if a.operator_details:command+=['--profile-ops',str(target/'operators.json')]
        with (target/'run.log').open('w') as log:
            start=time.perf_counter();proc=subprocess.run(command,cwd=str(board),env=runenv,stdout=log,stderr=subprocess.STDOUT)
            process_ms=(time.perf_counter()-start)*1000
        log=(target/'run.log').read_text();inf=re.search(r'^Inference 1: ([0-9.]+) ms$',log,re.M)
        jobs=re.search(r'^FPGA submissions: (\d+)$',log,re.M);expected=28 if fpga else 0
        if proc.returncode or not inf or not jobs or int(jobs.group(1))!=expected:
            raise RuntimeError('Failed inference or job count: '+str(target))
        values=scores(target/'output_0.f32')
        record=dict(inference_ms=float(inf.group(1)),process_ms=process_ms,fpga_jobs=expected,exit_code=proc.returncode,
                    output_sha256=sha(target/'output_0.f32'))
        if fpga:
            assert re.search(r'^Linear replacements: 1$',log,re.M)
            setup=json.loads(re.search(r'^LINEAR_SETUP_JSON (.+)$',log,re.M).group(1));record['linear_setup']=setup
            assert setup['ready'] and setup['operand_format']=='native_w8a8' and setup['weight_values']==65536
            assert setup['weight_peak_scan_ms']==setup['weight_quantize_ms']==0 and setup['per_channel_weight_scales']==512
            assert (setup['tile_m'],setup['tile_k'],setup['tile_n'],setup['jobs'])==(128,128,128,28)
        if profiled:
            profile=json.loads(re.search(r'^PROFILE_JSON (.+)$',log,re.M).group(1));record['profile']=profile
            assert profile['selected_qk_calls']==1 and profile['selected_operation_name']==OP
            assert profile['linear_replacements']==int(fpga) and profile['offload']['jobs']==expected
            if fpga:
                off=profile['offload'];t=off['ms']
                assert off['input_bytes']==860160 and off['output_bytes']==401408 and off['quantized_values']==0
                assert off['input_descriptors']==108 and off['output_descriptors']==52
                assert off['int8_vectorized_values']==401408
                assert all(t[key]==0 for key in ('peak_scan','quantize','dequantize','output_convert','int8_accumulate'))
            if a.operator_details:
                detail=json.loads((target/'operators.json').read_text());ops=detail['records']
                categories={}
                for op in ops:
                    kind=op['type'].split()[0]
                    if kind not in categories:categories[kind]=dict(calls=0,ms=0.)
                    categories[kind]['calls']+=1;categories[kind]['ms']+=op['ms']
                quant=[op for op in ops if op['name']==OP+'/quantize']
                dequant=[op for op in ops if op['name']==OP+'/dequantize']
                record['operator_details']=dict(categories=categories,sum_operator_ms=detail['sum_operator_ms'],
                    outside_operator_ms=detail['outside_operator_ms'],
                    fc1_explicit_quantize_ms=sum(op['ms'] for op in quant),
                    fc1_explicit_dequantize_ms=sum(op['ms'] for op in dequant),
                    fc1_explicit_quantize_calls=len(quant),fc1_explicit_dequantize_calls=len(dequant))
                if a.full_model=='tiny-vit-mixed-w8a8-accuracy.mnn' and family.startswith('full_'):
                    assert categories['FloatToInt8']['calls']==categories['Int8ToFloat']['calls']==37
                    assert len(quant)==len(dequant)==1
                    conv=[op for op in ops if op['type'].split()[0] in ('Convolution','ConvolutionDepthwise')]
                    assert sum(op['inputs'][0]['bits']==8 for op in conv)==37
                    assert sum(op['inputs'][0]['bits']==32 for op in conv)==31
        return record,values

    warmups={}
    for mode in MODES:warmups[mode],_=execute(mode,source/'images/random-00.ppm','warmup')
    (out/'warmups.json').write_text(json.dumps(warmups,indent=2))
    base=(0,1,9,2,8,3,7,4,6,5);rows=[]
    for old in original:
        i=old['input'];image=source/'images'/('random-%02d.ppm'%i);assert sha(image)==old['image_sha256']
        shift=((i-1)//2)%10;order=tuple(MODES[(x+shift)%10] for x in base)
        if i%2==0:order=tuple(reversed(order))
        runs,outputs={},{}
        for mode in order:runs[mode],outputs[mode]=execute(mode,image,'input-%02d'%i)
        for f in FAMILIES:assert runs[f+'_control']['output_sha256']==runs[f+'_profile']['output_sha256']
        assert runs['fp32_control']['output_sha256']==old['runs']['cpu_control']['output_sha256']
        checks={name:compare(outputs[left+'_control'],outputs[right+'_control']) for name,left,right in (
            ('full_cpu_vs_fp32','fp32','full_cpu'),('full_fpga_vs_full_cpu','full_cpu','full_fpga'),
            ('hybrid_cpu_vs_fp32','fp32','hybrid_cpu'),('hybrid_fpga_vs_hybrid_cpu','hybrid_cpu','hybrid_fpga'),
            ('full_fpga_vs_fp32','fp32','full_fpga'),('hybrid_fpga_vs_fp32','fp32','hybrid_fpga'))}
        rows.append(dict(input=i,seed=old['seed'],image_sha256=old['image_sha256'],order=list(order),runs=runs,scores=checks))
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        print('INPUT %02d inference ms: FP32 %.3f full CPU/FPGA %.3f/%.3f hybrid CPU/FPGA %.3f/%.3f; fc1 full %.3f/%.3f hybrid %.3f/%.3f'%(
            i,runs['fp32_control']['inference_ms'],runs['full_cpu_control']['inference_ms'],runs['full_fpga_control']['inference_ms'],
            runs['hybrid_cpu_control']['inference_ms'],runs['hybrid_fpga_control']['inference_ms'],
            runs['full_cpu_profile']['profile']['selected_operation_ms'],runs['full_fpga_profile']['profile']['selected_operation_ms'],
            runs['hybrid_cpu_profile']['profile']['selected_operation_ms'],runs['hybrid_fpga_profile']['profile']['selected_operation_ms']),flush=True)
    metadata['cpu_state_after']=cpu_state();(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('COMPLETE %d images, %d measured inferences; native INT8 fc1 only on FPGA'%(len(rows),len(rows)*10),flush=True)

if __name__=='__main__':main()
