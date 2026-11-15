#!/usr/bin/env python3
"""Analyze paired 20-image FP32/full-W8A8/fc1-W8A8 controls and profiles."""
import argparse,csv,json,pathlib,statistics as st
import numpy as np

FAMILIES=('fp32','full_cpu','full_fpga','hybrid_cpu','hybrid_fpga')
STAGES=('peak_scan','quantize','workspace','tile_plan','pack_pad','scale_bound','configure','upload','launch',
        'completion_wait','download','dequantize','output_convert','int8_accumulate','int8_requantize')
LAYOUT=('layout_allocate','input_layout','output_layout')
WORKFLOW=('preprocess_ms','model_load_ms','session_ms','input_copy_ms','linear_setup_ms','output_ms','main_elapsed_ms')

def stats(x):
    return dict(mean=st.mean(x),median=st.median(x),stddev=st.stdev(x),minimum=min(x),maximum=max(x))

def delta(left,right):
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64);d=b-a
    ids=np.random.default_rng(20261112).integers(0,len(a),size=(20000,len(a)))
    aa=a[ids].mean(axis=1);bb=b[ids].mean(axis=1)
    return dict(baseline=stats(left),other=stats(right),added_ms=stats(d.tolist()),
        latency_change_pct=float(100*(b.mean()/a.mean()-1)),baseline_over_other=float(a.mean()/b.mean()),
        paired_bootstrap_95_ci_added_ms=np.quantile(bb-aa,[.025,.975]).tolist(),
        paired_bootstrap_95_ci_change_pct=np.quantile(100*(bb/aa-1),[.025,.975]).tolist(),
        faster_inputs=int(np.count_nonzero(d<0)),slower_inputs=int(np.count_nonzero(d>0)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path);root=p.parse_args().directory
    rows=json.loads((root/'results.json').read_text());meta=json.loads((root/'metadata.json').read_text())
    assert len(rows)==meta['inputs']==20 and len({r['image_sha256'] for r in rows})==20
    for a in meta['modes']:
        for b in meta['modes']:
            if a!=b:assert sum(r['order'].index(a)<r['order'].index(b) for r in rows)==10
        assert all(sum(r['order'][i]==a for r in rows)==2 for i in range(10))
    for r in rows:
        for f in FAMILIES:
            control=r['runs'][f+'_control'];profile=r['runs'][f+'_profile']['profile'];off=profile['offload'];fpga=f.endswith('_fpga')
            assert control['output_sha256']==r['runs'][f+'_profile']['output_sha256']
            assert control['exit_code']==r['runs'][f+'_profile']['exit_code']==0 and profile['selected_qk_calls']==1
            assert control['fpga_jobs']==off['jobs']==(28 if fpga else 0)
            if fpga:
                t=off['ms'];casts=off['framework_casts'];sc=casts['selected_quantize_ms']+casts['selected_dequantize_ms']
                assert off['quantized_values']==0 and off['int8_vectorized_values']==401408
                assert off['input_bytes']==860160 and off['output_bytes']==401408
                assert all(t[k]==0 for k in ('peak_scan','quantize','dequantize','output_convert','int8_accumulate'))
                assert sum(t[k] for k in STAGES)<=t['adapter_total']+1e-5
                assert t['adapter_total']+t['device_init']<=t['bridge_total']+1e-5
                assert t['bridge_total']+sum(t[k] for k in LAYOUT)+sc<=profile['selected_operation_ms']+1e-5
    get=lambda f,kind,key:[r['runs'][f+'_'+kind][key] for r in rows]
    op=lambda f:[r['runs'][f+'_profile']['profile']['selected_operation_ms'] for r in rows]
    summary=dict(inputs=20,measured_inferences=200,warmup_inferences=10,bootstrap_replicates=20000,bootstrap_seed=20261112,
        controls={key:{f:stats(get(f,'control',key)) for f in FAMILIES} for key in ('inference_ms','process_ms')},
        selected_operation_ms={f:stats(op(f)) for f in FAMILIES},comparisons={},stages_ms={},residuals_ms={},workflow={},
        framework_casts={},setup={},scores={},instrumentation={})
    for left,right in (('fp32','full_cpu'),('full_cpu','full_fpga'),('fp32','hybrid_cpu'),('hybrid_cpu','hybrid_fpga'),
                       ('fp32','full_fpga'),('fp32','hybrid_fpga')):
        summary['comparisons'][left+'_to_'+right]={key:delta(get(left,'control',key),get(right,'control',key)) for key in ('inference_ms','process_ms')}
        summary['comparisons'][left+'_to_'+right]['selected_operation_ms']=delta(op(left),op(right))
    for f in FAMILIES:
        profiles=[r['runs'][f+'_profile']['profile'] for r in rows]
        summary['workflow'][f]={k:stats([t[k] for t in profiles]) for k in WORKFLOW+('inference_ms','selected_operation_ms')}
        casts=[p['offload']['framework_casts'] for p in profiles]
        summary['framework_casts'][f]={k:stats([t[k] for t in casts]) for k in casts[0]}
        summary['instrumentation'][f]=delta(get(f,'control','inference_ms'),get(f,'profile','inference_ms'))
        if not f.endswith('_fpga'):continue
        times=[p['offload']['ms'] for p in profiles]
        summary['stages_ms'][f]={k:stats([t[k] for t in times]) for k in times[0]}
        residual=dict(adapter_unassigned=[t['adapter_total']-sum(t[k] for k in STAGES) for t in times],
            bridge_remainder=[t['bridge_total']-t['adapter_total']-t['device_init'] for t in times],
            selected_framework_casts=[c['selected_quantize_ms']+c['selected_dequantize_ms'] for c in casts],
            outer_remainder=[p['selected_operation_ms']-t['bridge_total']-sum(t[k] for k in LAYOUT)-c['selected_quantize_ms']-c['selected_dequantize_ms'] for p,t,c in zip(profiles,times,casts)],
            non_compute=[p['selected_operation_ms']-t['completion_wait'] for p,t in zip(profiles,times)],
            rest_of_model_delta=[(p['inference_ms']-p['selected_operation_ms'])-
                (r['runs'][f.replace('_fpga','_cpu')+'_profile']['profile']['inference_ms']-
                 r['runs'][f.replace('_fpga','_cpu')+'_profile']['profile']['selected_operation_ms']) for r,p in zip(rows,profiles)])
        summary['residuals_ms'][f]={k:stats(v) for k,v in residual.items()}
        setups=[r['runs'][f+'_profile']['linear_setup'] for r in rows]
        summary['setup'][f]={k:stats([t[k] for t in setups]) for k in ('weight_setup_ms','weight_peak_scan_ms','weight_quantize_ms','plan_selection_ms')}
        summary['setup'][f]['parameters']={k:setups[0][k] for k in ('M','K','N','tile_m','tile_k','tile_n','jobs','operand_bytes','result_bytes','input_scale','output_scale','per_channel_weight_scales','operand_format')}
    for key in rows[0]['scores']:
        values=[r['scores'][key] for r in rows]
        summary['scores'][key]=dict(top1_agreement=sum(t['top1_agrees'] for t in values),top5_ranking_agreement=sum(t['top5_ranking_agrees'] for t in values),
            **{k:stats([t[k] for t in values]) for k in ('max_abs_error','rmse','relative_l2','cosine_similarity')})
    if meta.get('operator_details'):
        summary['operator_details']={}
        for f in FAMILIES:
            details=[r['runs'][f+'_profile']['operator_details'] for r in rows]
            categories=sorted({k for d in details for k in d['categories']})
            values={k:stats([d[k] for d in details]) for k in ('sum_operator_ms','outside_operator_ms',
                'fc1_explicit_quantize_ms','fc1_explicit_dequantize_ms','fc1_explicit_quantize_calls','fc1_explicit_dequantize_calls')}
            values['categories']={k:{field:stats([d['categories'].get(k,{}).get(field,0) for d in details])
                                      for field in ('calls','ms')} for k in categories}
            values['whole_model_conversion_ms']=stats([sum(d['categories'].get(k,{}).get('ms',0)
                for k in ('FloatToInt8','Int8ToFloat')) for d in details])
            values['complete_fc1_section_ms']=stats([v+q['fc1_explicit_quantize_ms']+q['fc1_explicit_dequantize_ms']
                for v,q in zip(op(f),details)])
            summary['operator_details'][f]=values
        summary['comparisons']['full_cpu_to_full_fpga']['complete_fc1_section_ms']=delta(
            [r['runs']['full_cpu_profile']['profile']['selected_operation_ms']+
             r['runs']['full_cpu_profile']['operator_details']['fc1_explicit_quantize_ms']+
             r['runs']['full_cpu_profile']['operator_details']['fc1_explicit_dequantize_ms'] for r in rows],
            [r['runs']['full_fpga_profile']['profile']['selected_operation_ms']+
             r['runs']['full_fpga_profile']['operator_details']['fc1_explicit_quantize_ms']+
             r['runs']['full_fpga_profile']['operator_details']['fc1_explicit_dequantize_ms'] for r in rows])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    phase_keys=list(summary['stages_ms']['full_fpga']);cast_keys=list(summary['framework_casts']['fp32'])
    fields=['input','seed','family','inference_ms','process_ms','profile_inference_ms','profile_process_ms','selected_operation_ms']+list(WORKFLOW)+['fpga_jobs']+phase_keys+['framework_'+k for k in cast_keys]
    with (root/'timing-breakdown.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            for f in FAMILIES:
                c=r['runs'][f+'_control'];p=r['runs'][f+'_profile']['profile']
                line=dict(input=r['input'],seed=r['seed'],family=f,inference_ms=c['inference_ms'],process_ms=c['process_ms'],
                    profile_inference_ms=p['inference_ms'],profile_process_ms=r['runs'][f+'_profile']['process_ms'],selected_operation_ms=p['selected_operation_ms'],fpga_jobs=c['fpga_jobs'])
                line.update({k:p[k] for k in WORKFLOW});line.update(p['offload']['ms']);line.update({'framework_'+k:v for k,v in p['offload']['framework_casts'].items()})
                writer.writerow(line)
    if meta.get('operator_details'):
        fields=['input','family','fc1_explicit_quantize_ms','native_or_offloaded_fc1_ms','fc1_explicit_dequantize_ms',
            'complete_fc1_section_ms','all_quantize_ms','all_dequantize_ms','all_quantize_calls','all_dequantize_calls',
            'sum_operator_ms','outside_operator_ms']
        with (root/'conversion-breakdown.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for r in rows:
                for f in FAMILIES:
                    d=r['runs'][f+'_profile']['operator_details'];native=r['runs'][f+'_profile']['profile']['selected_operation_ms']
                    q=d['categories'].get('FloatToInt8',dict(ms=0,calls=0));dq=d['categories'].get('Int8ToFloat',dict(ms=0,calls=0))
                    writer.writerow(dict(input=r['input'],family=f,fc1_explicit_quantize_ms=d['fc1_explicit_quantize_ms'],
                        native_or_offloaded_fc1_ms=native,fc1_explicit_dequantize_ms=d['fc1_explicit_dequantize_ms'],
                        complete_fc1_section_ms=native+d['fc1_explicit_quantize_ms']+d['fc1_explicit_dequantize_ms'],
                        all_quantize_ms=q['ms'],all_dequantize_ms=dq['ms'],all_quantize_calls=q['calls'],all_dequantize_calls=dq['calls'],
                        sum_operator_ms=d['sum_operator_ms'],outside_operator_ms=d['outside_operator_ms']))
    print(json.dumps({k:summary[k] for k in ('controls','selected_operation_ms','scores')},indent=2))

if __name__=='__main__':main()
