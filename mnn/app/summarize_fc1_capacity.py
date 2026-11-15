#!/usr/bin/env python3
"""Analyze the 20-image CPU/fixed/capacity fc1 experiment."""
import argparse,csv,json,pathlib,statistics as st
from summarize_qk_profile import stats,delta_stats

STAGES=('peak_scan','quantize','workspace','tile_plan','pack_pad','scale_bound','configure',
        'upload','launch','completion_wait','download','dequantize','output_convert')
LAYOUT=('layout_allocate','input_layout','output_layout')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path);a=p.parse_args();root=a.directory
    rows=json.loads((root/'results.json').read_text());meta=json.loads((root/'metadata.json').read_text())
    assert len(rows)==meta['inputs']==20 and len({r['image_sha256'] for r in rows})==20
    for left in meta['modes']:
        for right in meta['modes']:
            if left!=right:assert sum(r['order'].index(left)<r['order'].index(right) for r in rows)==10
    for r in rows:
        for family in ('cpu','fixed','capacity'):
            control=r['runs'][family+'_control'];profile=r['runs'][family+'_profile'];pr=profile['profile'];off=pr['offload']
            assert control['exit_code']==profile['exit_code']==0 and control['output_sha256']==profile['output_sha256']
            assert pr['selected_qk_calls']==1
            if family=='cpu':assert control['fpga_jobs']==profile['fpga_jobs']==off['jobs']==0
            else:
                jobs=16 if family=='fixed' else 28;setup=profile['linear_setup'];t=off['ms']
                assert control['fpga_jobs']==profile['fpga_jobs']==off['jobs']==jobs
                assert setup['jobs']==jobs and pr['linear_replacements']==1 and off['quantized_values']==100352
                assert off['input_bytes']==setup['operand_bytes'] and off['output_bytes']==setup['result_bytes']
                assert sum(t[k] for k in STAGES)<=t['adapter_total']+1e-5
                assert t['adapter_total']+t['device_init']<=t['bridge_total']+1e-5
                assert t['bridge_total']+sum(t[k] for k in LAYOUT)<=pr['selected_operation_ms']+1e-5
                assert sum(t[k] for k in ('input_stage_copy','input_dma_setup','input_dma_wait'))<=t['upload']+1e-5
                assert sum(t[k] for k in ('output_stage_copy','output_dma_setup','output_dma_wait'))<=t['download']+1e-5
    get=lambda family,kind,metric:[r['runs'][family+'_'+kind][metric] for r in rows]
    op=lambda family:[r['runs'][family+'_profile']['profile']['selected_operation_ms'] for r in rows]
    summary=dict(inputs=20,measured_inferences=120,warmup_inferences=6,
                 control_inference={f:stats(get(f,'control','inference_ms')) for f in ('cpu','fixed','capacity')},
                 selected_operation_ms={f:stats(op(f)) for f in ('cpu','fixed','capacity')},
                 comparisons={},stages_ms={},residuals_ms={},setup={},scores={},instrumentation={},workflow={})
    for left,right in (('cpu','fixed'),('cpu','capacity'),('fixed','capacity')):
        key=left+'_to_'+right
        summary['comparisons'][key]={metric:delta_stats(get(left,'control',metric),get(right,'control',metric)) for metric in ('inference_ms','process_ms')}
        summary['comparisons'][key]['selected_operation_ms']=delta_stats(op(left),op(right))
    for family in ('cpu','fixed','capacity'):
        summary['instrumentation'][family]=delta_stats(get(family,'control','inference_ms'),get(family,'profile','inference_ms'))
        summary['workflow'][family]={key:stats([r['runs'][family+'_profile']['profile'][key] for r in rows]) for key in (
            'preprocess_ms','model_load_ms','session_ms','input_copy_ms','linear_setup_ms','inference_ms','selected_operation_ms','output_ms','main_elapsed_ms')}
        if family=='cpu':continue
        times=[r['runs'][family+'_profile']['profile']['offload']['ms'] for r in rows]
        summary['stages_ms'][family]={key:stats([t[key] for t in times]) for key in times[0]}
        summaries=[r['runs'][family+'_profile']['linear_setup'] for r in rows]
        summary['setup'][family]={key:stats([t[key] for t in summaries]) for key in ('weight_setup_ms','weight_peak_scan_ms','weight_quantize_ms','plan_selection_ms')}
        summary['setup'][family]['plan']={key:summaries[0][key] for key in ('M','K','N','tile_m','tile_k','tile_n','jobs','operand_bytes','result_bytes','accumulated_values','cost_proxy_ns','weight_values','policy')}
        residuals=dict(adapter_unassigned=[t['adapter_total']-sum(t[k] for k in STAGES) for t in times],
                       bridge_remainder=[t['bridge_total']-t['adapter_total']-t['device_init'] for t in times],
                       callback_remainder=[r['runs'][family+'_profile']['profile']['selected_operation_ms']-t['bridge_total']-sum(t[k] for k in LAYOUT) for r,t in zip(rows,times)],
                       layout_total=[sum(t[k] for k in LAYOUT) for t in times],
                       input_transport_remainder=[t['upload']-sum(t[k] for k in ('input_stage_copy','input_dma_setup','input_dma_wait')) for t in times],
                       output_transport_remainder=[t['download']-sum(t[k] for k in ('output_stage_copy','output_dma_setup','output_dma_wait')) for t in times],
                       non_compute=[r['runs'][family+'_profile']['profile']['selected_operation_ms']-t['completion_wait'] for r,t in zip(rows,times)],
                       rest_of_model_delta=[(r['runs'][family+'_profile']['profile']['inference_ms']-r['runs'][family+'_profile']['profile']['selected_operation_ms'])-
                                            (r['runs']['cpu_profile']['profile']['inference_ms']-r['runs']['cpu_profile']['profile']['selected_operation_ms']) for r in rows])
        summary['residuals_ms'][family]={key:stats(value) for key,value in residuals.items()}
        summary['scores'][family]=dict(top1_agreement=sum(r['scores'][family]['top1_agrees'] for r in rows),
            top5_ranking_agreement=sum(r['scores'][family]['top5_ranking_agrees'] for r in rows),
            max_abs_error=stats([r['scores'][family]['max_abs_error'] for r in rows]),
            relative_l2=stats([r['scores'][family]['relative_l2'] for r in rows]),
            cosine_similarity=stats([r['scores'][family]['cosine_similarity'] for r in rows]))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    workflow=('preprocess_ms','model_load_ms','session_ms','input_copy_ms','linear_setup_ms','output_ms','main_elapsed_ms')
    fields=['input','seed','family','inference_ms','process_ms','profile_inference_ms','profile_process_ms','selected_operation_ms']+list(workflow)+['weight_setup_ms','plan_selection_ms','jobs','tile_m','tile_k','tile_n']+list(summary['stages_ms']['fixed'])
    with (root/'timing-breakdown.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            for family in ('cpu','fixed','capacity'):
                control=r['runs'][family+'_control'];pr=r['runs'][family+'_profile']['profile']
                line=dict(input=r['input'],seed=r['seed'],family=family,inference_ms=control['inference_ms'],
                          process_ms=control['process_ms'],profile_inference_ms=pr['inference_ms'],
                          profile_process_ms=r['runs'][family+'_profile']['process_ms'],
                          selected_operation_ms=pr['selected_operation_ms'],jobs=control['fpga_jobs'])
                line.update({key:pr[key] for key in workflow})
                if family!='cpu':
                    setup=r['runs'][family+'_profile']['linear_setup']
                    line.update({key:setup[key] for key in ('weight_setup_ms','plan_selection_ms','tile_m','tile_k','tile_n')})
                    line.update(pr['offload']['ms'])
                writer.writerow(line)
    print(json.dumps({key:summary[key] for key in ('control_inference','selected_operation_ms','scores')},indent=2))
    for name,c in summary['comparisons'].items():
        print(name,'inference +ms',c['inference_ms']['added_ms']['mean'],'pct',c['inference_ms']['latency_increase_pct'],
              'CI',c['inference_ms']['paired_bootstrap_95_ci_added_ms'],'selected +ms',c['selected_operation_ms']['added_ms']['mean'])


if __name__=='__main__':main()
