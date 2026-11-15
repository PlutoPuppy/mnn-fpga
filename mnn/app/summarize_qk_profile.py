#!/usr/bin/env python3
"""Analyze the CPU/DMA quartet experiment; stage times are host wall times."""
import argparse
import csv
import json
import pathlib
import random
import statistics as st

STAGES = ('peak_scan','quantize','workspace','pack_pad','scale_bound','configure',
          'upload','launch','completion_wait','download','dequantize','output_convert')


def stats(values):
    return dict(mean=st.mean(values),median=st.median(values),stddev=st.stdev(values),
                minimum=min(values),maximum=max(values))


def delta_stats(cpu,dma):
    rng = random.Random(20261111)
    values = [b-a for a,b in zip(cpu,dma)]
    boot, pct = [], []
    for _ in range(20000):
        ids = [rng.randrange(len(cpu)) for _ in cpu]
        a,b = st.mean(cpu[i] for i in ids),st.mean(dma[i] for i in ids)
        boot.append(b-a);pct.append(100*(b-a)/a)
    boot.sort();pct.sort()
    return dict(cpu=stats(cpu),dma=stats(dma),added_ms=stats(values),
                latency_increase_pct=100*(st.mean(dma)-st.mean(cpu))/st.mean(cpu),
                cpu_over_dma_ratio=st.mean(cpu)/st.mean(dma),
                paired_bootstrap_95_ci_added_ms=[boot[500],boot[19499]],
                paired_bootstrap_95_ci_increase_pct=[pct[500],pct[19499]],
                dma_faster_inputs=sum(v<0 for v in values),dma_slower_inputs=sum(v>0 for v in values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=pathlib.Path)
    a=p.parse_args();root=a.directory
    rows=json.loads((root/'results.json').read_text())
    meta=json.loads((root/'metadata.json').read_text())
    assert len(rows)==meta['inputs']==20
    zeros=meta.get('input_mode')=='zero_fp32'
    assert len({r['image_sha256'] for r in rows})==(1 if zeros else 20)
    for mode in meta['modes']:
        assert all(sum(r['order'][i]==mode for r in rows)==5 for i in range(4))
    if meta.get('order_design')=='williams':
        for left in meta['modes']:
            for right in meta['modes']:
                if left!=right: assert sum(r['order'].index(left)<r['order'].index(right) for r in rows)==10
    for row in rows:
        r=row['runs'];cp=r['cpu_profile']['profile'];dp=r['dma_profile']['profile'];off=dp['offload']
        assert cp['selected_qk_calls']==dp['selected_qk_calls']==1
        assert cp['offload']['cpu_execute_calls']==64
        assert off['cpu_execute_calls']==0
        assert off['jobs']==64 and off['quantized_values']==200704
        assert off['input_bytes']==off['output_bytes']==262144
        assert off['input_descriptors']==128 and off['output_descriptors']==64
        assert r['cpu_control']['output_sha256']==r['cpu_profile']['output_sha256']
        assert r['dma_control']['output_sha256']==r['dma_profile']['output_sha256']
        t=off['ms']
        assert sum(t[k] for k in STAGES)<=t['adapter_total']+1e-5
        assert t['adapter_total']+t['device_init']<=t['bridge_total']+1e-5
        assert t['bridge_total']<=dp['selected_qk_ms']+1e-5
        assert sum(t[k] for k in ('input_stage_copy','input_dma_setup','input_dma_wait'))<=t['upload']+1e-5
        assert sum(t[k] for k in ('output_stage_copy','output_dma_setup','output_dma_wait'))<=t['download']+1e-5
    get=lambda mode,key:[r['runs'][mode][key] for r in rows]
    prof=lambda mode,key:[r['runs'][mode]['profile'][key] for r in rows]
    times=[r['runs']['dma_profile']['profile']['offload']['ms'] for r in rows]
    summary=dict(inputs=20,measured_inferences=80,warmup_inferences=4,
                 input_mode=meta.get('input_mode','random_rgb'),
                 bootstrap_replicates=20000,bootstrap_seed=20261111,
                 controls={k:delta_stats(get('cpu_control',k),get('dma_control',k)) for k in ('inference_ms','process_ms')},
                 profiles={k:delta_stats(get('cpu_profile',k),get('dma_profile',k)) for k in ('inference_ms','process_ms')},
                 selected_qk=delta_stats(prof('cpu_profile','selected_qk_ms'),prof('dma_profile','selected_qk_ms')),
                 cpu_execute_aggregate_thread_ms=stats([r['runs']['cpu_profile']['profile']['offload']['cpu_execute_ms'] for r in rows]),
                 stages_ms={k:stats([t[k] for t in times]) for k in times[0]},
                 instrumentation={m:delta_stats(get(m+'_control','inference_ms'),get(m+'_profile','inference_ms')) for m in ('cpu','dma')},
                 runner_phases={m:{k:stats(prof(m,k)) for k in ('preprocess_ms','model_load_ms','session_ms','input_copy_ms','inference_ms','selected_qk_ms','output_ms','main_elapsed_ms')} for m in ('cpu_profile','dma_profile')})
    residuals={
        'adapter_unassigned':[t['adapter_total']-sum(t[k] for k in STAGES) for t in times],
        'bridge_logging_and_bookkeeping':[t['bridge_total']-t['adapter_total']-t['device_init'] for t in times],
        'mnn_qk_outside_bridge':[r['runs']['dma_profile']['profile']['selected_qk_ms']-t['bridge_total'] for r,t in zip(rows,times)],
        'input_transport_unassigned':[t['upload']-sum(t[k] for k in ('input_stage_copy','input_dma_setup','input_dma_wait')) for t in times],
        'output_transport_unassigned':[t['download']-sum(t[k] for k in ('output_stage_copy','output_dma_setup','output_dma_wait')) for t in times],
        'dma_qk_excluding_completion':[r['runs']['dma_profile']['profile']['selected_qk_ms']-t['completion_wait'] for r,t in zip(rows,times)],
        'profiled_rest_of_inference_delta':[(r['runs']['dma_profile']['profile']['inference_ms']-r['runs']['dma_profile']['profile']['selected_qk_ms'])-(r['runs']['cpu_profile']['profile']['inference_ms']-r['runs']['cpu_profile']['profile']['selected_qk_ms']) for r in rows]}
    summary['residuals_ms']={k:stats(v) for k,v in residuals.items()}
    summary['scores']=dict(top1_agreement=sum(r['scores']['top1_agrees'] for r in rows),
        top5_ranking_agreement=sum(r['scores']['top5_ranking_agrees'] for r in rows),
        max_abs_error=max(r['scores']['max_abs_error'] for r in rows))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    columns=['input','seed','cpu_inference_ms','dma_inference_ms','added_ms','increase_pct',
             'cpu_process_ms','dma_process_ms','cpu_qk_ms','dma_qk_ms','cpu_execute_aggregate_thread_ms']+list(times[0])
    with (root/'timing-breakdown.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row,t in zip(rows,times):
            r=row['runs'];cpu=r['cpu_control']['inference_ms'];dma=r['dma_control']['inference_ms']
            line=dict(input=row['input'],seed=row['seed'],cpu_inference_ms=cpu,dma_inference_ms=dma,
                      added_ms=dma-cpu,increase_pct=100*(dma-cpu)/cpu,
                      cpu_process_ms=r['cpu_control']['process_ms'],dma_process_ms=r['dma_control']['process_ms'],
                      cpu_qk_ms=r['cpu_profile']['profile']['selected_qk_ms'],dma_qk_ms=r['dma_profile']['profile']['selected_qk_ms'],
                      cpu_execute_aggregate_thread_ms=r['cpu_profile']['profile']['offload']['cpu_execute_ms'])
            line.update(t);writer.writerow(line)
    lines=['| Phase (ms per inference, all 64 heads) | Mean | Median | SD | Min | Max |',
           '|---|---:|---:|---:|---:|---:|']
    for k,v in summary['stages_ms'].items():
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f | %.6f |'%(k,v['mean'],v['median'],v['stddev'],v['minimum'],v['maximum']))
    (root/'phase-statistics.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('controls','selected_qk','cpu_execute_aggregate_thread_ms','instrumentation','residuals_ms')},indent=2))


if __name__=='__main__':
    main()
