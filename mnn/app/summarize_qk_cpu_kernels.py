#!/usr/bin/env python3
"""Summarize CPU-only kernel intervals alongside the original DMA phases."""
import argparse
import csv
import json
import pathlib
import statistics as st
from summarize_qk_profile import stats,delta_stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=pathlib.Path)
    p.add_argument('--dma-results',type=pathlib.Path,required=True)
    a=p.parse_args();root=a.directory
    rows=json.loads((root/'results.json').read_text())
    original=json.loads((a.dma_results/'results.json').read_text())
    assert len(rows)==len(original)==20
    for r,o in zip(rows,original):
        assert r['input']==o['input'] and r['image_sha256']==o['image_sha256']
        assert r['runs']['control']['output_sha256']==r['runs']['profile']['output_sha256']==o['runs']['cpu_control']['output_sha256']
        prof=r['runs']['profile']['profile'];off=prof['offload']
        assert off['cpu_execute_calls']==64 and off['cpu_kernel_calls']==320 and off['cpu_interval_overflow']==0
        assert off['cpu_kernel_union_ms']<=off['cpu_execute_union_ms']<=prof['selected_qk_ms']
    values={k:[r['runs']['profile']['profile']['offload'][k] for r in rows] for k in (
            'cpu_execute_ms','cpu_execute_union_ms','cpu_kernel_thread_ms','cpu_kernel_union_ms')}
    values['selected_qk_ms']=[r['runs']['profile']['profile']['selected_qk_ms'] for r in rows]
    compute=[r['runs']['dma_profile']['profile']['offload']['ms']['completion_wait'] for r in original]
    summary=dict(inputs=20,measured_inferences=40,warmup_inferences=2,kernel_calls_per_inference=320,
                 cpu_ms={k:stats(v) for k,v in values.items()},
                 fpga_completion_ms=stats(compute),
                 cpu_kernel_union_over_fpga_completion_ratio=st.mean(values['cpu_kernel_union_ms'])/st.mean(compute),
                 cpu_qk_over_fpga_completion_ratio_original=st.mean(r['runs']['cpu_profile']['profile']['selected_qk_ms'] for r in original)/st.mean(compute),
                 instrumentation_inference=delta_stats([r['runs']['control']['inference_ms'] for r in rows],[r['runs']['profile']['inference_ms'] for r in rows]),
                 qk_wall_without_any_packed_kernel=stats([a-b for a,b in zip(values['selected_qk_ms'],values['cpu_kernel_union_ms'])]))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'cpu-kernel-baseline.csv').open('w',newline='') as stream:
        fields=['input','selected_qk_ms','cpu_execute_union_ms','cpu_kernel_union_ms','cpu_execute_ms','cpu_kernel_thread_ms','cpu_kernel_calls']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            pr=row['runs']['profile']['profile'];off=pr['offload']
            line={k:off[k] for k in fields if k not in ('input','selected_qk_ms')}
            line.update(input=row['input'],selected_qk_ms=pr['selected_qk_ms']);writer.writerow(line)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
