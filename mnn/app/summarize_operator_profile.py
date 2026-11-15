#!/usr/bin/env python3
"""Rank whole-model CPU operators and derive potential dense GEMM shapes.

Convolution dimensions describe equivalent GEMMs, not observed GEMM calls.
Shape/layout transformations and im2col are explicitly identified as extra work.
"""
import argparse
import collections
import csv
import json
import math
import pathlib
import re
import statistics as st
from summarize_qk_profile import stats,delta_stats


def product(values):
    value=1
    for x in values:value*=x
    return value


def nc_hw(tensor):
    shape=tensor['shape']
    if len(shape)!=4:return None
    return (shape[0],shape[3],shape[1],shape[2]) if tensor['layout']==0 else tuple(shape)


def candidate(record,node,constant_rhs=False):
    result=dict(kind='other',static_weights=False,gemm=None,notes='')
    if not node:return result
    inputs,outputs=record['inputs'],record['outputs']
    if 'conv' in node:
        c=node['conv'];group=c['group'];kh,kw=c['kernel']
        inp=nc_hw(inputs[0]) if inputs else None
        out=nc_hw(outputs[0]) if outputs else None
        if not inp or not out:return result
        batch,ic,ih,iw=inp;ob,oc,oh,ow=out
        assert batch==ob and ic>0 and oc>0 and group>0
        assert not c['input_channels'] or ic==c['input_channels']
        assert not c['output_channels'] or oc==c['output_channels']
        result.update(static_weights=bool(node.get('weight_elements')),gemm=dict(
            batch=group,M=batch*oh*ow,N=oc//group,K=ic//group*kh*kw))
        if group==ic and group==oc:
            result.update(kind='depthwise_convolution',notes='Depthwise; dense systolic mapping has low utilization, needs different dataflow')
        elif group!=1:
            result.update(kind='grouped_convolution',notes='Separate per-group GEMMs plus input layout adaptation')
        elif kh==kw==1 and c['stride']==[1,1]:
            dense_view=False  # Logical strides do not establish NC4HW4 physical order.
            result.update(kind='static_pointwise_gemm',dense_activation_view_candidate=dense_view,
                          notes=('H=W=1 still needs explicit NC4HW4 tensor conversion; needs convolution hook, prepared weights and bias handling' if ih==iw==oh==ow==1 else
                                 'Spatial 1x1 convolution requires NC4HW4/NCHW to dense operand/output adaptation plus bias/fused activation handling'))
        else:
            result.update(kind='spatial_convolution',notes='Equivalent GEMM only; CPU im2col or native convolution/streaming hardware required')
        result['kernel']=c['kernel'];result['stride']=c['stride'];result['group']=group
        result['fused_relu']=c['relu'];result['fused_relu6']=c['relu6']
    elif node['type'] in ('MatMul','BatchMatMul') and len(inputs)>=2 and outputs:
        a,b=inputs[0]['shape'],inputs[1]['shape'];out=outputs[0]['shape']
        if min(len(a),len(b),len(out))<2:return result
        ta,tb=node.get('transpose_a',False),node.get('transpose_b',False)
        m,k=(a[-1],a[-2]) if ta else (a[-2],a[-1])
        bk,n=(b[-1],b[-2]) if tb else (b[-2],b[-1])
        if k!=bk or out[-2:]!=[m,n]:
            result.update(kind='matmul_unresolved',notes='Runtime lowering changed tensor view; inspect model/runtime shapes before selecting GEMM dimensions')
            return result
        result.update(kind='static_weight_matmul' if constant_rhs else 'dynamic_matmul',static_weights=constant_rhs,
                      gemm=dict(batch=product(out[:-2]),M=m,N=n,K=k),
                      notes='Existing dense MatMul hook is a candidate; verify transposes, broadcasting and runtime type')
    if result['gemm']:
        g=result['gemm'];result['macs']=g['batch']*g['M']*g['N']*g['K']
        result['current_host_jobs']=g['batch']*math.ceil(g['M']/256)*math.ceil(g['N']/256)*math.ceil(g['K']/64)
        result['logical_input_elements']=g['batch']*(g['M']*g['K']+g['K']*g['N'])
        result['logical_output_elements']=g['batch']*g['M']*g['N']
        result['reduction_panels']=math.ceil(g['K']/64)
        result['current_panel_output_elements']=result['logical_output_elements']*result['reduction_panels']
        result['int8_weight_bytes']=node.get('weight_elements',g['K']*g['N'] if constant_rhs else 0)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=pathlib.Path);a=p.parse_args();root=a.directory
    rows=json.loads((root/'results.json').read_text());meta=json.loads((root/'metadata.json').read_text())
    catalog=json.loads((root/'operator-catalog.json').read_text());records=catalog['records']
    assert len(rows)==meta['inputs']==20 and len({r['image_sha256'] for r in rows})==20
    assert sum(r['order'][0]=='control' for r in rows)==10
    assert all(r['runs']['profile']['operator_count']==len(records) for r in rows)
    assert all(r['runs']['control']['output_sha256']==r['runs']['profile']['output_sha256'] for r in rows)
    assert all(run['exit_code']==run['fpga_jobs']==0 for r in rows for run in r['runs'].values())
    model={n['name']:n for n in catalog['model_nodes']}
    producers={index:n for n in catalog['model_nodes'] for index in n['outputs']}
    constant_cache={}
    def constant(index,seen=None):
        if index in constant_cache:return constant_cache[index]
        seen=set() if seen is None else seen
        if index in seen:return False
        seen=seen|{index};node=producers.get(index)
        result=False
        if node:
            if node['type'] in ('Const','TrainableParam'):result=True
            elif node['type'] not in ('Input','RandomUniform','While') and node['inputs']:
                result=all(constant(i,seen) for i in node['inputs'])
        constant_cache[index]=result
        return result
    control=[r['runs']['control']['inference_ms'] for r in rows]
    profile=[r['runs']['profile']['exact_inference_ms'] for r in rows]
    full=st.mean(profile)
    ranking=[]
    for i,rec in enumerate(records):
        values=[r['runs']['profile']['operator_ms'][i] for r in rows]
        node=model.get(rec['name'])
        rhs=bool(node and len(node['inputs'])>=2 and constant(node['inputs'][1]))
        info=candidate(rec,node,rhs)
        ranking.append(dict(index=i,name=rec['name'],runtime_type=rec['type'],model_type=node['type'] if node else None,
                            calls_per_inference=1,profile_samples=len(rows),
                            ms=stats(values),share_profiled_inference_pct=100*st.mean(values)/full,
                            inputs=rec['inputs'],outputs=rec['outputs'],mnn_estimated_mflops=rec['mnn_estimated_mflops'],**info))
    ranking.sort(key=lambda r:r['ms']['mean'],reverse=True)
    by_type=collections.defaultdict(list)
    for op in ranking:by_type[op['runtime_type']].append(op['index'])
    type_ranking=[]
    for name,indices in by_type.items():
        values=[sum(r['runs']['profile']['operator_ms'][i] for i in indices) for r in rows]
        type_ranking.append(dict(type=name,commands_per_inference=len(indices),ms=stats(values),share_pct=100*st.mean(values)/full))
    type_ranking.sort(key=lambda r:r['ms']['mean'],reverse=True)
    by_name=collections.defaultdict(list)
    for op in ranking:by_name[op['name']].append(op['index'])
    grouped=[]
    for name,indices in by_name.items():
        values=[sum(r['runs']['profile']['operator_ms'][i] for i in indices) for r in rows]
        grouped.append(dict(name=name,runtime_commands=len(indices),ms=stats(values),share_pct=100*st.mean(values)/full))
    grouped.sort(key=lambda r:r['ms']['mean'],reverse=True)
    regions=collections.defaultdict(set)
    for op in ranking:
        stage=re.match(r'(/layers\.\d+)(?:/blocks\.\d+)?',op['name'])
        if stage:regions[stage.group(1)].add(op['index'])
        mlp=re.match(r'(/layers\.\d+/blocks\.\d+/mlp)/',op['name'])
        if mlp:regions[mlp.group(1)].add(op['index'])
    region_ranking=[]
    for name,indices in regions.items():
        values=[sum(r['runs']['profile']['operator_ms'][i] for i in indices) for r in rows]
        region_ranking.append(dict(name=name,commands=len(indices),ms=stats(values),share_pct=100*st.mean(values)/full))
    region_ranking.sort(key=lambda r:r['ms']['mean'],reverse=True)
    families=collections.defaultdict(list)
    for op in ranking:
        if op['gemm']:
            g=op['gemm'];families[(op['kind'],g['batch'],g['M'],g['N'],g['K'])].append(op['index'])
    shape_families=[]
    for key,indices in families.items():
        values=[sum(row['runs']['profile']['operator_ms'][i] for i in indices) for row in rows]
        shape_families.append(dict(kind=key[0],gemm=dict(zip(('batch','M','N','K'),key[1:])),
                                   calls_per_inference=len(indices),ms=stats(values),share_pct=100*st.mean(values)/full))
    shape_families.sort(key=lambda r:r['ms']['mean'],reverse=True)
    summary=dict(inputs=20,measured_cpu_inferences=40,operator_count=len(records),unique_names=len(by_name),
                 control_inference_ms=stats(control),profiled_inference_ms=stats(profile),
                 profiling_comparison=delta_stats(control,profile),
                 sum_operator_ms=stats([r['runs']['profile']['sum_operator_ms'] for r in rows]),
                 outside_operator_ms=stats([r['runs']['profile']['outside_operator_ms'] for r in rows]),
                 operator_coverage_pct=100*st.mean(r['runs']['profile']['sum_operator_ms'] for r in rows)/full,
                 types=type_ranking,grouped_by_name=grouped,
                 regions=region_ranking,
                 shape_families=shape_families,
                 candidate_count=dict(collections.Counter(r['kind'] for r in ranking)),
                 current_first_qk=[r for r in ranking if r['name']=='/layers.1/blocks.0/attn/MatMul_output_0'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'operator-ranking.json').write_text(json.dumps(ranking,indent=2)+'\n')
    fields=['rank','name','runtime_type','model_type','kind','calls_per_inference','profile_samples','mean_ms','median_ms','stddev_ms','min_ms','max_ms','share_pct',
            'static_weights','batch','M','N','K','macs','current_host_jobs','int8_weight_bytes','input_shapes','output_shapes','notes']
    with (root/'operator-ranking.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for rank,op in enumerate(ranking,1):
            g=op['gemm'] or {}
            line=dict(rank=rank,name=op['name'],runtime_type=op['runtime_type'],model_type=op['model_type'],kind=op['kind'],
                      calls_per_inference=op['calls_per_inference'],profile_samples=op['profile_samples'],
                      mean_ms=op['ms']['mean'],median_ms=op['ms']['median'],stddev_ms=op['ms']['stddev'],
                      min_ms=op['ms']['minimum'],max_ms=op['ms']['maximum'],share_pct=op['share_profiled_inference_pct'],
                      static_weights=op['static_weights'],input_shapes=json.dumps([t['shape'] for t in op['inputs']]),
                      output_shapes=json.dumps([t['shape'] for t in op['outputs']]),notes=op['notes'])
            line.update({k:g.get(k) for k in ('batch','M','N','K')})
            line.update({k:op.get(k) for k in ('macs','current_host_jobs','int8_weight_bytes')})
            writer.writerow(line)
    candidates=[r for r in ranking if r['kind'] in ('static_pointwise_gemm','static_weight_matmul','dynamic_matmul','spatial_convolution')]
    (root/'offload-candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('control_inference_ms','profiled_inference_ms','operator_count','operator_coverage_pct','types','candidate_count')},indent=2))
    print('TOP 20:')
    for op in ranking[:20]:print('%.3f ms %.2f%% %s %s %s'%(op['ms']['mean'],op['share_profiled_inference_pct'],op['kind'],op['name'],op['gemm']))


if __name__=='__main__':
    main()
