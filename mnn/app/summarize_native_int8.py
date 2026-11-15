#!/usr/bin/env python3
"""Summarize NI82 board evidence, verifying paired score/metadata records."""
import argparse
import csv
import hashlib
import json
import pathlib
import random
import statistics as s
import struct
import collections
import math


def stats(values):
    return dict(n=len(values),mean=s.mean(values),median=s.median(values),sd=s.stdev(values) if len(values)>1 else 0,
                minimum=min(values),maximum=max(values))


def paired(cpu,fpga):
    differences=[b-a for a,b in zip(cpu,fpga)]
    rng=random.Random(20261114)
    means=sorted(s.mean(rng.choices(differences,k=len(differences))) for _ in range(10000))
    return dict(delta_ms=stats(differences),delta_percent=100*(s.mean(fpga)/s.mean(cpu)-1),
                speedup=s.mean(cpu)/s.mean(fpga),faster=sum(d<0 for d in differences),
                delta_ms_bootstrap_95=[means[250],means[9749]])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=pathlib.Path);a=p.parse_args()
    root=a.results
    rows=json.loads((root/'results.json').read_text());metadata=json.loads((root/'metadata.json').read_text())
    assert len(rows)==metadata['inputs']
    modes=metadata['modes'];summary=dict(metadata=metadata,configurations={},comparisons={},phases={},scores={})
    operators={m:[] for m in modes}
    for row in rows:
        for mode in modes:
            control=row['runs'][mode+'_control'];profile=row['runs'][mode+'_profile']
            assert control['output_sha256']==profile['output_sha256']
            for kind in ('control','profile'):
                target=root/('input-%02d'%row['input'])/(mode+'_'+kind)
                output=target/'output_0.f32'
                assert output.is_file() and output.stat().st_size==4000
                assert hashlib.sha256(output.read_bytes()).hexdigest()==row['runs'][mode+'_'+kind]['output_sha256']
            if 'native' in control:
                assert control['native']['pending']==-1 and control['jobs']>0
                assert control['jobs']==profile['jobs']==profile['profile']['offload']['jobs']
                assert profile['native']['local_feedback_bytes']==profile['profile']['offload']['local_bytes']
            detail=json.loads((root/('input-%02d'%row['input'])/(mode+'_profile')/'operators.json').read_text())
            operators[mode].append(detail['records'])
    for mode in modes:
        controls=[r['runs'][mode+'_control'] for r in rows]
        profiles=[r['runs'][mode+'_profile']['profile'] for r in rows]
        summary['configurations'][mode]={k:stats([r[k] for r in controls]) for k in ('inference_ms','process_wall_ms','jobs')}
        summary['configurations'][mode]['setup_ms']=stats([r['linear_setup_ms'] for r in profiles])
        summary['configurations'][mode]['selected_operation_ms']=stats([r['selected_operation_ms'] for r in profiles])
        for counter in ('input_bytes','output_bytes','local_bytes','input_descriptors','output_descriptors','local_descriptors'):
            summary['configurations'][mode][counter]=stats([r['offload'].get(counter,0) for r in profiles])
        for counter in ('local_feedback_bytes','hidden_roundtrip_bytes_avoided','fused_pairs','skipped_runtime_operations'):
            if 'native' in controls[0]:summary['configurations'][mode][counter]=stats([r['native'][counter] for r in controls])
        summary['phases'][mode]={key:stats([r['offload']['ms'][key] for r in profiles]) for key in profiles[0]['offload']['ms']}
        if 'cpu' in modes and mode!='cpu':
            summary['comparisons'][mode+'_vs_cpu']=paired([r['runs']['cpu_control']['inference_ms'] for r in rows],
                                                          [r['inference_ms'] for r in controls])
        if 'fp32' in modes and mode!='fp32':
            summary['comparisons'][mode+'_vs_fp32']=paired([r['runs']['fp32_control']['inference_ms'] for r in rows],
                                                           [r['inference_ms'] for r in controls])
    for key in rows[0]['scores']:
        checks=[r['scores'][key] for r in rows]
        summary['scores'][key]=dict(top1_agreement=sum(c['top1_agrees'] for c in checks),
            mean_cosine=s.mean(c['cosine_similarity'] for c in checks),
            mean_relative_l2=s.mean(c['relative_l2'] for c in checks),max_abs=max(c['max_abs_error'] for c in checks))
    # Also compare directly to FP32; agreement with the mixed CPU model alone
    # does not establish agreement with the original model.
    for reference in ('cpu','fp32','all'):
        if reference not in modes:continue
        for mode in modes:
            if mode==reference:continue
            key=mode+'_vs_'+reference
            if key not in summary['comparisons']:
                summary['comparisons'][key]=paired([r['runs'][reference+'_control']['inference_ms'] for r in rows],
                                                  [r['runs'][mode+'_control']['inference_ms'] for r in rows])
            summary['comparisons'][key]['process_wall']=paired(
                [r['runs'][reference+'_control']['process_wall_ms'] for r in rows],
                [r['runs'][mode+'_control']['process_wall_ms'] for r in rows])
            if key in summary['scores']:continue
            checks=[]
            for row in rows:
                folder=root/('input-%02d'%row['input'])
                a=struct.unpack('<1000f',(folder/(reference+'_control')/'output_0.f32').read_bytes())
                b=struct.unpack('<1000f',(folder/(mode+'_control')/'output_0.f32').read_bytes())
                assert all(math.isfinite(v) for v in a+b)
                aa=sum(v*v for v in a);bb=sum(v*v for v in b);error=sum((x-y)**2 for x,y in zip(a,b))
                checks.append(dict(agrees=max(range(1000),key=lambda j:a[j])==max(range(1000),key=lambda j:b[j]),
                    cosine=sum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb),relative_l2=math.sqrt(error/aa),
                    max_abs=max(abs(x-y) for x,y in zip(a,b))))
            summary['scores'][key]=dict(top1_agreement=sum(c['agrees'] for c in checks),
                mean_cosine=s.mean(c['cosine'] for c in checks),mean_relative_l2=s.mean(c['relative_l2'] for c in checks),
                max_abs=max(c['max_abs'] for c in checks))
    layer_rows=[]
    if 'cpu' in modes:
        candidates=[r for r in operators['cpu'][0] if r['type'].startswith('Convolution') and
                    r['inputs'][0]['bits']==8 and r['inputs'][0]['shape'][-2:]==[1,1]]
        for candidate in candidates:
            name=candidate['name'];shape=candidate['inputs'][0]['shape'];line=dict(name=name,M=shape[0],K=shape[1],N=candidate['outputs'][0]['shape'][1])
            for mode in modes:
                values=[sum(op['ms'] for op in record if op['name']==name) for record in operators[mode]]
                line[mode+'_mean_ms']=s.mean(values);line[mode+'_sd_ms']=s.stdev(values) if len(values)>1 else 0
            layer_rows.append(line)
        summary['cpu_candidate_wall_ms']=stats([sum(op['ms'] for op in record if op['name'] in {c['name'] for c in candidates}) for record in operators['cpu']])
        names={c['name'] for c in candidates}
        pair_names=[(name,name.replace('/fc1/','/fc2/')) for name in names if '/fc1/' in name and name.replace('/fc1/','/fc2/') in names]
        summary['operator_categories_ms']={};summary['complete_candidate_region_ms']={};summary['resident_pair_regions_ms']={}
        for mode in modes:
            categories=collections.defaultdict(list);regions=[];pairs=collections.defaultdict(list)
            for record in operators[mode]:
                group=collections.defaultdict(float)
                for op in record:
                    key=op['type'].split()[0]
                    if key.startswith('Convolution'):key+=' INT8' if op['inputs'][0]['bits']==8 else ' FP32'
                    group[key]+=op['ms']
                for key,val in group.items():categories[key].append(val)
                positions={op['name']:i for i,op in enumerate(record)};middle=0
                for first,last in pair_names:
                    start,end=positions[first],positions[last]
                    pairs[first].append(sum(op['ms'] for op in record[start:end+1]))
                    middle+=sum(op['ms'] for op in record[start+1:end])
                regions.append(sum(op['ms'] for op in record if op['name'] in names)+middle)
            summary['operator_categories_ms'][mode]={k:stats(v) for k,v in categories.items()}
            summary['complete_candidate_region_ms'][mode]=stats(regions)
            summary['resident_pair_regions_ms'][mode]={k:stats(v) for k,v in pairs.items()}
    if metadata.get('input_kind')=='natural images':
        labels={'n01440764':0,'n02102040':217,'n02979186':482,'n03000684':491,'n03028079':497,
                'n03394916':566,'n03417042':569,'n03425413':571,'n03445777':574,'n03888257':701}
        accuracy={m:0 for m in modes};known=0
        for row in rows:
            wnid=pathlib.Path(row['image_path']).name.split('-')[0]
            if wnid not in labels:continue
            known+=1
            for mode in modes:
                file=root/('input-%02d'%row['input'])/(mode+'_control')/'output_0.f32'
                logits=struct.unpack('<1000f',file.read_bytes())
                accuracy[mode]+=max(range(1000),key=lambda j:logits[j])==labels[wnid]
        summary['natural_image_top1']=dict(known_labels=known,correct=accuracy,scope='These images only, not full ImageNet accuracy')
    (root/'native-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if layer_rows:
        with (root/'native-layers.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(layer_rows[0]));writer.writeheader();writer.writerows(layer_rows)
    timing_rows=[]
    for row in rows:
        for mode in modes:
            c=row['runs'][mode+'_control'];p=row['runs'][mode+'_profile']['profile']
            line=dict(input=row['input'],mode=mode,inference_ms=c['inference_ms'],process_wall_ms=c['process_wall_ms'],jobs=c['jobs'],setup_ms=p['linear_setup_ms'])
            line.update({'phase_'+k+'_ms':v for k,v in p['offload']['ms'].items()})
            timing_rows.append(line)
    fields=list(dict.fromkeys(k for line in timing_rows for k in line))
    with (root/'native-timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(timing_rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('metadata','phases')},indent=2))


if __name__=='__main__':main()
