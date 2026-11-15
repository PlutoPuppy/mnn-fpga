#!/usr/bin/env python3
"""Select native layers from paired operator profiles, preserving MLP groups.

This is a measured policy for the recorded model/shapes, not a universal cost
model. Keep only groups with >=0.2 ms mean gain and a positive bootstrap bound.
"""
import argparse
import csv
import json
import pathlib
import random
import statistics as s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=pathlib.Path)
    p.add_argument('--minimum-gain-ms',type=float,default=.2);a=p.parse_args();root=a.directory
    rows=json.loads((root/'results.json').read_text());assert len(rows)>=20
    layers=rows[0]['runs']['pairs_control']['native']['layers']
    cpu=[];device=[]
    for row in rows:
        folder=root/('input-%02d'%row['input'])
        cpu.append(json.loads((folder/'cpu_profile/operators.json').read_text())['records'])
        device.append(json.loads((folder/'pairs_profile/operators.json').read_text())['records'])
    used=set();selected=set();decisions=[];rng=random.Random(20261114)
    for layer in layers:
        name=layer['name']
        if name in used:continue
        names=[name]
        if layer['fused_second']>=0:names.append(layers[layer['fused_second']]['name'])
        used.update(names)
        def region(record):
            positions={op['name']:i for i,op in enumerate(record)}
            return sum(op['ms'] for op in record[positions[names[0]]:positions[names[-1]]+1])
        c=[region(record) for record in cpu];d=[region(record) for record in device]
        gains=[x-y for x,y in zip(c,d)]
        boot=sorted(s.mean(rng.choices(gains,k=len(gains))) for _ in range(10000))
        keep=s.mean(gains)>=a.minimum_gain_ms and boot[250]>0
        if keep:selected.update(names)
        decisions.append(dict(first=names[0],last=names[-1],operations=len(names),cpu_ms=s.mean(c),fpga_ms=s.mean(d),
                              mean_gain_ms=s.mean(gains),gain_95_low_ms=boot[250],gain_95_high_ms=boot[9749],keep=keep))
    if not selected:raise RuntimeError('No native layer group passed the performance threshold')
    ordered=[layer['name'] for layer in layers if layer['name'] in selected]
    (root/'native-profitable-allowlist.txt').write_text('# Paired profile policy; >=%.3f ms mean gain, positive 95%% lower bound\n'%a.minimum_gain_ms+'\n'.join(ordered)+'\n')
    with (root/'native-policy.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(decisions[0]));writer.writeheader();writer.writerows(decisions)
    print('Selected',len(ordered),'of',len(layers),'linear operations')
    for d in decisions:print(('KEEP ' if d['keep'] else 'CPU  ')+d['first']+' gain=%.3f ms'%d['mean_gain_ms'])


if __name__=='__main__':main()
