#!/usr/bin/env python3
"""Verify and summarize saved same-input NI83 CPU/FPGA diagnostics.

Optionally check that diagnostic CPU graph outputs match ordinary CPU controls
from the separate natural-image performance experiment. Run on the host after
the board experiment, avoiding analysis work during timed board runs.
"""
import argparse
import collections
import csv
import hashlib
import json
import pathlib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=pathlib.Path)
    parser.add_argument('--natural-results',type=pathlib.Path)
    args=parser.parse_args()
    root=args.results
    metadata=json.loads((root/'metadata.json').read_text())
    records=json.loads((root/'results.json').read_text())
    assert metadata['abi']=='NI83' and len(records)==metadata['inputs']*2
    assert len({(r['input'],r['scope']) for r in records})==len(records)
    natural=None
    if args.natural_results:
        normal_metadata=json.loads((args.natural_results/'metadata.json').read_text())
        assert normal_metadata['input_kind']=='natural images'
        assert normal_metadata['runner_sha256']==metadata['runner_sha256']
        natural={r['input']:r for r in json.loads((args.natural_results/'results.json').read_text())}
    result=dict(metadata=metadata,scopes={},passed=True)
    layer_rows=[]
    for scope,expected in (('transformer',40),('static',51)):
        selected=[r for r in records if r['scope']==scope]
        assert len(selected)==metadata['inputs']
        layers=collections.defaultdict(list)
        preserved=0
        for record in selected:
            native=record['native']
            assert native['abi']=='NI83' and native['operations']==expected
            assert native['pending']==-1 and native['fused_pairs']==0
            assert len(native['verification'])==expected
            assert len({v['name'] for v in native['verification']})==expected
            for value in native['verification']:
                assert 0<=value['over_one']<=value['mismatches']<=value['elements']
                layers[value['name']].append(value)
            assert sum(v['elements'] for v in native['verification'])==record['total_elements']
            assert sum(v['over_one'] for v in native['verification'])==record['over_one']
            assert max(v['maximum'] for v in native['verification'])==record['maximum']
            output=root/('input-%02d'%record['input'])/scope/'output_0.f32'
            if not record['over_one']:
                assert output.stat().st_size==4000
                assert digest(output)==record['output_sha256']
            if natural is not None:
                normal=natural[record['input']]
                assert normal['image_sha256']==record['image_sha256']
                model='tiny-vit-mixed-w8a8-all-'+scope+'.mnn'
                assert normal_metadata['models'][model]==metadata['models'][scope]
                control=normal['runs'][scope+'_cpu_control']
                assert record.get('output_sha256')==control['output_sha256']
                control_output=args.natural_results/('input-%02d'%record['input'])/(scope+'_cpu_control')/'output_0.f32'
                assert digest(control_output)==control['output_sha256']
                preserved+=1
        assert len(layers)==expected and all(len(v)==metadata['inputs'] for v in layers.values())
        elements=sum(r['total_elements'] for r in selected)
        mismatches=sum(r['mismatches'] for r in selected)
        over=sum(r['over_one'] for r in selected)
        absolute=sum(v['sum_abs'] for values in layers.values() for v in values)
        maximum=max(r['maximum'] for r in selected)
        result['scopes'][scope]=dict(images=len(selected),operators_per_image=expected,
            elements=elements,mismatches=mismatches,mismatch_fraction=mismatches/elements,
            over_one=over,maximum_code_error=maximum,mean_absolute_code_error=absolute/elements,
            diagnostic_cpu_outputs_verified_against_controls=preserved,
            passed=over==0 and all(r['returncode']==0 for r in selected))
        result['passed']=result['passed'] and result['scopes'][scope]['passed']
        for name,values in layers.items():
            count=sum(v['elements'] for v in values)
            layer_rows.append(dict(scope=scope,name=name,images=len(values),elements=count,
                mismatches=sum(v['mismatches'] for v in values),over_one=sum(v['over_one'] for v in values),
                maximum=max(v['maximum'] for v in values),
                mean_absolute_code_error=sum(v['sum_abs'] for v in values)/count))
    (root/'numeric-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'numeric-layers.csv').open('w',newline='') as output:
        writer=csv.DictWriter(output,fieldnames=list(layer_rows[0]))
        writer.writeheader();writer.writerows(layer_rows)
    print(json.dumps(result['scopes'],indent=2))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
