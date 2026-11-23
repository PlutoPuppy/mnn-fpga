#!/usr/bin/env python3
"""Produce additive CPU/FPGA pipeline accounting from the optimization records."""
import argparse
import collections
import csv
import json
import pathlib

from summarize_native_int8 import paired, read_operator_profile, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=pathlib.Path)
    a = p.parse_args()
    root = a.results
    rows = json.loads((root / 'results.json').read_text())
    metadata = json.loads((root / 'metadata.json').read_text())
    assert len(rows) == metadata['inputs'] == 20
    modes = metadata['modes']
    summary = dict(timing={}, comparisons={}, pipeline={}, native_phases={}, accounting={})
    per_image = []
    native_names = {r['name'] for r in rows[0]['runs']['optimized_control']['native']['layers']}
    for mode in modes:
        projection = rows[0]['runs'][mode + '_control'].get('projection', {})
        native_names.update(r['name'] for r in projection.get('operations', []))
    # Transport polling must preserve integer arithmetic bit for bit.
    if 'pairs' in modes and 'fast' in modes:
        assert all(r['runs']['pairs_control']['output_sha256'] == r['runs']['fast_control']['output_sha256'] for r in rows)
    if 'unchained' in modes:
        for row in rows:
            for suffix in ('_control', '_profile'):
                assert (row['runs']['unchained' + suffix]['output_sha256'] ==
                        row['runs']['optimized' + suffix]['output_sha256']), 'DMA chaining changed output bytes'
    optimized_env = metadata['configurations']['optimized'].get('environment', {})
    if 'previous' in modes and optimized_env.get('MNN_MATMUL_SOC_PIPELINE_C') == '1':
        for row in rows:
            for suffix in ('_control', '_profile'):
                assert (row['runs']['previous' + suffix]['output_sha256'] ==
                        row['runs']['optimized' + suffix]['output_sha256']), 'Output staging changed score bytes'
    attention_sources = [r['name'] for r in rows[0]['runs']['optimized_control']['attention']['regions']]
    for mode in modes:
        controls = [r['runs'][mode + '_control'] for r in rows]
        summary['timing'][mode] = {key: stats([r[key] for r in controls])
                                   for key in ('inference_ms', 'process_wall_ms')}
        if mode != 'fp32':
            summary['comparisons'][mode + '_vs_fp32'] = paired(
                [r['runs']['fp32_control']['inference_ms'] for r in rows], [r['inference_ms'] for r in controls])
        groups, driver = [], []
        for row in rows:
            result = row['runs'][mode + '_profile']
            detail = read_operator_profile(root / ('input-%02d' % row['input']) / (mode + '_profile'))
            ops = detail['records']
            positions = {op['name']: i for i, op in enumerate(ops)}
            attention_indices = set()
            for source in attention_sources:
                projection = source.replace('/qkv/', '/proj/')
                target = projection + '/quantize' if projection + '/quantize' in positions else projection
                first, last = positions[source], positions[target]
                assert first < last
                attention_indices.update(range(first + 1, last))
            group = collections.defaultdict(float)
            for i, op in enumerate(ops):
                typ = op['type'].split()[0]
                if op['name'] in native_names:
                    category = 'selected_linear_callbacks'
                elif i in attention_indices:
                    category = 'attention'
                elif typ.startswith('Convolution'):
                    category = 'remaining_cpu_convolutions'
                elif typ == 'UnaryOp':
                    category = 'remaining_gelu'
                elif typ in ('FloatToInt8', 'Int8ToFloat'):
                    category = 'remaining_activation_casts'
                elif typ == 'Raster':
                    category = 'remaining_layout_copies'
                elif typ == 'BinaryOp':
                    category = 'remaining_residual_and_elementwise'
                elif typ in ('LayerNorm', 'Reduction'):
                    category = 'normalization_and_reduction'
                else:
                    category = 'other_operator_callbacks'
                group[category] += op['ms']
            manual_attention = sum(r['compute_ms'] + r['copy_or_verify_ms']
                                   for r in result.get('attention', {}).get('regions', []))
            group['attention'] += manual_attention
            group['framework_outside_callbacks'] = detail['outside_operator_ms'] - manual_attention
            assert group['framework_outside_callbacks'] >= -.05
            assert abs(sum(group.values()) - detail['inference_ms']) < .05
            groups.append(group)
            profile = result['profile']
            phase = profile['offload']['ms']
            if 'native' in result:
                disjoint = {k: phase[k] for k in ('configure', 'upload', 'launch', 'completion_wait', 'download', 'local_feedback',
                                                 'peak_scan', 'quantize', 'input_layout', 'dequantize')}
                disjoint['other_adapter'] = phase['adapter_total'] - sum(disjoint.values())
                assert disjoint['other_adapter'] >= -.05
                disjoint['callback_over_adapter'] = group['selected_linear_callbacks'] - phase['adapter_total']
                assert disjoint['callback_over_adapter'] >= -.05
                driver.append(disjoint)
            record = dict(input=row['input'], mode=mode, profiled_inference_ms=detail['inference_ms'],
                          control_inference_ms=row['runs'][mode + '_control']['inference_ms'])
            record.update(group)
            per_image.append(record)
        keys = sorted({k for group in groups for k in group})
        summary['pipeline'][mode] = {k: stats([g.get(k, 0) for g in groups]) for k in keys}
        summary['accounting'][mode] = dict(profiled_inference_ms=stats([
            r['runs'][mode + '_profile']['profile']['inference_ms'] for r in rows]),
            sum_pipeline_mean_ms=sum(v['mean'] for v in summary['pipeline'][mode].values()))
        if driver:
            summary['native_phases'][mode] = {k: stats([d[k] for d in driver]) for k in driver[0]}
    for ref, mode in [('pairs', 'fast'), ('fast', 'optimized'), ('pairs', 'optimized'), ('cpu_optimized', 'optimized'),
                      ('unchained', 'optimized'), ('baseline', 'unchained'),
                      ('previous', 'optimized'), ('no_park', 'optimized'),
                      ('baseline', 'optimized'), ('baseline', 'poll_batch'), ('optimized', 'poll_batch')]:
        if ref not in modes or mode not in modes:
            continue
        summary['comparisons'][mode + '_vs_' + ref] = paired(
            [r['runs'][ref + '_control']['inference_ms'] for r in rows],
            [r['runs'][mode + '_control']['inference_ms'] for r in rows])
    candidate = min((m for m in modes if metadata['configurations'][m].get('offload', False)),
                    key=lambda m: summary['timing'][m]['inference_ms']['mean'])
    summary['target'] = dict(required_speedup=1.5,
                             best_measured_configuration=candidate,
                             maximum_inference_ms=summary['timing']['fp32']['inference_ms']['mean'] / 1.5,
                             achieved=summary['comparisons'][candidate + '_vs_fp32']['speedup'] >= 1.5)
    summary['target']['remaining_ms'] = (summary['timing'][candidate]['inference_ms']['mean'] -
                                        summary['target']['maximum_inference_ms'])
    (root / 'optimization-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fields = list(dict.fromkeys(k for r in per_image for k in r))
    with (root / 'optimization-pipeline.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_image)
    print(json.dumps(dict(timing=summary['timing'], comparisons=summary['comparisons'], target=summary['target']), indent=2))


if __name__ == '__main__':
    main()
