#!/usr/bin/env python3
"""Validate the frozen NI83 optimization against FP32 on labeled RGB images.

Runs on HPS using only Python's standard library. Latency here is diagnostic;
use benchmark_native_int8.py for the separate balanced timing experiment.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import time

from benchmark_board_random_qk import MEAN, STD, compare, cpu_state, scores, sha

TRANSPORT_OPTIONS = {
    'reuse-a': 'MNN_MATMUL_SOC_REUSE_A',
    'live-rows': 'MNN_MATMUL_SOC_LIVE_ROWS',
    'batch-poll': 'MNN_MATMUL_SOC_DMA_POLL_BATCH',
    'dma-chain': 'MNN_MATMUL_SOC_DMA_CHAIN',
    'pipeline-c': 'MNN_MATMUL_SOC_PIPELINE_C',
    'direct-a': 'MNN_MATMUL_SOC_DIRECT_A',
    'direct-c': 'MNN_MATMUL_SOC_DIRECT_C',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board-dir', default='/home/ubuntu/tinyvit')
    p.add_argument('--runner', required=True)
    p.add_argument('--images', required=True, type=pathlib.Path)
    p.add_argument('--manifest', required=True, type=pathlib.Path)
    p.add_argument('--output-dir', required=True, type=pathlib.Path)
    p.add_argument('--bitstream-sha256', required=True)
    p.add_argument('--experiment-date', required=True)
    p.add_argument('--reference-results', type=pathlib.Path,
                   help='Reuse completed FP32 outputs with identical model, manifest and preprocessing')
    p.add_argument('--resume', action='store_true', help='Continue a verified partial run')
    p.add_argument('--park-fpga-workers', action='store_true')
    p.add_argument('--projection-ranges', type=pathlib.Path,
                   help='Calibrated FP32-boundary FPGA projection selection')
    p.add_argument('--transport', action='append', choices=TRANSPORT_OPTIONS, default=[])
    p.add_argument('--require-identical-fpga', action='store_true',
                   help='Require byte-identical FPGA scores to the reference experiment')
    a = p.parse_args()
    if os.geteuid() != 0 or not re.fullmatch('[0-9a-f]{64}', a.bitstream_sha256):
        p.error('Root and the programmed SOF hash are required')
    if a.require_identical_fpga and not a.reference_results:
        p.error('--require-identical-fpga requires --reference-results')
    transport_env = {TRANSPORT_OPTIONS[name]: '1' for name in a.transport}
    board = pathlib.Path(a.board_dir).resolve()
    projection_names = set()
    if a.projection_ranges:
        a.projection_ranges = a.projection_ranges.resolve()
        projection_names = {s.split()[0] for s in a.projection_ranges.read_text().splitlines()
                            if s.strip() and not s.lstrip().startswith('#')}
        if not projection_names:
            p.error('Empty projection range file')
    out = pathlib.Path(os.path.abspath(str(a.output_dir)))
    if not a.resume:
        out.mkdir(parents=True, exist_ok=False)
    images = json.loads(a.manifest.read_text())['images']
    if len({x['file'] for x in images}) != len(images) or not images:
        raise ValueError('Invalid image manifest')
    for row in images:
        if pathlib.Path(row['file']).name != row['file'] or not 0 <= row['label'] < 1000:
            raise ValueError('Invalid file or ImageNet label')
        if sha(a.images / row['file']) != row['sha256']:
            raise ValueError('Image hash mismatch')
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('MNN_FPGA', 'MNN_MATMUL_SOC', 'MNN_INT8_TRACE')):
            env.pop(key)
    metadata = dict(date=a.experiment_date, images=len(images), threads=2,
                    runner_sha256=sha(board / a.runner), bitstream_sha256=a.bitstream_sha256,
                    models={n: sha(board / n) for n in ('tiny-vit.mnn', 'tiny-vit-mixed-w8a8-accuracy.mnn')},
                    manifest_sha256=sha(a.manifest), cpu_before=cpu_state(),
                    limitations='Imagenette ten-class subset; does not establish full ImageNet accuracy',
                    thresholds=dict(min_agreement=.95, min_cosine=.995, max_accuracy_drop=.01))
    metadata.update(park_fpga_workers=a.park_fpga_workers, transport_environment=transport_env,
                    require_identical_fpga=a.require_identical_fpga)
    if a.projection_ranges:
        metadata['projection_ranges'] = {'sha256':sha(a.projection_ranges),'operations':sorted(projection_names)}
    references = None
    if a.reference_results:
        ref_meta = json.loads((a.reference_results / 'metadata.json').read_text())
        references = json.loads((a.reference_results / 'results.json').read_text())
        assert len(references) == len(images)
        assert ref_meta['manifest_sha256'] == metadata['manifest_sha256']
        assert ref_meta['models']['tiny-vit.mnn'] == metadata['models']['tiny-vit.mnn']
        assert ref_meta['threads'] == 2
        metadata.update(reference_results=str(a.reference_results),
                        reference_runner_sha256=ref_meta['runner_sha256'])
    rows = []
    if a.resume:
        recorded = json.loads((out / 'metadata.json').read_text())
        for key in metadata.keys() - {'cpu_before'}:
            if metadata[key] != recorded.get(key):
                raise ValueError('Cannot resume with changed metadata: ' + key)
        metadata = recorded
        rows = json.loads((out / 'results.json').read_text())
        if len(rows) >= len(images):
            raise ValueError('The accuracy run has already completed')
        for i, row in enumerate(rows):
            if (row['image'], row['label']) != (images[i]['file'], images[i]['label']):
                raise ValueError('Saved image order differs from the manifest')
            for run in row['runs'].values():
                command = run['command']
                destination = pathlib.Path(command[command.index('--output-dir') + 1])
                if sha(destination / 'output_0.f32') != run['output_sha256']:
                    raise ValueError('Saved score hash mismatch in ' + str(destination))
        unfinished = out / ('image-%03d' % len(rows))
        if unfinished.exists():
            count = len(list(out.glob(unfinished.name + '-interrupted-*')))
            unfinished.rename(out / (unfinished.name + '-interrupted-%02d' % count))
        metadata.setdefault('resume_checkpoints', []).append(len(rows))
        print('RESUMING', len(rows), '/', len(images), flush=True)
    else:
        (out / 'results.json').write_text('[]\n')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    def execute(mode, image, destination):
        destination.mkdir(parents=True, exist_ok=False)
        model = 'tiny-vit-mixed-w8a8-accuracy.mnn' if mode == 'fpga' else 'tiny-vit.mnn'
        command = [str(board / a.runner), '--model', str(board / model), '--height', '224', '--width', '224',
                   '--image', str(image), '--mean', MEAN, '--std', STD, '--threads', '2', '--runs', '1',
                   '--output-dir', str(destination)]
        active = dict(env)
        if mode in ('fpga', 'cpu_optimized'):
            command += ['--fused-attention', '--gelu-neon']
        if mode == 'fpga':
            command += ['--fpga-native-int8', '--fpga-native-pairs', '--require-fpga']
            if a.park_fpga_workers:
                command += ['--park-fpga-workers']
            if a.projection_ranges:
                command += ['--native-projections',str(a.projection_ranges)]
            active.update(MNN_MATMUL_SOC_DEVMEM='1', MNN_MATMUL_SOC_DMA='1', MNN_MATMUL_SOC_FAST_POLL='1')
            active.update(transport_env)
        start = time.perf_counter()
        with (destination / 'run.log').open('w') as log:
            status = subprocess.call(command, cwd=str(board), env=active, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=60)
        wall = (time.perf_counter() - start) * 1000
        text = (destination / 'run.log').read_text()
        if status:
            raise RuntimeError(str(destination) + ': ' + text[-2000:])
        timings = re.findall(r'^Inference \d+: ([\d.e+-]+) ms$', text, re.M)
        jobs = re.findall(r'^FPGA submissions: (\d+)$', text, re.M)
        assert len(timings) == len(jobs) == 1
        result = dict(inference_ms=float(timings[0]), process_wall_ms=wall, command=command,
                      jobs=int(jobs[0]), output_sha256=sha(destination / 'output_0.f32'))
        for label in ('NATIVE', 'ATTENTION', 'GELU', 'PARK', 'PROJECTION'):
            details = re.findall('^' + label + '_JSON (.+)$', text, re.M)
            if details:
                result[label.lower()] = json.loads(details[0])
        if mode == 'fpga':
            if projection_names:
                assert result['jobs'] > 296
                ops=result['projection']['operations']
                assert {op['name'] for op in ops}==projection_names and all(op['calls']==1 for op in ops)
            else:
                assert result['jobs'] == 296
            assert result['native']['operations'] == 30 and result['native']['fused_pairs'] == 4
            assert result['native']['pending'] == -1 and result['native']['abi'] == 'NI83'
            if a.park_fpga_workers:
                assert result['park']['calls'] == 26 and result['park']['pending'] == 0
        else:
            assert result['jobs'] == 0
        if mode in ('fpga', 'cpu_optimized'):
            assert len(result['attention']['regions']) == 10 and result['attention']['pending'] == -1
            assert all(r['calls'] == 1 for r in result['attention']['regions'])
            assert sum(r['calls'] for r in result['gelu']['layers']) == (19 if mode == 'fpga' else 23)
        return result, scores(destination / 'output_0.f32')

    for i in range(len(rows), len(images)):
        entry = images[i]
        modes = ['fp32', 'fpga'] if i % 2 == 0 else ['fpga', 'fp32']
        # Check CPU kernel substitutions separately on twenty distributed images.
        if i % 10 == 0:
            modes.append('cpu_optimized')
        runs, outputs = {}, {}
        for mode in modes:
            if mode == 'fp32' and references is not None:
                assert references[i]['image'] == entry['file'] and references[i]['label'] == entry['label']
                reference = references[i]['runs']['fp32']
                path = a.reference_results / ('image-%03d' % i) / 'fp32' / 'output_0.f32'
                assert sha(path) == reference['output_sha256']
                command = reference['command']
                for flag, value in [('--height','224'),('--width','224'),('--threads','2'),
                                    ('--mean',MEAN),('--std',STD)]:
                    assert command[command.index(flag)+1] == value
                assert not any(flag in command for flag in
                               ('--fpga-native-int8','--fused-attention','--gelu-neon','--memory-low'))
                runs[mode], outputs[mode] = reference, scores(path)
                continue
            runs[mode], outputs[mode] = execute(mode, a.images / entry['file'], out / ('image-%03d' % i) / mode)
        ref = max(range(1000), key=lambda k: outputs['fp32'][k])
        candidate = max(range(1000), key=lambda k: outputs['fpga'][k])
        row = dict(image=entry['file'], label=entry['label'], reference_top1=ref, candidate_top1=candidate,
                   reference_correct=ref == entry['label'], candidate_correct=candidate == entry['label'],
                   comparison=compare(outputs['fp32'], outputs['fpga']), runs=runs)
        if 'cpu_optimized' in outputs:
            row['cpu_kernel_comparison'] = compare(outputs['fp32'], outputs['cpu_optimized'])
        if references is not None:
            row['identical_fpga_output_to_reference_experiment'] = (
                runs['fpga']['output_sha256'] == references[i]['runs']['fpga']['output_sha256'])
        rows.append(row)
        checkpoint = out / 'results.next.json'
        checkpoint.write_text(json.dumps(rows, indent=2))
        checkpoint.replace(out / 'results.json')
        if (i + 1) % 10 == 0:
            os.sync()
            print('VALIDATED', i + 1, '/', len(images), flush=True)
    count = float(len(rows))
    summary = dict(images=len(rows), reference_accuracy=sum(r['reference_correct'] for r in rows) / count,
                   fpga_accuracy=sum(r['candidate_correct'] for r in rows) / count,
                   top1_agreement=sum(r['comparison']['top1_agrees'] for r in rows) / count,
                   mean_cosine=sum(r['comparison']['cosine_similarity'] for r in rows) / count)
    summary['accuracy_drop'] = summary['reference_accuracy'] - summary['fpga_accuracy']
    if references is not None:
        summary['identical_fpga_outputs_to_reference_experiment'] = sum(
            r['identical_fpga_output_to_reference_experiment'] for r in rows)
    summary['passed'] = (summary['top1_agreement'] >= .95 and summary['mean_cosine'] >= .995 and
                         summary['accuracy_drop'] <= .01 + 1e-12)
    if a.require_identical_fpga:
        summary['passed'] &= summary['identical_fpga_outputs_to_reference_experiment'] == len(rows)
    metadata['cpu_after'] = cpu_state()
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', json.dumps(summary), flush=True)
    if not summary['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
