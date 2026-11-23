"""Audit extracted board archives without executing models or contacting hardware.

Usage: python audit_evidence.py EXTRACTED_V22 EXTRACTED_V25 [OUTPUT_JSON]
Extract each archive into its own directory first. Python standard library only.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def audit(initial, final):
    roots = {**{p.name: p for p in initial.iterdir() if p.is_dir()},
             **{p.name: p for p in final.iterdir() if p.is_dir()}}
    checked = {}

    def scores(run):
        command = run['command']
        remote = command[command.index('--output-dir') + 1]
        parts = remote.split('/')
        start = next(i for i, name in enumerate(parts) if name in roots)
        path = roots[parts[start]].joinpath(*parts[start + 1:], 'output_0.f32')
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == run['output_sha256'], path
        assert len(data) == 4000, (path, len(data))
        values = struct.unpack('<1000f', data)
        assert all(math.isfinite(v) for v in values), path
        checked[str(path)] = run['output_sha256']
        return values

    random = {}
    for version in ('v22', 'v25'):
        rows = json.loads((roots['optimization-' + version + '-random20'] / 'results.json').read_text())
        assert len(rows) == 20
        random[version] = rows
        for row in rows:
            for name, run in row['runs'].items():
                scores(run)
                if name.endswith('_control'):
                    assert run['output_sha256'] == row['runs'][name.replace('_control', '_profile')]['output_sha256']
            if version == 'v22':
                assert row['runs']['pairs_control']['output_sha256'] == row['runs']['fast_control']['output_sha256']
            else:
                hashes = {row['runs'][m + '_control']['output_sha256'] for m in ('previous', 'no_park', 'optimized')}
                assert len(hashes) == 1
        if version == 'v25':
            for before, after in zip(random['v22'], rows):
                assert before['image_sha256'] == after['image_sha256']
                assert before['runs']['fp32_control']['output_sha256'] == after['runs']['fp32_control']['output_sha256']

    accuracy = {}
    for version in ('v22', 'v25'):
        directory = roots['optimization-' + version + '-accuracy200']
        rows = json.loads((directory / 'results.json').read_text())
        assert len(rows) == 200
        correct_ref = correct_fpga = agreement = 0
        cosines = []
        for row in rows:
            for run in row['runs'].values():
                scores(run)
            ref, fpga = scores(row['runs']['fp32']), scores(row['runs']['fpga'])
            r, f = max(range(1000), key=ref.__getitem__), max(range(1000), key=fpga.__getitem__)
            correct_ref += r == row['label']
            correct_fpga += f == row['label']
            agreement += r == f
            cosine = math.fsum(a*b for a, b in zip(ref, fpga)) / math.sqrt(
                math.fsum(a*a for a in ref) * math.fsum(b*b for b in fpga))
            cosines.append(cosine)
            assert abs(cosine - row['comparison']['cosine_similarity']) < 1e-10
        summary = json.loads((directory / 'summary.json').read_text())
        result = dict(images=200, fp32_correct=correct_ref, fpga_correct=correct_fpga,
                      matching_top1=agreement, mean_cosine=math.fsum(cosines)/200)
        assert correct_ref/200 == summary['reference_accuracy']
        assert correct_fpga/200 == summary['fpga_accuracy']
        assert agreement/200 == summary['top1_agreement']
        assert abs(result['mean_cosine'] - summary['mean_cosine']) < 1e-10
        accuracy[version] = result
        if version == 'v22':
            old = rows
        else:
            assert all(a['image'] == b['image'] and a['runs']['fpga']['output_sha256'] == b['runs']['fpga']['output_sha256']
                       for a, b in zip(old, rows))
    return dict(passed=True, unique_output_files_checked=len(checked), finite_scores_per_file=1000,
                random_control_profile_pairs=200, final_fpga_old_new_identical_random=20,
                final_fpga_old_new_identical_labeled=200, accuracy=accuracy)


if __name__ == '__main__':
    result = audit(Path(sys.argv[1]), Path(sys.argv[2]))
    text = json.dumps(result, indent=2) + '\n'
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text)
    print(text)
