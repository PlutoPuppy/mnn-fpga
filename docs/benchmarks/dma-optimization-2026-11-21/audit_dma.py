"""Cross-check raw v27 score files and historical same-input references."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

root = Path(sys.argv[1])
evidence = Path(sys.argv[2])
old = Path(sys.argv[3])
timing = root / 'raw-timing' / 'dma-final-random20'
accuracy = root / 'raw-accuracy' / 'dma-final-accuracy200'
v25 = json.loads((old / 'final-random20' / 'results.json').read_text())
v25_accuracy = json.loads((old / 'final-accuracy200' / 'results.json').read_text())

def scores(path, expected):
    data = path.read_bytes()
    assert len(data) == 4000
    assert hashlib.sha256(data).hexdigest() == expected, path
    values = struct.unpack('<1000f', data)
    assert all(math.isfinite(x) for x in values), path
    return values

timed = json.loads((timing / 'results.json').read_text())
assert len(timed) == len(v25) == 20
for row, before in zip(timed, v25):
    assert row['image_sha256'] == before['image_sha256']
    for name, run in row['runs'].items():
        reference_mode = 'fp32_control' if name.startswith('fp32') else 'no_park_control'
        expected = before['runs'][reference_mode]['output_sha256']
        assert run['output_sha256'] == expected
        scores(timing / ('input-%02d' % row['input']) / name / 'output_0.f32', expected)
data = json.loads((accuracy / 'results.json').read_text())
assert len(data) == len(v25_accuracy) == 200
ncorrect = 0
agreement = 0
new_accuracy_outputs = 0
for i, (row, before) in enumerate(zip(data, v25_accuracy)):
    assert (row['image'], row['label']) == (before['image'], before['label'])
    for name, run in row['runs'].items():
        if name == 'fp32':
            assert run['output_sha256'] == before['runs']['fp32']['output_sha256']
            continue
        vals = scores(accuracy / ('image-%03d' % i) / name / 'output_0.f32', run['output_sha256'])
        new_accuracy_outputs += 1
        if name == 'fpga':
            assert run['output_sha256'] == before['runs']['fpga']['output_sha256']
            top = max(range(len(vals)), key=vals.__getitem__)
            assert top == row['candidate_top1']
            ncorrect += top == row['label']
            agreement += top == before['reference_top1']
summary = json.loads((accuracy / 'summary.json').read_text())
assert summary['passed'] and summary['fpga_accuracy'] == ncorrect/200
assert summary['top1_agreement'] == agreement/200
audit = dict(passed=True, raw_timing_outputs_verified=160, new_accuracy_outputs_verified=new_accuracy_outputs,
             random_inputs_identical_to_v25=20, labeled_fpga_outputs_identical_to_v25=200,
             fpga_top1_correct=ncorrect, fp32_top1_correct=sum(r['reference_correct'] for r in data),
             matching_fp32_top1=agreement, mean_cosine=summary['mean_cosine'])
(evidence / 'offline-audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
print(json.dumps(audit, indent=2))
