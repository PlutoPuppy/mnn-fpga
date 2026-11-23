import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
p=argparse.ArgumentParser()
p.add_argument('--bitstream',required=True)
p.add_argument('--date',required=True)
a=p.parse_args()
root=Path('/home/ubuntu/tinyvit')
out=root/'dma-final-random20'
rows=json.loads((out/'results.json').read_text())
assert len(rows)==20
meta=json.loads((out/'metadata.json').read_text())
assert meta['bitstream_sha256']==a.bitstream
prior=json.loads((root/'optimization-v25-random20'/'results.json').read_text())
for old,row in zip(prior,rows):
    assert old['image_sha256']==row['image_sha256']
    runs=row['runs']
    assert old['runs']['fp32_control']['output_sha256']==runs['fp32_control']['output_sha256']
    old_fpga=old['runs']['no_park_control']['output_sha256']
    for mode in ('baseline','optimized','poll_batch'):
        assert runs[mode+'_control']['output_sha256']==old_fpga, (row['input'],mode)
        assert runs[mode+'_profile']['output_sha256']==old_fpga, (row['input'],mode)
        assert runs[mode+'_profile']['jobs']==296
means={m:statistics.mean(r['runs'][m+'_control']['inference_ms'] for r in rows)
       for m in ('optimized','poll_batch')}
selected=min(means,key=means.get)
print('DMA_SELECTED',selected,means,flush=True)
settings=list(('reuse-a','live-rows'))
if selected=='poll_batch':
    settings.append('batch-poll')
selection=dict(mode=selected,inference_ms=means,settings=settings,exact_images=20)
(root/'dma-final-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
cmd=[sys.executable,'benchmark_dma_accuracy_v27.py','--runner','tinyvit_native_v27',
     '--images',str(root/'optimization-fresh200'),'--manifest',str(root/'optimization-fresh200/manifest.json'),
     '--output-dir',str(root/'dma-final-accuracy200'),'--experiment-date',a.date,'--bitstream-sha256',a.bitstream,
     '--reference-results',str(root/'optimization-v22-accuracy200'),'--require-identical-fpga']
for setting in settings:
    cmd+=['--transport',setting]
subprocess.run(cmd,cwd=str(root),check=True)
