"""Package the validated NI83 DMA runner using the existing licensed deployment."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('build')
BASE = ROOT / 'tinyvit-optimized-90mhz-v25'
NAME = 'tinyvit-dma-optimized-90mhz-v27'
DEST = ROOT / NAME
MODEL = 'tiny-vit-mixed-w8a8-accuracy.mnn'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('configuration', type=Path)
args = parser.parse_args()
configuration = json.loads(args.configuration.read_text())['optimized']
environment = dict(MNN_MATMUL_SOC_DEVMEM='1', MNN_MATMUL_SOC_DMA='1',
                   MNN_MATMUL_SOC_FAST_POLL='1', MNN_MATMUL_SOC_REUSE_A='0',
                   MNN_MATMUL_SOC_LIVE_ROWS='0', MNN_MATMUL_SOC_DMA_POLL_BATCH='0',
                   MNN_MATMUL_SOC_DIRECT_A='0', MNN_MATMUL_SOC_DIRECT_C='0',
                   MNN_MATMUL_SOC_STREAM_STAGE='0')
environment.update(configuration['environment'])
DEST.mkdir(exist_ok=False)
runner = ROOT / 'matmul-soc-deploy' / configuration['runner']
shutil.copyfile(runner, DEST / 'tinyvit_run')
shutil.copyfile(BASE / MODEL, DEST / MODEL)
for notice in sorted(BASE.glob('*.txt')):
    shutil.copyfile(notice, DEST / notice.name)
wrapper = (BASE / 'run_inference.sh').read_text(encoding='utf-8').splitlines()
current = next(i for i,line in enumerate(wrapper) if line.startswith('export '))
wrapper[current:current+1] = ['export '+key+'='+value for key,value in environment.items()]
(DEST/'run_inference.sh').write_bytes(('\n'.join(wrapper)+'\n').encode())
(DEST/'README.md').write_text("""# TinyViT FPGA DMA deployment

The included runner uses the tested transport settings in provenance.json.
Run on DE1-SoC Linux with the matching 90 MHz NI83 FPGA image already
programmed and the reserved DDR window excluded from Linux system RAM.

```sh
sha256sum -c SHA256SUMS
sudo ./run_inference.sh /absolute/path/image.png --output-dir result-01
```

The bundle includes the original validated mixed-precision model, an ARM
Linux executable, dependency licenses and runtime settings. The original
FP32 control model is available separately on the board.
See docs/tinyvit-dma-optimization.md in the source repository for measured
results and the I/O timing breakdown.
""",encoding='utf-8')
provenance=json.loads((BASE/'provenance.json').read_text())
provenance.update(runner_sha256=sha(DEST/'tinyvit_run'), model_sha256=sha(DEST/MODEL),
                  transport_environment=environment, board_runner=runner.name)
(DEST/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
files=sorted(p for p in DEST.rglob('*') if p.is_file())
(DEST/'SHA256SUMS').write_bytes(('\n'.join(sha(p)+'  '+p.relative_to(DEST).as_posix() for p in files)+'\n').encode())
archive=ROOT/(NAME+'.tar.gz')
with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
    with tarfile.open(fileobj=gz, mode='w') as tar:
        for p in files+[DEST/'SHA256SUMS']:
            member=tar.gettarinfo(str(p),arcname=NAME+'/'+p.relative_to(DEST).as_posix())
            member.mtime=0
            member.uid=member.gid=0
            member.uname=member.gname=''
            member.mode=0o755 if p.name in ('tinyvit_run','run_inference.sh') else 0o644
            with p.open('rb') as src:
                tar.addfile(member,src)
print(json.dumps(dict(path=str(archive), size=archive.stat().st_size, sha256=sha(archive)),indent=2))

