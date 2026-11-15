#!/usr/bin/env python3
"""Prepare128 Imagenette train images and20 disjoint validation images.

Archive: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
Ten classes are useful for an initial experiment, not complete ImageNet coverage.
"""
import argparse,hashlib,io,json,pathlib,random,tarfile
import PIL
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True,type=pathlib.Path)
    p.add_argument('--output-dir',required=True,type=pathlib.Path);p.add_argument('--seed',type=int,default=20261112)
    a=p.parse_args();root=a.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    cal=root/'calibration-128';valid=root/'validation-real-20';cal.mkdir(exist_ok=False);valid.mkdir(exist_ok=False)
    records=[];rng=random.Random(a.seed)
    with tarfile.open(a.archive) as tar:
        files=[m for m in tar if m.isfile() and m.name.lower().endswith(('.jpg','.jpeg'))]
        classes=sorted({pathlib.PurePosixPath(m.name).parts[2] for m in files if '/train/' in m.name})
        if len(classes)!=10:raise ValueError('Expected Imagenette archive with10 training classes')
        for i,cls in enumerate(classes):
            for split,count,target in [('train',13 if i<8 else 12,cal),('val',2,valid)]:
                candidates=sorted([m for m in files if '/'+split+'/'+cls+'/' in m.name],key=lambda m:m.name);rng.shuffle(candidates)
                for j,m in enumerate(candidates[:count]):
                    image=Image.open(io.BytesIO(tar.extractfile(m).read())).convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
                    dest=target/('%s-%02d.png'%(cls,j));image.save(dest)
                    records.append(dict(split=split,source=m.name,file=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    assert len(list(cal.iterdir()))==128 and len(list(valid.iterdir()))==20
    manifest=dict(source='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
        archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),selection_seed=a.seed,pillow_version=PIL.__version__,
        limitation='Only10 ImageNet classes; preliminary calibration, not full ImageNet accuracy validation',
        resize='RGB, PIL bilinear direct224x224; no crop',images=records)
    (root/'calibration-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared128 calibration images and20 disjoint real validation images in',root)

if __name__=='__main__':main()
