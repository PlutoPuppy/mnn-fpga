#!/usr/bin/env python3
"""Calibrate TinyViT convolution weights/activations with MNN's KL/ADMM tool.

Without --mixed-builder this produces an experimental full-quantization
candidate, which fails accuracy validation for this TinyViT model. Use
--mixed-builder and the checked accuracy allowlist for the repaired model.
Run validate_quantized_model.py on disjoint images before deployment.
"""
import argparse,hashlib,importlib.metadata,json,pathlib,subprocess,sys,time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path)
    p.add_argument('--calibration-dir',required=True,type=pathlib.Path);p.add_argument('--images',type=int,default=128)
    p.add_argument('--feature-method',choices=('KL','ADMM'),default='KL')
    p.add_argument('--python-path',type=pathlib.Path,help='Optional isolated MNN Python installation directory')
    p.add_argument('--mixed-builder',type=pathlib.Path,help='Compiled create_mixed_quantized_model executable')
    p.add_argument('--allowlist',type=pathlib.Path,default=pathlib.Path(__file__).with_name('tinyvit_w8a8_accuracy_allowlist.txt'))
    a=p.parse_args();source=a.model.resolve();dest=a.output.resolve();images=a.calibration_dir.resolve()
    if not source.is_file() or not images.is_dir() or a.images<=0:p.error('Model/calibration directory and positive image count required')
    if source==dest or dest.exists():p.error('Output must be a new path distinct from the input')
    calibrated=pathlib.Path(str(dest)+'.calibrated.mnn') if a.mixed_builder else dest
    if calibrated.exists():p.error('Calibration intermediate already exists; use a new output path')
    if a.mixed_builder and (not a.mixed_builder.is_file() or not a.allowlist.is_file()):
        p.error('Mixed builder executable and allowlist must exist')
    files=sorted(x for x in images.iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png','.ppm'))
    if len(files)<a.images:p.error('Not enough calibration images')
    dest.parent.mkdir(parents=True,exist_ok=True)
    config=dict(format='RGB',mean=[.485*255,.456*255,.406*255],normal=[1/(255*x) for x in (.229,.224,.225)],
                width=224,height=224,path=str(images),used_image_num=a.images,feature_quantize_method=a.feature_method,
                weight_quantize_method='MAX_ABS',quant_bits=8,feature_clamp_value=127,weight_clamp_value=127,debug=False)
    config_file=pathlib.Path(str(dest)+'.calibration.json');config_file.write_text(json.dumps(config,indent=2)+'\n')
    if a.python_path:sys.path.insert(0,str(a.python_path.resolve()))
    # Import MNN first to initialize its runtime. The tested3.6.2 native tools
    # binding takes one argv list; its packaged Python CLI wrapper differs.
    import MNN,_tools
    start=time.perf_counter();_tools.mnnquant(['mnnquant',str(source),str(calibrated),str(config_file)])
    if not calibrated.is_file() or not calibrated.stat().st_size:raise RuntimeError('MNN did not produce a model')
    if a.mixed_builder:
        subprocess.run([str(a.mixed_builder.resolve()),str(source),str(calibrated),str(a.allowlist.resolve()),str(dest)],check=True)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    provenance=dict(mnn_version=importlib.metadata.version('MNN'),source_sha256=sha(source),output_sha256=sha(dest),
                    source_bytes=source.stat().st_size,output_bytes=dest.stat().st_size,elapsed_seconds=time.perf_counter()-start,
                    calibration_config=config,calibration_files=[dict(file=x.name,sha256=sha(x)) for x in files],
                    calibrated_sha256=sha(calibrated),
                    quantization_policy='explicit INT8 convolution islands with FP32 fallback' if a.mixed_builder else 'unvalidated full-quantization candidate',
                    allowlist_sha256=sha(a.allowlist) if a.mixed_builder else None,
                    note='Mixed precision preserves sensitive layers; disjoint accuracy validation is still required')
    pathlib.Path(str(dest)+'.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
