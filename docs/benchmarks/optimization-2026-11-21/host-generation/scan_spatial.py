import hashlib,json,pathlib,subprocess,sys
r=pathlib.Path.cwd();root=r/'build/optimization-20261121/spatial';root.mkdir(exist_ok=False)
base=[s.strip() for s in (r/'mnn/app/tinyvit_w8a8_accuracy_allowlist.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
nodes=json.loads((r/'build/native-v4-raw-evidence/benchmark-native-v4-v10-random-20/input-01/fp32_profile/operators.json').read_text())['model_nodes']
names=[n['name'] for n in nodes if n['type'] in ('Convolution','ConvolutionDepthwise')]
groups={
 'block0':[n for n in names if n.startswith('/layers.0/blocks.0/')],
 'block1':[n for n in names if n.startswith('/layers.0/blocks.1/')],
 'blocks01':[n for n in names if n.startswith('/layers.0/blocks.')],
 'stage0':[n for n in names if n.startswith('/layers.0/')],
 'stem':[n for n in names if n.startswith('/patch_embed/')],
 'down0':[n for n in names if n.startswith('/layers.0/downsample/')],
 'down1':[n for n in names if n.startswith('/layers.1/downsample/')],
 'down2':[n for n in names if n.startswith('/layers.2/downsample/')],
 'all_cnn':[n for n in names if n.startswith('/layers.0/') or '/downsample/' in n or n.startswith('/patch_embed/')],
}
labels={'n01440764':0,'n02102040':217,'n02979186':482,'n03000684':491,'n03028079':497,'n03394916':566,'n03417042':569,'n03425413':571,'n03445777':574,'n03888257':701}
images=r/'build/mnn-w8a8-tools/validation-real-20'
manifest=root/'diagnostic-manifest.json';manifest.write_text(json.dumps(dict(images=[dict(file=p.name,label=labels[p.name.split('-')[0]],sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(images.glob('*.png'))]),indent=2))
for name,extra in groups.items():
 selected=list(dict.fromkeys(base+extra));policy=root/(name+'.txt');policy.write_text('\n'.join(selected)+'\n');model=root/('tiny-vit-spatial-'+name+'.mnn')
 subprocess.run([str(r/'build/create_mixed_quantized_model.exe'),str(r/'build/mnn-w8a8-tools/tiny-vit-fp32.mnn'),str(r/'build/mnn-w8a8-tools/tiny-vit-w8a8-kl.mnn'),str(policy),str(model)],check=True)
 command=[sys.executable,str(r/'mnn/app/validate_quantized_model.py'),'--reference',str(r/'build/mnn-w8a8-tools/tiny-vit-fp32.mnn'),'--candidate',str(model),'--images',str(images),'--manifest',str(manifest),'--output',str(root/(name+'.json')),'--python-path',str(r/'build/mnn-w8a8-python')]
 with (root/(name+'.log')).open('w') as log:status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
 assert status in (0,2),(name,status)
 result=json.loads((root/(name+'.json')).read_text());print('SPATIAL',name,'convs',len(selected),result['passed'],result['summary'],flush=True)
