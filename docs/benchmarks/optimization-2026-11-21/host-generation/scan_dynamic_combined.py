import json,pathlib,subprocess,sys
r=pathlib.Path.cwd();root=r/'build/optimization-20261121';d=root/'dynamic';transform=(d/'transformer.txt').read_text().splitlines();spatial=(d/'spatial.txt').read_text().splitlines()
groups={'transformer-blocks':transform+[n for n in spatial if '/blocks.' in n], 'transformer-down':transform+[n for n in spatial if '/downsample/' in n], 'transformer-nohead':transform+[n for n in spatial if n!='output__matmul_converted']}
for name,names in groups.items():
 policy=d/(name+'.txt');policy.write_text('\n'.join(names)+'\n');weights=d/('tiny-vit-weight-'+name+'.mnn');model=d/('tiny-vit-hybrid-'+name+'.mnn')
 subprocess.run([str(root/'weight_only.exe'),str(r/'build/mnn-w8a8-tools/tiny-vit-fp32.mnn'),str(r/'build/mnn-w8a8-tools/tiny-vit-w8a8-kl.mnn'),str(policy),str(weights)],check=True)
 subprocess.run([str(r/'build/create_mixed_quantized_model.exe'),str(weights),str(r/'build/mnn-w8a8-tools/tiny-vit-w8a8-kl.mnn'),str(r/'mnn/app/tinyvit_w8a8_accuracy_allowlist.txt'),str(model)],check=True)
 command=[sys.executable,str(r/'mnn/app/validate_quantized_model.py'),'--reference',str(r/'build/mnn-w8a8-tools/tiny-vit-fp32.mnn'),'--candidate',str(model),'--images',str(r/'build/mnn-w8a8-tools/validation-real-20'),'--manifest',str(root/'spatial/diagnostic-manifest.json'),'--output',str(d/(name+'.json')),'--python-path',str(r/'build/mnn-w8a8-python'),'--memory-low']
 with (d/(name+'.log')).open('w') as log:status=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
 assert status in (0,2)
 j=json.loads((d/(name+'.json')).read_text());print('DYNAMIC',name,j['passed'],j['summary'],flush=True)
