import hashlib,json,pathlib,subprocess,sys
r=pathlib.Path.cwd();root=r/'build/optimization-20261121/dynamic';root.mkdir(exist_ok=False)
base=set(s.strip() for s in (r/'mnn/app/tinyvit_w8a8_accuracy_allowlist.txt').read_text().splitlines() if s.strip() and not s.startswith('#'))
nodes=json.loads((r/'build/native-v4-raw-evidence/benchmark-native-v4-v10-random-20/input-01/fp32_profile/operators.json').read_text())['model_nodes']
remaining=[n['name'] for n in nodes if n['type']=='Convolution' and n['name'] not in base]
linears=(r/'mnn/app/tinyvit_w8a8_all_static_allowlist.txt').read_text().splitlines();linears=set(s.strip() for s in linears if s.strip() and not s.startswith('#'))
groups={'linear':[n for n in remaining if n in linears], 'spatial':[n for n in remaining if n in linears and not n.startswith(('/layers.1/blocks.','/layers.2/blocks.','/layers.3/blocks.'))], 'transformer':[n for n in remaining if n in linears and n.startswith(('/layers.1/blocks.','/layers.2/blocks.','/layers.3/blocks.'))], 'cnn':[n for n in remaining if n not in linears or not n.startswith(('/layers.1/blocks.','/layers.2/blocks.','/layers.3/blocks.'))]}
manifest=r/'build/optimization-20261121/spatial/diagnostic-manifest.json';images=r/'build/mnn-w8a8-tools/validation-real-20'
for name,selected in groups.items():
 policy=root/(name+'.txt');policy.write_text('\n'.join(selected)+'\n');weights=root/('tiny-vit-weight-'+name+'.mnn');model=root/('tiny-vit-hybrid-'+name+'.mnn')
 subprocess.run([str(r/'build/optimization-20261121/weight_only.exe'),str(r/'build/mnn-w8a8-tools/tiny-vit-fp32.mnn'),str(r/'build/mnn-w8a8-tools/tiny-vit-w8a8-kl.mnn'),str(policy),str(weights)],check=True)
 subprocess.run([str(r/'build/create_mixed_quantized_model.exe'),str(weights),str(r/'build/mnn-w8a8-tools/tiny-vit-w8a8-kl.mnn'),str(r/'mnn/app/tinyvit_w8a8_accuracy_allowlist.txt'),str(model)],check=True)
 command=[sys.executable,str(r/'mnn/app/validate_quantized_model.py'),'--reference',str(r/'build/mnn-w8a8-tools/tiny-vit-fp32.mnn'),'--candidate',str(model),'--images',str(images),'--manifest',str(manifest),'--output',str(root/(name+'.json')),'--python-path',str(r/'build/mnn-w8a8-python'),'--memory-low']
 with (root/(name+'.log')).open('w') as log:status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
 assert status in (0,2),(name,status)
 result=json.loads((root/(name+'.json')).read_text());print('DYNAMIC',name,'weight_only',len(selected),result['passed'],result['summary'],flush=True)
