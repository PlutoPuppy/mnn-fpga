import pathlib,subprocess,sys,json
r=pathlib.Path.cwd();d=r/'build/optimization-20261121'
for name in ['transformer-blocks','transformer-down']:
 cmd=[sys.executable,str(r/'mnn/app/validate_quantized_model.py'),'--reference',str(r/'build/mnn-w8a8-tools/tiny-vit-fp32.mnn'),'--candidate',str(d/('dynamic/tiny-vit-hybrid-'+name+'.mnn')),'--images',str(d/'fresh-real-200'),'--manifest',str(d/'fresh-manifest.json'),'--output',str(d/(name+'-fresh200.json')),'--python-path',str(r/'build/mnn-w8a8-python'),'--memory-low']
 with (d/(name+'-fresh200.log')).open('w') as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert code in (0,2);j=json.loads((d/(name+'-fresh200.json')).read_text());print('FRESH',name,j['passed'],j['summary'],flush=True)
