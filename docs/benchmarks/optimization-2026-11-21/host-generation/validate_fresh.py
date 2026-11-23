import pathlib,subprocess,sys,json
r=pathlib.Path.cwd();out=r/'build/optimization-20261121'
for name,model,memory in [('validated',r/'build/mnn-w8a8-tools/tiny-vit-mixed-w8a8-accuracy.mnn',False),('dynamic-transformer',out/'dynamic/tiny-vit-hybrid-transformer.mnn',True),('dynamic-spatial',out/'dynamic/tiny-vit-hybrid-spatial.mnn',True)]:
 command=[sys.executable,str(r/'mnn/app/validate_quantized_model.py'),'--reference',str(r/'build/mnn-w8a8-tools/tiny-vit-fp32.mnn'),'--candidate',str(model),'--images',str(out/'fresh-real-200'),'--manifest',str(out/'fresh-manifest.json'),'--output',str(out/(name+'-fresh200.json')),'--python-path',str(r/'build/mnn-w8a8-python')]+(['--memory-low'] if memory else [])
 with (out/(name+'-fresh200.log')).open('w') as log: status=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
 assert status in (0,2),(name,status)
 result=json.loads((out/(name+'-fresh200.json')).read_text());print('FRESH',name,result['passed'],result['summary'],flush=True)
