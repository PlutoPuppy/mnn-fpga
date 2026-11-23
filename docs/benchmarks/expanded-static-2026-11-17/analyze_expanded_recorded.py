import collections,csv,importlib.util,json,pathlib,statistics
base=pathlib.Path('build/expanded-static-20261117/board-evidence')
spec=importlib.util.spec_from_file_location('native_summary','mnn/app/summarize_native_int8.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
output={}
for kind in ('random','natural'):
 root=base/('benchmark-native-ni83-v12-'+kind+'-20');summary=json.loads((root/'native-summary.json').read_text());rows=json.loads((root/'results.json').read_text());modes=summary['metadata']['modes']
 comparisons={}
 for ref,test in [('pairs','transformer_fpga'),('pairs','static_fpga'),('transformer_fpga','static_fpga')]+([('static_unfused','static_fpga')] if kind=='random' else []):
  comparisons[test+'_vs_'+ref]=module.paired([r['runs'][ref+'_control']['inference_ms'] for r in rows],[r['runs'][test+'_control']['inference_ms'] for r in rows])
 names={v['name'] for v in rows[0]['runs']['static_fpga_control']['native']['layers']}
 pipeline={}
 for mode in modes:
  per_image=[]
  for row in rows:
   detail=json.loads((root/('input-%02d'%row['input'])/(mode+'_profile')/'operators.json').read_text());group=collections.defaultdict(float)
   for op in detail['records']:
    t=op['type'].split()[0]
    if op['name'] in names:key='Static linear callbacks (all 51)'
    elif t.startswith('Convolution'):key='Other convolutions'
    elif t in ('FloatToInt8','Int8ToFloat','Raster','UnaryOp','While','Softmax'):key=t
    else:key='Other CPU/operator scopes'
    group[key]+=op['ms']
   group['Outside operator callbacks']=detail['outside_operator_ms']
   assert abs(sum(group.values())-detail['inference_ms'])<0.05
   per_image.append(group)
  keys=set(k for g in per_image for k in g)
  pipeline[mode]={k:module.stats([g.get(k,0) for g in per_image]) for k in sorted(keys)}
 layer_rows=list(csv.DictReader((root/'native-layers.csv').open()))
 extras=[r for r in layer_rows if '/attn/' not in r['name'] and '/mlp/' not in r['name']]
 extra_values=[]
 for r in extras:
  extra_values.append(dict(name=r['name'],M=int(r['M']),K=int(r['K']),N=int(r['N']),
   transformer_cpu_scope_ms=float(r['transformer_fpga_mean_ms']),static_cpu_scope_ms=float(r['static_cpu_mean_ms']),
   static_fpga_scope_ms=float(r['static_fpga_mean_ms']),delta_vs_transformer=float(r['static_fpga_mean_ms'])-float(r['transformer_fpga_mean_ms'])))
 result=dict(additional_paired_comparisons=comparisons,pipeline_ms=pipeline,extra_11_operators=sorted(extra_values,key=lambda x:x['delta_vs_transformer'],reverse=True))
 (root/'expanded-analysis.json').write_text(json.dumps(result,indent=2)+'\n');output[kind]=result
print('WROTE expanded-analysis.json for both cohorts')
