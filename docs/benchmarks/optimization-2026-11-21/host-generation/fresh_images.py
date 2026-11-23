import hashlib,io,json,pathlib,random,tarfile,PIL
from PIL import Image
r=pathlib.Path('build/optimization-20261121');target=r/'fresh-real-200';target.mkdir(exist_ok=False)
old=pathlib.Path('build/mnn-w8a8-tools');used=set()
for p in [old/'calibration-manifest.json',old/'holdout-manifest.json']:
 j=json.loads(p.read_text());used.update(x['source'] for x in j['images'])
labels={'n01440764':0,'n02102040':217,'n02979186':482,'n03000684':491,'n03028079':497,'n03394916':566,'n03417042':569,'n03425413':571,'n03445777':574,'n03888257':701}
rows=[];rng=random.Random(20261121)
with tarfile.open(old/'imagenette2-160.tgz') as tar:
 members=[m for m in tar if m.isfile() and '/val/' in m.name and m.name.lower().endswith(('.jpg','.jpeg')) and m.name not in used]
 for cls,label in sorted(labels.items()):
  candidates=sorted((m for m in members if '/'+cls+'/' in m.name),key=lambda m:m.name);rng.shuffle(candidates)
  for i,m in enumerate(candidates[:20]):
   image=Image.open(io.BytesIO(tar.extractfile(m).read())).convert('RGB').resize((224,224),Image.Resampling.BILINEAR);name=f'{cls}-{i:02d}.png';p=target/name;image.save(p)
   rows.append(dict(file=name,source=m.name,label=label,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert len(rows)==200 and not used.intersection(x['source'] for x in rows)
(r/'fresh-manifest.json').write_text(json.dumps(dict(source='Imagenette2-160 validation; fresh and disjoint from calibration128/diagnostic20/previousholdout200',archive_sha256=hashlib.sha256((old/'imagenette2-160.tgz').read_bytes()).hexdigest(),seed=20261121,pillow=PIL.__version__,images=rows),indent=2))
print('PREPARED 200 fresh disjoint validation images')
