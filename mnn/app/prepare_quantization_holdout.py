#!/usr/bin/env python3
"""Prepare labeled Imagenette validation images disjoint from prior experiments."""
import argparse
import hashlib
import io
import json
import pathlib
import random
import tarfile

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=pathlib.Path, required=True)
    parser.add_argument('--exclude-manifest', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True)
    parser.add_argument('--per-class', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026111202)
    args = parser.parse_args()
    if args.output_dir.exists() or args.per_class <= 0:
        parser.error('Use a new output directory and a positive image count')
    previous = json.loads(args.exclude_manifest.read_text())
    archive_sha = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if previous.get('archive_sha256', archive_sha) != archive_sha:
        parser.error('Archive does not match the exclusion manifest')
    excluded = {x['source'] for x in previous['images']}
    # Imagenette synsets and ImageNet's standard 1000-output class ordering.
    labels = {'n01440764': 0, 'n02102040': 217, 'n02979186': 482, 'n03000684': 491,
              'n03028079': 497, 'n03394916': 566, 'n03417042': 569, 'n03425413': 571,
              'n03445777': 574, 'n03888257': 701}
    rng = random.Random(args.seed)
    selected = []
    with tarfile.open(args.archive) as tar:
        files = [m for m in tar if m.isfile() and '/val/' in m.name and m.name not in excluded
                 and m.name.lower().endswith(('.jpeg', '.jpg'))]
        for cls, label in labels.items():
            candidates = sorted([m for m in files if '/' + cls + '/' in m.name], key=lambda m: m.name)
            rng.shuffle(candidates)
            if len(candidates) < args.per_class:
                raise ValueError('Insufficient disjoint validation images')
            for i, member in enumerate(candidates[:args.per_class]):
                selected.append((member, f'{cls}-{i:02d}.png', label))
        args.output_dir.mkdir(parents=True)
        records = []
        # Archive order avoids repeatedly decompressing the gzip stream.
        for member, filename, label in sorted(selected, key=lambda x: x[0].offset_data):
            image = Image.open(io.BytesIO(tar.extractfile(member).read())).convert('RGB')
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            dest = args.output_dir / filename
            image.save(dest)
            records.append(dict(file=filename, source=member.name, label=label,
                                sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    manifest = dict(seed=args.seed, archive_sha256=archive_sha, calibration_and_tuning_disjoint=True,
                    class_order_source='https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt',
                    preprocessing='RGB, PIL bilinear direct 224x224, no crop',
                    limitation='Ten Imagenette classes, not full ImageNet validation',
                    images=sorted(records, key=lambda x: x['file']))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(records)} disjoint labeled images')


if __name__ == '__main__':
    main()
