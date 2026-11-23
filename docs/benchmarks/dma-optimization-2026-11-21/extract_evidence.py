import argparse
from pathlib import Path
import tarfile

p=argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('target', type=Path)
a=p.parse_args()
a.target.mkdir(exist_ok=False)
with tarfile.open(a.source) as archive:
    members=archive.getmembers()
    for member in members:
        assert member.isdir() or member.isfile()
        assert (a.target/member.name).resolve().is_relative_to(a.target.resolve())
    archive.extractall(a.target, members=members)
print('Extracted',len(members),'safe members')
