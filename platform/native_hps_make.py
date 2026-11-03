#!/usr/bin/env python3
"""Adapt Quartus's generated HPS ROM Makefile paths for its Windows make."""
import pathlib
import re
import shutil
import subprocess
import sys

if len(sys.argv)<2 or sys.argv[1]!="make":
    raise SystemExit("Native HPS launcher supports only the generated make hps workflow")
makefile=pathlib.Path("Makefile")
source=makefile.read_text()
converted=re.sub(r"/mnt/([a-zA-Z])/",lambda match: match[1].upper()+":/",source)
if converted!=source:
    makefile.with_name("Makefile.before-native-paths").write_text(source)
    makefile.write_text(converted)
make=shutil.which("make.exe")
if not make:
    raise SystemExit("Installed native Windows make.exe was not found")
sys.exit(subprocess.call([make,*sys.argv[2:]]))
