#!/usr/bin/env python3
"""Apply the integration patch to an existing checkout at the pinned revision."""
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 mnn/prepare.py /path/to/MNN-git-checkout")
checkout = pathlib.Path(sys.argv[1]).resolve()
pin = (root / "PINNED_REVISION").read_text().strip()
head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
if head != pin:
    raise SystemExit("MNN must be checked out at " + pin)
patch = str(root / "patches" / "0001-fpga-tile.patch")
subprocess.run(["git", "-C", str(checkout), "apply", "--check", patch], check=True)
subprocess.run(["git", "-C", str(checkout), "apply", patch], check=True)
print("Applied FPGA tile integration to", checkout)
