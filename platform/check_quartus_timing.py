#!/usr/bin/env python3
"""Reject a completed Quartus STA summary with negative or missing slack."""
import argparse
import pathlib
import re


def require_timing(path):
    path=pathlib.Path(path)
    text=path.read_text(errors="replace")
    slacks=[float(value) for value in re.findall(r"^\s*Slack\s*:\s*(-?\d+(?:\.\d+)?)\s*$",text,re.MULTILINE)]
    if not slacks or not any("Setup" in line and "pll_0" in line for line in text.splitlines()):
        raise RuntimeError("Missing core PLL setup timing checks: "+str(path))
    worst=min(slacks)
    if worst<0:
        raise RuntimeError("Timing failed: worst slack %.3f ns; see %s" % (worst,path))
    print("Timing passed: %d checks, minimum slack %.3f ns" % (len(slacks),worst),flush=True)
    return worst


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=pathlib.Path)
    args=parser.parse_args()
    try:
        require_timing(args.summary)
    except (OSError,RuntimeError) as error:
        parser.exit(1,str(error)+"\n")
