#!/usr/bin/env python3
"""Compare saved tinyvit_run outputs; metrics do not certify model accuracy."""
import argparse
import array
import json
import math
import pathlib
import sys


def load(directory):
    root = pathlib.Path(directory)
    manifest = json.loads((root / "outputs.json").read_text())
    outputs = {}
    for item in manifest["outputs"]:
        if item["dtype"] != "float32" or item["byte_order"] != "little":
            raise ValueError("Expected little-endian FP32 output")
        values = array.array("f")
        values.frombytes((root / item["file"]).read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        if len(values) != math.prod(item["shape"]) or not values:
            raise ValueError("Output shape/length mismatch")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite output")
        if item["name"] in outputs:
            raise ValueError("Duplicate output name")
        outputs[item["name"]] = (item["shape"], values)
    return outputs


def compare(cpu, fpga):
    if cpu.keys() != fpga.keys():
        raise ValueError("Output names differ")
    metrics = {}
    for name, (shape, a) in cpu.items():
        other_shape, b = fpga[name]
        if shape != other_shape:
            raise ValueError("Output shapes differ: " + name)
        squared = sum((x-y)**2 for x, y in zip(a, b))
        aa, bb = sum(x*x for x in a), sum(y*y for y in b)
        row = {
            "max_abs_error": max(abs(x-y) for x, y in zip(a, b)),
            "rmse": math.sqrt(squared/len(a)),
            "relative_l2": math.sqrt(squared/aa) if aa else None,
            "cosine_similarity": sum(x*y for x, y in zip(a, b))/math.sqrt(aa*bb) if aa and bb else None,
        }
        if len(shape) == 1 or (len(shape) == 2 and shape[0] == 1):
            ka = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:5]
            kb = sorted(range(len(b)), key=lambda i: b[i], reverse=True)[:5]
            row.update(cpu_top5=ka, fpga_top5=kb, top1_agrees=ka[0] == kb[0])
        metrics[name] = row
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_dir")
    parser.add_argument("fpga_dir")
    parser.add_argument("--max-abs-error", type=float, help="Optional explicit acceptance threshold")
    args = parser.parse_args()
    if args.max_abs_error is not None and (not math.isfinite(args.max_abs_error) or args.max_abs_error < 0):
        parser.error("--max-abs-error must be finite and nonnegative")
    metrics = compare(load(args.cpu_dir), load(args.fpga_dir))
    print(json.dumps(metrics, indent=2, allow_nan=False))
    if args.max_abs_error is not None and any(x["max_abs_error"] > args.max_abs_error for x in metrics.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
