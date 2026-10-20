#!/usr/bin/env python3
"""Integration checks for the actual full TinyViT classifier and runner."""
import argparse
import array
import json
import math
import os
import pathlib
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--exe", required=True)
p.add_argument("--model", required=True)
args = p.parse_args()
exe, model = str(pathlib.Path(args.exe).resolve()), str(pathlib.Path(args.model).resolve())
env = dict(os.environ)
env.pop("MNN_FPGA_UIO", None)
base = [exe, "--model", model, "--height", "224", "--width", "224", "--threads", "1"]

def run(options, ok=True):
    r = subprocess.run(base + list(map(str, options)), env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if (r.returncode == 0) != ok:
        raise AssertionError(r.stdout)
    return r.stdout

def floats(path):
    a = array.array("f")
    a.frombytes(path.read_bytes())
    return a

with tempfile.TemporaryDirectory(prefix="tinyvit-runner-") as tmp:
    tmp = pathlib.Path(tmp)
    out = tmp / "zero"
    text = run(["--zeros", "--output-dir", out])
    assert "FPGA submissions: 0" in text
    meta = json.loads((out / "outputs.json").read_text())
    assert meta["input_shape"] == [1, 3, 224, 224]
    assert len(meta["outputs"]) == 1 and meta["outputs"][0]["shape"] == [1, 1000]
    assert all(math.isfinite(x) for x in floats(out / "output_0.f32"))
    assert (out / "output_0.f32").stat().st_size == 4000

    # Constant RGB image: resize/crop cannot alter its known per-channel values.
    image = tmp / "color.ppm"
    image.write_bytes(b"P6\n3 2\n255\n" + bytes([255, 0, 128]) * 6)
    for resize in ([], ["--resize-short", "256"]):
        dump, image_out, raw_out = tmp / "input.f32", tmp / "image", tmp / "raw"
        run(["--image", image, "--mean", "0.5,0.25,0", "--std", "0.5,0.25,1",
             "--dump-input", dump, "--output-dir", image_out] + resize)
        x = floats(dump)
        n = 224 * 224
        assert len(x) == 3*n
        for c, expected in enumerate([1.0, -1.0, 128/255]):
            assert all(abs(v-expected) < 1e-6 for v in x[c*n:(c+1)*n])
        run(["--input-f32", dump, "--output-dir", raw_out])
        a, b = floats(image_out / "output_0.f32"), floats(raw_out / "output_0.f32")
        assert len(a) == len(b) == 1000
        assert max(abs(x-y) for x, y in zip(a,b)) < 1e-5

    short = tmp / "short.f32"
    short.write_bytes(b"\x00"*4)
    assert "exactly 3*H*W" in run(["--input-f32", short], False)
    assert "explicit --mean" in run(["--image", image], False)
    assert "std must" in run(["--image", image, "--mean", "0,0,0", "--std", "1,0,1"], False)
    assert "Choose exactly one" in run(["--zeros", "--image", image], False)
    assert "No FPGA submissions" in run(["--zeros", "--require-fpga"], False)
    assert "Cannot load MNN model" in run(["--zeros", "--model", tmp / "missing.mnn"], False)
print("PASS: full model, RGB normalization, resize/crop, raw replay, outputs, error paths")
