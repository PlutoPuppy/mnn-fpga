"""Check the System Console host script with an independent Python MMIO model.

This validates host-side packing, reference arithmetic and failure detection;
it is not RTL simulation or evidence of physical offload.
"""
from pathlib import Path
import tkinter
import unittest


class Transport:
    def __init__(self, tcl):
        self.tcl = tcl
        self.words = {0: 0x53413136, 4: 0x00010000, 24: 0x20101008}
        self.jobs = 0
        self.corrupt = False
        self.bad_cycles = False
        self.error = False

    def write(self, master, address, values):
        address = int(address, 0)
        values = [int(v, 0) for v in self.tcl.splitlist(values)]
        for offset, value in enumerate(values):
            target = address + offset * 4
            if target == 8:
                if value & 4:
                    self.words[12] = 0
                    self.words[16] = 0
                elif value & 1:
                    self.jobs += 1
                    a = [self.words[0x400 + 4*i] for i in range(256)]
                    b = [self.words[0x800 + 4*i] for i in range(256)]
                    a = [v if v < 128 else v - 256 for v in a]
                    b = [v if v < 128 else v - 256 for v in b]
                    for row in range(16):
                        for col in range(16):
                            result = sum(a[row*16+k]*b[k*16+col] for k in range(16))
                            self.words[0xc00 + 4*(16*row+col)] = result & 0xffffffff
                    if self.corrupt:
                        self.words[0xc00] ^= 1
                    self.words[12] = 4 if self.error else 2
                    self.words[16] = 383 if self.bad_cycles else 384
                elif value & 2:
                    self.words[12] = 0
            elif 0x400 <= target < 0xc00:
                self.words[target] = value & 255
            else:
                self.words[target] = value & 0xffffffff
        return ""

    def read(self, master, address, count):
        address = int(address, 0)
        result = []
        for i in range(int(count)):
            target = address + 4*i
            value = self.words.get(target, 0)
            if 0x400 <= target < 0xc00 and value >= 128:
                value = (value - 256) & 0xffffffff
            result.append(hex(value))
        return " ".join(result)


class JtagScriptTest(unittest.TestCase):
    def setUp(self):
        self.tcl = tkinter.Tcl()
        self.tcl.eval("set ::GEMM_TEST_LIBRARY_ONLY 1")
        script = Path(__file__).resolve().parents[1] / "platform/jtag-smoke.tcl"
        self.tcl.call("source", str(script))
        self.transport = Transport(self.tcl)
        self.tcl.createcommand("master_write_32", self.transport.write)
        self.tcl.createcommand("master_read_32", self.transport.read)
        self.tcl.eval("proc puts {args} {}")

    def test_full_smoke(self):
        self.tcl.call("gemm_smoke", "mock")
        self.assertEqual(self.transport.jobs, 12)

    def test_detects_wrong_result(self):
        self.transport.corrupt = True
        with self.assertRaisesRegex(tkinter.TclError, "expected"):
            self.tcl.call("gemm_smoke", "mock")

    def test_detects_wrong_cycles(self):
        self.transport.bad_cycles = True
        with self.assertRaisesRegex(tkinter.TclError, "cycle count"):
            self.tcl.call("gemm_smoke", "mock")

    def test_detects_device_error(self):
        self.transport.error = True
        with self.assertRaisesRegex(tkinter.TclError, "accelerator error"):
            self.tcl.call("gemm_smoke", "mock")

    def test_rejects_invalid_input_before_transfer(self):
        values = [0] * 256
        values[0] = 128
        with self.assertRaisesRegex(tkinter.TclError, "Invalid INT8"):
            self.tcl.call("gemm_check_tile", "mock", "invalid", tuple(values), tuple([0]*256))
        self.assertEqual(self.transport.jobs, 0)


if __name__ == "__main__":
    unittest.main()
