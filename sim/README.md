# RTL/software simulation

On Windows, `./sim/run-questa.ps1` compiles and runs `mmio_tb.sv` with the
installed Questa toolchain. It checks 12 actual RTL matrix multiplications and
all 3072 outputs; execution requires a configured simulator license.
`python ./sim/test_jtag_smoke.py` separately validates the System Console Tcl
script against a Python MMIO model. That host-only test is not RTL validation.

`make lint sim` builds the real array/MMIO wrapper with Verilator, then uses
the C++ GEMM library to submit jobs through a simulated Avalon-MM master.
The testbench covers registered reads and successive-cycle accesses.

`sw/tests/test_vectors.hpp` generates deterministic inputs and CPU references.
The same suite runs on Linux hardware: identity, zero, INT8 extremes, mixed
signs, a k=0 impulse at the far corner, and seven pseudorandom matrix pairs.

Additional checks cover byte enables, busy write/START rejection, ACK/ABORT,
reset during RUN, recovery, float conversion eligibility, and software
timeout/error aborts using an intentionally stuck test transport.

This does not simulate HPS boot, generated interconnect, Linux UIO, or Quartus
fitting. See `docs/validation.md` for the tested scope.
