# RTL/software simulation

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
