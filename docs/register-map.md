# Hardware/software ABI 1.0

Fixed operation: `C[i,j] = sum(k=0..15, A[i,k] * B[k,j])`.
Inputs are signed two's-complement INT8; outputs are signed INT32. No bias,
transpose, batching, quantization scales, or partial-tile support.

The peripheral occupies **4096 bytes**. CPU addresses are byte offsets below.
The RTL `avs_address[9:0]` is a **32-bit word index** (`byte_offset / 4`).
Platform Designer uses `addressUnits WORDS`, 32-bit data, four byte enables,
fixed read latency 1, and zero wait states. The device is little-endian.

| Byte offset | Access | Register |
|---|---|---|
| `0x000` | RO | ID = `0x53413136` (`SA16`) |
| `0x004` | RO | ABI = `0x00010000` (major 1, minor 0) |
| `0x008` | WO | CONTROL: bit 0 START, bit 1 ACK, bit 2 ABORT |
| `0x00c` | RO | STATUS: bit 0 BUSY, bit 1 DONE, bit 2 ERROR |
| `0x010` | RO | CYCLES: elapsed job clocks, frozen at completion |
| `0x014` | RW | SCRATCH: all four byte enables honored |
| `0x018` | RO | FORMAT = `0x20101008`: ACC_W=32, N=16, K=16, DATA_W=8 |
| `0x400..0x7fc` | RW idle | A, one INT8 element per word |
| `0x800..0xbfc` | RW idle | B, one INT8 element per word |
| `0xc00..0xffc` | RO | C, one INT32 element per word |

Element `(row,col)` is at `base + 4*(row*16+col)`. **A/B are not packed four
elements per word.** Write the INT8 representation in bits 7:0 with byteenable
bit 0 set. Upper write bytes are ignored; reads sign-extend to 32 bits.
Software always uses aligned 32-bit accesses.

## Submission and ownership

1. Verify ID, ABI and FORMAT; require BUSY=0.
2. ACK previous DONE/ERROR.
3. Write all 256 elements of both operands.
4. Order device writes before writing START.
5. Poll STATUS with a deadline; ERROR aborts the software operation.
6. Only after DONE=1 and BUSY=0, read all outputs.

START clears DONE/ERROR and the cycle counter. START while busy is ignored and
sets ERROR. A/B writes while busy are ignored and set ERROR. Writing read-only
or reserved registers with an active byte enable sets ERROR. Reserved reads
and CONTROL reads return zero. CONTROL only honors byte enable 0; unsupported
control bits are ignored.

ACK clears sticky DONE/ERROR without stopping a job. Software must not ACK
while running: it can obscure an error or coincide with completion. ABORT has
priority over START/ACK, clears the array/status, returns idle, and resets
SCRATCH/CYCLES. Reset has the same externally visible effect. Neither abort
nor reset initializes memories. Reload operands before a new job; C is valid
only after that job completes.

DONE persists until ACK, START, ABORT or reset. A/B can be read while busy;
C may contain a partial result and must not be consumed while busy. UIO access
holds an exclusive advisory lock; the MNN adapter serializes calls with a mutex.
All clients must cooperate. `RegisterIo`/`gemm` alone are not thread-safe.

## Array timing

The original PE and array datapath are reused. At RUN cycle `t`, left lane `i`
gets `A[i,t-i]`, top lane `j` gets `B[t-j,j]`, or zero outside bounds. Reduction
index `k` reaches PE `(i,j)` at `t=k+i+j`. The last product is at t=45.

- CLEAR: 1 clock, clears accumulators and inter-PE registers.
- RUN: 46 clocks, t=0 through 45 inclusive.
- DRAIN: 1 clock with zero external operands.
- STORE: 256 clocks, one C element per clock.

CYCLES reads **304** after success, excluding HPS upload/download. DONE is
asserted on the edge storing the last output. Zero feed outside RUN avoids
unintended accumulation between jobs.

The original LED controller registers `clear_array` one phase late, overlapping
the first RUN edge, and stays in DONE until reset. The original file is
preserved in `rtl/legacy/`; the new controller replaces those behaviors.
