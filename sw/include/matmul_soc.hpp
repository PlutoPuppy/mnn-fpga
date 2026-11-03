#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace fpga {
// j3rryhu/matmul-soc ABI, distinct from systolic_mmio's INT32 ABI.
class MatmulSocIo {
public:
    using Beat = std::array<int8_t, 16>;
    virtual ~MatmulSocIo() = default;
    virtual uint32_t read(unsigned offset) = 0;
    virtual void write(unsigned offset, uint32_t value) = 0;
    // Buffer ports REQUIRE full 128-bit transactions: they ignore byte enables.
    virtual void write_beat(unsigned offset, const Beat& data) = 0;
    virtual Beat read_beat(unsigned offset) = 0;
};
struct MatmulSocStats { size_t submissions = 0; };
// Dense A[M,K] * B[K,N], optionally stored transposed. Stages output until
// every tile completes. Throws on invalid inputs/device failure; C is unchanged.
// Approximate FP32 via symmetric INT8 operands and hardware Q0.16 floor/saturate.
MatmulSocStats matmul_soc_gemm(MatmulSocIo& io, const float* a, const float* b,
    float* c, size_t m, size_t n, size_t k, bool transpose_a = false,
    bool transpose_b = false, bool transpose_c = false, const float* bias = nullptr,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));
}
