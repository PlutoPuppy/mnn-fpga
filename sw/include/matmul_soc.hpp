#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "matmul_soc_profile.hpp"

namespace fpga {
// j3rryhu/matmul-soc ABI, distinct from systolic_mmio's INT32 ABI.
class MatmulSocIo {
public:
    MatmulSocProfile* profile = nullptr;
    using Beat = std::array<int8_t, 16>;
    virtual ~MatmulSocIo() = default;
    virtual uint32_t read(unsigned offset) = 0;
    virtual void write(unsigned offset, uint32_t value) = 0;
    // Buffer ports REQUIRE full 128-bit transactions: they ignore byte enables.
    virtual void write_beat(unsigned offset, const Beat& data) = 0;
    virtual Beat read_beat(unsigned offset) = 0;
    virtual void write_buffer(unsigned offset, const int8_t* data, size_t bytes);
    virtual void read_buffer(unsigned offset, int8_t* data, size_t bytes);
    struct CachedBuffer {
        std::vector<int8_t> bytes;
        uint32_t physical=0;
        size_t size=0;
        const void* owner=nullptr;
    };
    virtual CachedBuffer cache_buffer(const int8_t* data,size_t bytes);
    virtual void write_cached(unsigned offset,const CachedBuffer& data);
    virtual void write_nc4(unsigned offset,const int8_t* data,size_t rows,size_t channels,
                           size_t first,size_t count,size_t padded_rows,size_t padded_channels);
    virtual void read_nc4(unsigned offset,int8_t* data,size_t rows,size_t channels,
                          size_t first,size_t column,size_t count,size_t columns,
                          size_t padded_rows,size_t padded_columns);
    // NI82 resident ABI only. Source is FPGA output, destination is operand A.
    virtual void feedback(unsigned source,unsigned destination,size_t bytes);
};
struct MatmulSocStats {
    size_t submissions = 0;
    size_t operand_bytes = 0, result_bytes = 0, quantized_values = 0;
};
enum class MatmulSocTilePolicy { Fixed64, Capacity };
struct MatmulSocPlan {
    size_t m=0,n=0,k=0,tile_m=0,tile_n=0,tile_k=0;
    uint64_t jobs=0,operand_bytes=0,result_bytes=0,accumulated_values=0,cost_proxy_ns=0;
    MatmulSocTilePolicy policy=MatmulSocTilePolicy::Fixed64;
};
// Capacity policy enumerates 16-aligned shapes satisfying both operand banks,
// the output banks and the 10-bit input extent. Cost is a host-work proxy,
// not a guarantee of latency: 30 ns/input byte, 14 ns/output byte,
// 20 ns/host accumulation and 5 us/job, rounded from prior QK phase profiles.
MatmulSocPlan plan_matmul_soc(size_t m,size_t n,size_t k,
    MatmulSocTilePolicy policy=MatmulSocTilePolicy::Fixed64);
// Dense row-major signed INT8. real_value = values[i] * scale; zero point 0.
// Callers with already quantized activations may supply this directly.
struct MatmulSocOperand {
    size_t rows = 0, cols = 0;
    std::vector<int8_t> values;
    double scale = 1;
};
MatmulSocOperand prepare_matmul_soc_operand(const float* data, size_t rows,
    size_t cols, bool transpose = false, MatmulSocProfile* profile = nullptr);
MatmulSocStats matmul_soc_gemm_prepared(MatmulSocIo& io,
    const MatmulSocOperand& a, const MatmulSocOperand& b, float* c,
    bool transpose_c = false, const float* bias = nullptr,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000),
    const MatmulSocPlan* plan = nullptr);
// Native W8A8 operands, per-output-channel scaling and output-unit Q16 bias.
// No FP32 activation/output arrays. The fabric still floors each K=16 band;
// this adapter models that arithmetic, not an exact INT32 dot product.
struct MatmulSocInt8Requantization {
    std::vector<double> multiplier; // input_scale * weight_scale[channel] / output_scale
    std::vector<int64_t> bias_q16;   // (bias/output_scale + output_zero_point) * 65536
    int minimum=-127,maximum=127;
};
MatmulSocStats matmul_soc_gemm_int8(MatmulSocIo& io,const MatmulSocOperand& a,
    const MatmulSocOperand& b,int8_t* c,const MatmulSocInt8Requantization& requant,
    const MatmulSocPlan& plan,std::chrono::milliseconds timeout=std::chrono::milliseconds(1000));
// Dense A[M,K] * B[K,N], optionally stored transposed. Stages output until
// every tile completes. Throws on invalid inputs/device failure; C is unchanged.
// Quantizes each operand once, then batches up to 256x64 * 64x256 per job.
// Hardware performs internal 16x16 tiling; host sums reduction panels.
// Approximate FP32 via symmetric INT8 operands and hardware Q0.16 floor/saturate.
MatmulSocStats matmul_soc_gemm(MatmulSocIo& io, const float* a, const float* b,
    float* c, size_t m, size_t n, size_t k, bool transpose_a = false,
    bool transpose_b = false, bool transpose_c = false, const float* bias = nullptr,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));
}
