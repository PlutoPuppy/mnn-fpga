#pragma once
#include "matmul_soc.hpp"

namespace fpga {
// NI82: full INT32 reduction, one per-column requantization in hardware.
// Never usable with the legacy per-band INT8 bitstream.
constexpr uint32_t native_int8_abi=0x4e493832;
constexpr uint32_t native_int8_extended_abi=0x4e493833;
struct NativeInt8Weights {
    size_t k=0,n=0,kp=0,tile_m=0,tile_n=0;
    size_t operand_b_bytes=16384;
    size_t operand_a_bytes=16384,operand_b_offset=0x5000;
    std::vector<MatmulSocIo::CachedBuffer> weights;
    std::vector<std::vector<int8_t>> parameters;
};
// Update only the epilogue. The packed B images remain resident in DDR.
void set_native_int8_parameters(NativeInt8Weights& weights,const MatmulSocInt8Requantization& requant);
NativeInt8Weights prepare_native_int8(MatmulSocIo& io,const int8_t* weights_oi,
    size_t k,size_t n,const MatmulSocInt8Requantization& requant);
MatmulSocStats native_int8_linear(MatmulSocIo& io,const NativeInt8Weights& weights,
    const int8_t* input,int8_t* output,size_t rows,bool row_major=false);
// The intermediate is GELU/requantized by the supplied exact 256-entry table,
// retained in FPGA output RAM, and copied to A using FPGA-local DMA.
struct NativePairStats : MatmulSocStats {size_t local_bytes=0,hidden_bytes_avoided=0;};
NativePairStats native_int8_pair(MatmulSocIo& io,const NativeInt8Weights& first,
    const NativeInt8Weights& second,const std::array<int8_t,256>& activation,
    const int8_t* input_nc4,int8_t* output_nc4,size_t rows);
}
