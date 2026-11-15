#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace fpga {
// Output-unit Q16. Caller ensures |code*gain+bias| <= INT32_MAX-32768.
// Signed round-to-nearest, ties away from zero, then clamp to INT8 range.
inline void requantize_int8(const int8_t* codes,int8_t* out,const int32_t* gains,
    const int32_t* bias,size_t count,int minimum,int maximum,bool vectorize=true) {
    size_t j=0;
#if defined(__ARM_NEON)
    if(vectorize) {
        const auto half=vdupq_n_s32(32768),zero=vdupq_n_s32(0);
        const auto low=vdupq_n_s32(minimum),high=vdupq_n_s32(maximum);
        auto convert=[&](int32x4_t code,size_t offset) {
            const auto value=vaddq_s32(vmulq_s32(code,vld1q_s32(gains+offset)),vld1q_s32(bias+offset));
            const auto magnitude=vshrq_n_s32(vaddq_s32(vabsq_s32(value),half),16);
            const auto rounded=vbslq_s32(vcltq_s32(value,zero),vnegq_s32(magnitude),magnitude);
            return vmaxq_s32(low,vminq_s32(high,rounded));
        };
        for(;j+8<=count;j+=8) {
            const auto code=vmovl_s8(vld1_s8(codes+j));
            const auto a=convert(vmovl_s16(vget_low_s16(code)),j);
            const auto b=convert(vmovl_s16(vget_high_s16(code)),j+4);
            vst1_s8(out+j,vmovn_s16(vcombine_s16(vmovn_s32(a),vmovn_s32(b))));
        }
    }
#else
    (void)vectorize;
#endif
    for(;j<count;++j) {
        const int32_t value=int32_t(codes[j])*gains[j]+bias[j];
        const int32_t rounded=value>=0?(value+32768)>>16:-((-value+32768)>>16);
        out[j]=int8_t(std::max(minimum,std::min(maximum,int(rounded))));
    }
}
}
