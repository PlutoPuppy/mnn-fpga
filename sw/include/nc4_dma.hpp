#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__arm__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace fpga {
namespace nc4_dma {
#if defined(__arm__) && defined(__ARM_NEON)
inline void store(volatile int8_t* p,uint32x4_t v) {
    asm volatile("vst1.8 {%e0-%f0}, [%1 :128]" :: "w"(v),"r"(p):"memory");
}
inline uint32x4_t load(const volatile int8_t* p) {
    uint32x4_t v;
    asm volatile("vld1.8 {%e0-%f0}, [%1 :128]" : "=w"(v):"r"(p):"memory");return v;
}
inline void transpose(uint32x4_t& a,uint32x4_t& b,uint32x4_t& c,uint32x4_t& d) {
    const auto ab=vtrnq_u32(a,b),cd=vtrnq_u32(c,d);
    a=vcombine_u32(vget_low_u32(ab.val[0]),vget_low_u32(cd.val[0]));
    b=vcombine_u32(vget_low_u32(ab.val[1]),vget_low_u32(cd.val[1]));
    c=vcombine_u32(vget_high_u32(ab.val[0]),vget_high_u32(cd.val[0]));
    d=vcombine_u32(vget_high_u32(ab.val[1]),vget_high_u32(cd.val[1]));
}
#endif
// Callers validate bounds/16-byte row alignment and issue device barriers.
// MNN pack4 H=W=1: (channel_group*rows + row)*4 + channel_lane.
inline void pack(volatile int8_t* target,const int8_t* source,size_t rows,size_t channels,
                 size_t first,size_t count,size_t mp,size_t kp) {
    size_t i=0;
#if defined(__arm__) && defined(__ARM_NEON)
    for(;i+4<=count;i+=4)for(size_t ch=0;ch<kp;ch+=16) {
        auto group=[&](size_t g){return ch+g<channels ?
            vld1q_u32(reinterpret_cast<const uint32_t*>(source+((ch+g)/4*rows+first+i)*4)):vdupq_n_u32(0);};
        auto a=group(0),b=group(4),c=group(8),d=group(12);transpose(a,b,c,d);
        store(target+i*kp+ch,a);store(target+(i+1)*kp+ch,b);
        store(target+(i+2)*kp+ch,c);store(target+(i+3)*kp+ch,d);
    }
    for(;i<mp;++i)for(size_t ch=0;ch<kp;ch+=16) {
        alignas(16) int8_t beat[16]={};
        if(i<count)for(size_t j=0;j<16 && ch+j<channels;++j)beat[j]=source[((ch+j)/4*rows+first+i)*4+(ch+j)%4];
        store(target+i*kp+ch,vreinterpretq_u32_s8(vld1q_s8(beat)));
    }
#else
    for(;i<mp;++i)for(size_t ch=0;ch<kp;++ch)
        target[i*kp+ch]=i<count && ch<channels ? source[(ch/4*rows+first+i)*4+ch%4]:0;
#endif
}
inline void unpack(int8_t* target,const volatile int8_t* source,size_t rows,size_t first,size_t column,
                   size_t count,size_t columns,size_t np) {
    size_t i=0;
#if defined(__arm__) && defined(__ARM_NEON)
    for(;i+4<=count;i+=4) {
        size_t j=0;
        for(;j+16<=columns;j+=16) {
            auto a=load(source+i*np+j),b=load(source+(i+1)*np+j),c=load(source+(i+2)*np+j),d=load(source+(i+3)*np+j);
            transpose(a,b,c,d);
            vst1q_u32(reinterpret_cast<uint32_t*>(target+((column+j)/4*rows+first+i)*4),a);
            vst1q_u32(reinterpret_cast<uint32_t*>(target+((column+j+4)/4*rows+first+i)*4),b);
            vst1q_u32(reinterpret_cast<uint32_t*>(target+((column+j+8)/4*rows+first+i)*4),c);
            vst1q_u32(reinterpret_cast<uint32_t*>(target+((column+j+12)/4*rows+first+i)*4),d);
        }
        // Rare output-channel tail. Read complete aligned beats, then select
        // only live channels; device memory never sees byte transactions.
        if(j<columns)for(size_t r=0;r<4;++r) {
            alignas(16) int8_t beat[16];vst1q_s8(beat,vreinterpretq_s8_u32(load(source+(i+r)*np+j)));
            for(size_t q=0;j+q<columns;++q)target[((column+j+q)/4*rows+first+i+r)*4+(column+j+q)%4]=beat[q];
        }
    }
    for(;i<count;++i)for(size_t j=0;j<columns;j+=16) {
        alignas(16) int8_t beat[16];vst1q_s8(beat,vreinterpretq_s8_u32(load(source+i*np+j)));
        for(size_t q=0;q<16 && j+q<columns;++q)target[((column+j+q)/4*rows+first+i)*4+(column+j+q)%4]=beat[q];
    }
#else
    for(;i<count;++i)for(size_t j=0;j<columns;++j)target[((column+j)/4*rows+first+i)*4+(column+j)%4]=source[i*np+j];
#endif
}
}
}
