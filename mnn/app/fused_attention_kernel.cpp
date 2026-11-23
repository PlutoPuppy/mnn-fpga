#include "fused_attention_kernel.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#if defined(__arm__) && defined(__ARM_NEON)
#include <arm_neon.h>
extern "C" void TinyVitAttentionQK4x8(const float*,const float*,float*,size_t);
extern "C" void TinyVitAttentionAV4x8(const float*,const float*,float*,size_t,size_t,size_t,size_t);
#endif

namespace tinyvit_attention {
#if defined(__arm__) && defined(__ARM_NEON)
namespace {
void pack_key_rows(float* target,const void* source,bool quantized,float scale,size_t index) {
    float32x4_t a,b,c,d;
    if(quantized) {
        const auto x=vld1q_s8(static_cast<const int8_t*>(source)+index);
        const auto lo=vmovl_s8(vget_low_s8(x)),hi=vmovl_s8(vget_high_s8(x));
        a=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),scale);
        b=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),scale);
        c=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),scale);
        d=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),scale);
    } else {
        const auto* x=static_cast<const float*>(source)+index;
        a=vld1q_f32(x);b=vld1q_f32(x+4);c=vld1q_f32(x+8);d=vld1q_f32(x+12);
    }
    const auto ab=vtrnq_f32(a,b),cd=vtrnq_f32(c,d);
    vst1q_f32(target,vcombine_f32(vget_low_f32(ab.val[0]),vget_low_f32(cd.val[0])));
    vst1q_f32(target+8,vcombine_f32(vget_low_f32(ab.val[1]),vget_low_f32(cd.val[1])));
    vst1q_f32(target+16,vcombine_f32(vget_high_f32(ab.val[0]),vget_high_f32(cd.val[0])));
    vst1q_f32(target+24,vcombine_f32(vget_high_f32(ab.val[1]),vget_high_f32(cd.val[1])));
}
}
#endif
void head(const void* source,bool quantized,float input_scale,const float* bias,float scale,
          size_t rows,size_t tokens,size_t heads,size_t dim,size_t window,size_t h,
          float* q,float* k,float* v,float* scores,float* output,Softmax softmax,size_t first_query,size_t last_query) {
    if(!last_query)last_query=tokens;
    float* parts[3]={q,k,v};
    std::fill(k,k+(tokens+8)*dim,0.0f);
    for(size_t part=0;part<3;++part) {
        auto* target=parts[part];
        size_t i=0;
#if defined(__arm__) && defined(__ARM_NEON)
        if(part==1)for(;i+4<=tokens;i+=4)for(size_t d=0;d<dim;d+=4) {
            const size_t index=(((h*3+part)*dim+d)/4*rows+window*tokens+i)*4;
            pack_key_rows(target+(i/8)*dim*8+d*8+i%8,source,quantized,input_scale,index);
        }
#endif
        for(;i<tokens;++i)for(size_t d=0;d<dim;d+=4) {
            const size_t index=(((h*3+part)*dim+d)/4*rows+window*tokens+i)*4;
#if defined(__arm__) && defined(__ARM_NEON)
            float32x4_t value;
            if(quantized) {
                uint32_t bytes;std::memcpy(&bytes,static_cast<const int8_t*>(source)+index,4);
                const auto x=vmovl_s8(vreinterpret_s8_u64(vcreate_u64(bytes)));
                value=vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(x))),input_scale);
            } else value=vld1q_f32(static_cast<const float*>(source)+index);
            if(part==1) {
                float* cell=target+(i/8)*dim*8+d*8+i%8;
                cell[0]=vgetq_lane_f32(value,0);cell[8]=vgetq_lane_f32(value,1);
                cell[16]=vgetq_lane_f32(value,2);cell[24]=vgetq_lane_f32(value,3);
            } else vst1q_f32(target+i*dim+d,value);
#else
            for(size_t lane=0;lane<4;++lane)target[i*dim+d+lane]=quantized?
                float(static_cast<const int8_t*>(source)[index+lane])*input_scale:static_cast<const float*>(source)[index+lane];
#endif
        }
#if defined(__arm__) && defined(__ARM_NEON)
        if(part!=1)
#endif
        std::fill(target+tokens*dim,target+(tokens+8)*dim,0.0f);
    }
    const size_t stride=tokens+8;
    for(size_t i=first_query;i<last_query;i+=4) {
        const size_t live=std::min(size_t(4),last_query-i);
        const float* queries[4];for(size_t r=0;r<4;++r)queries[r]=q+(i+std::min(r,live-1))*dim;
        for(size_t j=0;j<tokens;j+=8) {
#if defined(__arm__) && defined(__ARM_NEON)
            float dots[32];
            TinyVitAttentionQK4x8(q+i*dim,k+(j/8)*dim*8,dots,dim);
            for(size_t r=0;r<live;++r) {
                auto* out=scores+r*stride+j;
                const auto* position=bias+(i+r)*tokens+j;
                if(j+8<=tokens) {
                    for(size_t col=0;col<8;col+=4)
                        vst1q_f32(out+col,vaddq_f32(vmulq_n_f32(vld1q_f32(dots+r*8+col),scale),vld1q_f32(position+col)));
                } else {
                    for(size_t col=0;col<8;++col)out[col]=j+col<tokens?dots[r*8+col]*scale+position[col]:0;
                }
            }
#else
            for(size_t r=0;r<live;++r)for(size_t col=0;col<8;++col) {
                float value=0;for(size_t d=0;d<dim;++d)if(j+col<tokens)value+=queries[r][d]*k[(j+col)*dim+d];
                scores[r*stride+j+col]=j+col<tokens?value*scale+bias[(i+r)*tokens+j+col]:0;
            }
#endif
        }
        for(size_t r=0;r<live;++r)softmax(scores+r*stride,scores+r*stride,tokens);
        size_t d=0;
#if defined(__arm__) && defined(__ARM_NEON)
        for(;d+8<=dim;d+=8) {
            if(live==4) {
                const size_t base=((h*dim+d)/4*rows+window*tokens+i)*4;
                TinyVitAttentionAV4x8(scores,v+d,output+base,tokens,dim,stride,rows);continue;
            }

            auto a00=vdupq_n_f32(0),a01=a00,a10=a00,a11=a00,a20=a00,a21=a00,a30=a00,a31=a00;
            for(size_t j=0;j<tokens;++j) {
                const auto v0=vld1q_f32(v+j*dim+d),v1=vld1q_f32(v+j*dim+d+4);
                const float p0=scores[j];
                a00=vmlaq_n_f32(a00,v0,p0);a01=vmlaq_n_f32(a01,v1,p0);
                if(live>1){const float p1=scores[stride+j];a10=vmlaq_n_f32(a10,v0,p1);a11=vmlaq_n_f32(a11,v1,p1);}
                if(live>2){const float p2=scores[2*stride+j];a20=vmlaq_n_f32(a20,v0,p2);a21=vmlaq_n_f32(a21,v1,p2);}
                if(live>3){const float p3=scores[3*stride+j];a30=vmlaq_n_f32(a30,v0,p3);a31=vmlaq_n_f32(a31,v1,p3);}
            }
            const size_t base=((h*dim+d)/4*rows+window*tokens+i)*4;
            vst1q_f32(output+base,a00);vst1q_f32(output+base+rows*4,a01);
            if(live>1){vst1q_f32(output+base+4,a10);vst1q_f32(output+base+rows*4+4,a11);}
            if(live>2){vst1q_f32(output+base+8,a20);vst1q_f32(output+base+rows*4+8,a21);}
            if(live>3){vst1q_f32(output+base+12,a30);vst1q_f32(output+base+rows*4+12,a31);}
        }
#endif
        for(;d<dim;d+=4) {
#if defined(__arm__) && defined(__ARM_NEON)
            auto a0=vdupq_n_f32(0),a1=a0,a2=a0,a3=a0;
            for(size_t j=0;j<tokens;++j) {
                const auto value=vld1q_f32(v+j*dim+d);
                a0=vmlaq_n_f32(a0,value,scores[j]);
                if(live>1)a1=vmlaq_n_f32(a1,value,scores[stride+j]);
                if(live>2)a2=vmlaq_n_f32(a2,value,scores[2*stride+j]);
                if(live>3)a3=vmlaq_n_f32(a3,value,scores[3*stride+j]);
            }
            const size_t base=((h*dim+d)/4*rows+window*tokens+i)*4;
            vst1q_f32(output+base,a0);if(live>1)vst1q_f32(output+base+4,a1);
            if(live>2)vst1q_f32(output+base+8,a2);if(live>3)vst1q_f32(output+base+12,a3);
#else
            for(size_t r=0;r<live;++r)for(size_t lane=0;lane<4;++lane) {
                float value=0;for(size_t j=0;j<tokens;++j)value+=scores[r*stride+j]*v[j*dim+d+lane];
                output[((h*dim+d)/4*rows+window*tokens+i+r)*4+lane]=value;
            }
#endif
        }
    }
    (void)heads;
}
}
