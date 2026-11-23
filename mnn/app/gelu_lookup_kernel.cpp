#include "gelu_lookup_kernel.hpp"
#include <algorithm>
#if defined(__arm__) && defined(__ARM_NEON)
#include <arm_neon.h>
extern "C" void TinyVitGeluReciprocal(float*,const float*,size_t,const float*);
#endif
namespace tinyvit_gelu {
void coefficients(float* table,const float* samples) {
    for(size_t i=0;i<intervals;++i) {
        const float slope=(samples[i+1]-samples[i])*256.0f;
        table[i*2]=slope;table[i*2+1]=samples[i]-slope*(float(i)/256.0f-8.0f);
    }
}
void apply(float* destination,const float* source,size_t count,const float* table) {
    size_t i=0;
#if defined(__arm__) && defined(__ARM_NEON)
    const auto low=vdupq_n_f32(-8),high=vdupq_n_f32(8),last=vdupq_n_f32(4095);
    for(;i+4<=count;i+=4) {
        const auto x=vld1q_f32(source+i);
        const auto ix=vcvtq_s32_f32(vminq_f32(last,vmaxq_f32(vdupq_n_f32(0),vmulq_n_f32(vsubq_f32(x,low),256))));
        const auto p0=vld1_f32(table+2*vgetq_lane_s32(ix,0)),p1=vld1_f32(table+2*vgetq_lane_s32(ix,1));
        const auto p2=vld1_f32(table+2*vgetq_lane_s32(ix,2)),p3=vld1_f32(table+2*vgetq_lane_s32(ix,3));
        const auto ab=vtrn_f32(p0,p1),cd=vtrn_f32(p2,p3);
        const auto a=vcombine_f32(ab.val[0],cd.val[0]),b=vcombine_f32(ab.val[1],cd.val[1]);
        auto value=vmlaq_f32(b,a,x);
        value=vbslq_f32(vcgeq_f32(x,high),x,value);
        value=vbslq_f32(vcleq_f32(x,low),vdupq_n_f32(-0.0f),value);vst1q_f32(destination+i,value);
    }
#endif
    for(;i<count;++i) {
        const float x=source[i];
        if(x>=8){destination[i]=x;continue;}if(x<=-8){destination[i]=-0.0f;continue;}
        const int index=std::max(0,std::min(4095,int((x+8)*256)));
        destination[i]=table[index*2]*x+table[index*2+1];
    }
}
void reciprocal(float* destination,const float* source,size_t count) {
    size_t i=0;
#if defined(__arm__) && defined(__ARM_NEON)
    if(count>=8) {
        constexpr float parameters[8]={0.044715f,0.79788458f,378,17325,135135,28,3150,62370};
        TinyVitGeluReciprocal(destination,source,count/8,parameters);i=count/8*8;
    }
#endif
    for(;i<count;++i) {
        const float x=source[i];const float u=std::max(-5.0f,std::min(5.0f,0.79788458f*(0.044715f*x*x*x+x))),z=u*u;
        const float a=u*(135135+z*(17325+z*(378+z))),b=135135+z*(62370+z*(3150+28*z));
        destination[i]=(1+std::max(-1.0f,std::min(1.0f,a/b)))*x*0.5f;
    }
}
void polynomial(float* destination,const float* source,size_t count) {
    // Fit the existing rational GELU tail P(a) = -GELU(-a)/a on a in [0,3.9].
    // GELU(x) = max(x,0) - |x| P(|x|). The clamped reference is linear
    // beyond the fit interval. Coefficients are in powers of a/3.9.
    constexpr float p[]={0.500081062f,-1.56162882f,0.129085794f,2.59862399f,7.97955227f,
                         -36.7657623f,56.2551231f,-43.9097595f,17.7161007f,-2.94141817f};
    constexpr float limit=3.9f, inverse=1.0f/limit;
    size_t i=0;
#if defined(__arm__) && defined(__ARM_NEON)
    const auto zero=vdupq_n_f32(0),bound=vdupq_n_f32(limit);
    for(;i+4<=count;i+=4) {
        const auto x=vld1q_f32(source+i),a=vabsq_f32(x);
        const auto t=vmulq_n_f32(vminq_f32(bound,a),inverse);
        auto y=vdupq_n_f32(p[9]);
        for(int j=8;j>=0;--j)y=vmlaq_f32(vdupq_n_f32(p[j]),y,t);
        const auto correction=vbslq_f32(vcltq_f32(a,bound),vmaxq_f32(zero,y),zero);
        vst1q_f32(destination+i,vmlsq_f32(vmaxq_f32(zero,x),a,correction));
    }
#endif
    for(;i<count;++i) {
        const float x=source[i],a=x<0 ? -x : x;
        float correction=0;
        if(a<limit) {
            const float t=a*inverse;float y=p[9];
            for(int j=8;j>=0;--j)y=p[j]+y*t;
            correction=a*std::max(0.0f,y);
        }
        destination[i]=std::max(0.0f,x)-correction;
    }
}
}
