#include "matmul_soc.hpp"
#include "int8_requant.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fpga {
namespace {
size_t area(size_t x,size_t y) {
    if(!x || !y || x>std::numeric_limits<size_t>::max()/y) throw std::invalid_argument("Invalid matrix dimensions");
    return x*y;
}
size_t padded(size_t x) {return (x+15)/16*16;}
uint64_t mul(uint64_t a,uint64_t b) {
    if(b && a>std::numeric_limits<uint64_t>::max()/b) throw std::overflow_error("Tile cost overflow");
    return a*b;
}
uint64_t add(uint64_t a,uint64_t b) {
    if(a>std::numeric_limits<uint64_t>::max()-b) throw std::overflow_error("Tile cost overflow");
    return a+b;
}
bool fits(size_t m,size_t n,size_t k) {
    return m && n && k && m%16==0 && n%16==0 && k%16==0 && n<=1008 &&
           m<=1024 && k<=1024 && m*k<=16384 && k*n<=16384 && m*n<=65536;
}
}
MatmulSocPlan plan_matmul_soc(size_t m,size_t n,size_t k,MatmulSocTilePolicy policy) {
    area(m,n);area(m,k);area(k,n);
    const auto limit=std::numeric_limits<size_t>::max()-15;
    if(m>limit || n>limit || k>limit) throw std::overflow_error("Padded shape overflow");
    const size_t pm=padded(m),pn=padded(n),pk=padded(k);
    auto make=[&](size_t tm,size_t tn,size_t tk) {
        MatmulSocPlan p;p.m=m;p.n=n;p.k=k;p.tile_m=tm;p.tile_n=tn;p.tile_k=tk;p.policy=policy;
        const uint64_t rm=(m-1)/tm+1,cn=(n-1)/tn+1,kp=(k-1)/tk+1;
        p.jobs=mul(mul(rm,cn),kp);
        p.operand_bytes=add(mul(mul(pm,pk),cn),mul(mul(pk,pn),rm));
        p.result_bytes=mul(mul(pm,pn),kp);p.accumulated_values=mul(mul(m,n),kp);
        p.cost_proxy_ns=add(add(mul(30,p.operand_bytes),mul(14,p.result_bytes)),
                            add(mul(20,p.accumulated_values),mul(5000,p.jobs)));
        return p;
    };
    auto best=make(std::min(pm,size_t(256)),std::min(pn,size_t(256)),std::min(pk,size_t(64)));
    if(policy==MatmulSocTilePolicy::Fixed64) return best;
    for(size_t tm=16;tm<=std::min(pm,size_t(1024));tm+=16)
        for(size_t tn=16;tn<=std::min(pn,size_t(1008));tn+=16) {
            if(tm*tn>65536) continue;
            const size_t maxk=std::min({pk,size_t(1024),size_t(16384)/tm,size_t(16384)/tn})/16*16;
            for(size_t tk=16;tk<=maxk;tk+=16) {
                auto p=make(tm,tn,tk);
                if(p.cost_proxy_ns<best.cost_proxy_ns ||
                   (p.cost_proxy_ns==best.cost_proxy_ns && (p.jobs<best.jobs ||
                    (p.jobs==best.jobs && tm*tn>best.tile_m*best.tile_n)))) best=p;
            }
        }
    return best;
}
void MatmulSocIo::write_buffer(unsigned off,const int8_t* data,size_t bytes) {
    if(!data || !bytes || bytes%16 || off%16) throw std::invalid_argument("Unaligned buffer image");
    for(size_t i=0;i<bytes;i+=16) {
        Beat beat{};std::copy_n(data+i,16,beat.begin());write_beat(off+unsigned(i),beat);
    }
}
void MatmulSocIo::read_buffer(unsigned off,int8_t* data,size_t bytes) {
    if(!data || !bytes || bytes%16 || off%16) throw std::invalid_argument("Unaligned buffer image");
    for(size_t i=0;i<bytes;i+=16) {
        const auto beat=read_beat(off+unsigned(i));std::copy(beat.begin(),beat.end(),data+i);
    }
}
MatmulSocIo::CachedBuffer MatmulSocIo::cache_buffer(const int8_t* data,size_t bytes) {
    if(!data || !bytes || bytes%16)throw std::invalid_argument("Invalid cache image");
    CachedBuffer c;c.bytes.assign(data,data+bytes);c.size=bytes;c.owner=this;return c;
}
void MatmulSocIo::write_cached(unsigned off,const CachedBuffer& c) {
    if(c.owner!=this || c.bytes.size()!=c.size)throw std::invalid_argument("Invalid cache owner");
    write_buffer(off,c.bytes.data(),c.size);
}
void MatmulSocIo::write_nc4(unsigned off,const int8_t* data,size_t rows,size_t channels,
    size_t first,size_t count,size_t mp,size_t kp) {
    if(!data || !count || first+count>rows || count>mp || channels>kp || mp%16 || kp%16)
        throw std::invalid_argument("Invalid NC4 input tile");
    std::vector<int8_t> dense(mp*kp,0);
    for(size_t i=0;i<count;++i)for(size_t k=0;k<channels;++k)
        dense[i*kp+k]=data[(k/4*rows+first+i)*4+k%4];
    write_buffer(off,dense.data(),dense.size());
}
void MatmulSocIo::read_nc4(unsigned off,int8_t* data,size_t rows,size_t channels,
    size_t first,size_t column,size_t count,size_t columns,size_t mp,size_t np) {
    if(!data || first+count>rows || column+columns>channels || count>mp || columns>np)
        throw std::invalid_argument("Invalid NC4 output tile");
    std::vector<int8_t> dense(mp*np);read_buffer(off,dense.data(),dense.size());
    for(size_t i=0;i<count;++i)for(size_t j=0;j<columns;++j)
        data[((column+j)/4*rows+first+i)*4+(column+j)%4]=dense[i*np+j];
}
void MatmulSocIo::feedback(unsigned,unsigned,size_t) {throw std::runtime_error("FPGA-local DMA unavailable");}
MatmulSocOperand prepare_matmul_soc_operand(const float* data,size_t rows,size_t cols,bool trans,MatmulSocProfile* profile) {
    ProfileSteps timer(profile);
    const auto count=area(rows,cols);
    if(!data) throw std::invalid_argument("Null operand");
    double peak=0;
    for(size_t i=0;i<count;++i) {
        if(!std::isfinite(data[i])) throw std::invalid_argument("Nonfinite operand");
        peak=std::max(peak,std::abs(double(data[i])));
    }
    timer.end(MatmulSocProfile::peak_scan);
    MatmulSocOperand out;out.rows=rows;out.cols=cols;out.scale=peak ? peak/127.0:1.0;
    out.values.resize(count);
    for(size_t i=0;i<rows;++i) for(size_t j=0;j<cols;++j)
        out.values[i*cols+j]=int8_t(std::round(data[trans ? j*rows+i:i*cols+j]/out.scale));
    timer.end(MatmulSocProfile::quantize);
    return out;
}
MatmulSocStats matmul_soc_gemm_prepared(MatmulSocIo& io,
    const MatmulSocOperand& a,const MatmulSocOperand& b,float* c,
    bool tc,const float* bias,std::chrono::milliseconds timeout,const MatmulSocPlan* supplied_plan) {
    auto* profile=io.profile;
    ProfileSteps timer(profile);
    const auto ac=area(a.rows,a.cols),bc=area(b.rows,b.cols),cc=area(a.rows,b.cols);
    if(a.cols!=b.rows || a.values.size()!=ac || b.values.size()!=bc || !c || timeout.count()<=0 ||
       !std::isfinite(a.scale) || a.scale<=0 || !std::isfinite(b.scale) || b.scale<=0)
        throw std::invalid_argument("Invalid prepared INT8 GEMM");
    const size_t m=a.rows,n=b.cols,k=a.cols;
    timer.end(MatmulSocProfile::workspace);
    const auto plan=supplied_plan ? *supplied_plan:plan_matmul_soc(m,n,k);
    if(plan.m!=m || plan.n!=n || plan.k!=k || !fits(plan.tile_m,plan.tile_n,plan.tile_k))
        throw std::invalid_argument("Invalid matrix tile plan");
    if(profile) {profile->tile_m=plan.tile_m;profile->tile_n=plan.tile_n;profile->tile_k=plan.tile_k;}
    timer.end(MatmulSocProfile::tile_plan);
    if(bias) for(size_t j=0;j<n;++j) if(!std::isfinite(bias[j])) throw std::invalid_argument("Nonfinite bias");
    std::vector<double> result(cc,0.0);
    MatmulSocStats stats;
    timer.end(MatmulSocProfile::workspace);
    // The selected plan obeys both 16 KiB operand buffers and the 64 KiB
    // result buffer. Fixed64 retains the previous 256x64x256 policy; capacity
    // can use deeper K panels by reducing M/N. Edge jobs pad to multiples of16.
    for(size_t r=0;r<m;r+=plan.tile_m) for(size_t col=0;col<n;col+=plan.tile_n)
        for(size_t p=0;p<k;p+=plan.tile_k) {
            const size_t mr=std::min(plan.tile_m,m-r),nc=std::min(plan.tile_n,n-col),kr=std::min(plan.tile_k,k-p);
            const size_t mp=padded(mr),np=padded(nc),kp=padded(kr),bands=kp/16;
            std::vector<int8_t> qa(mp*kp,0),qb(kp*np,0),output(mp*np);
            for(size_t i=0;i<mr;++i) std::copy_n(a.values.begin()+(r+i)*k+p,kr,qa.begin()+i*kp);
            for(size_t t=0;t<kr;++t) std::copy_n(b.values.begin()+(p+t)*n+col,nc,qb.begin()+t*np);
            timer.end(MatmulSocProfile::pack_pad);
            unsigned row_bound=0,bpeak=0;
            for(size_t i=0;i<mr;++i) {
                unsigned sum=0;
                for(size_t t=0;t<kr;++t) sum+=unsigned(std::abs(int(qa[i*kp+t])));
                row_bound=std::max(row_bound,sum);
            }
            for(int v:qb) bpeak=std::max(bpeak,unsigned(std::abs(v)));
            const unsigned bound=row_bound*bpeak;
            // Hardware floors and saturates each 16-wide band, then sums
            // with saturation. Reserve one negative rounding LSB per band.
            const uint32_t scale=bound ? std::min(65535u,unsigned((127-bands)*65536)/bound):65535u;
            if(!scale) throw std::overflow_error("Output multiplier underflow");
            timer.end(MatmulSocProfile::scale_bound);
            if(io.read(0x50)&1) throw std::runtime_error("matmul-soc busy; reset hardware before retry");
            io.write(0x10,unsigned(mp));io.write(0x20,unsigned(kp));
            io.write(0x30,unsigned(np));io.write(0x40,unsigned(np));
            io.write(0x60,scale);io.write(0x70,0);io.write(0x80,unsigned(np));
            timer.end(MatmulSocProfile::configure);
            io.write_buffer(0x1000,qa.data(),qa.size());
            io.write_buffer(0x5000,qb.data(),qb.size());
            timer.end(MatmulSocProfile::upload);
            (void)io.read(0x50);io.write(0,1);
            timer.end(MatmulSocProfile::launch);
            const auto deadline=std::chrono::steady_clock::now()+timeout;
            for(;;) {
                const auto status=io.read(0x50);
                if(profile) ++profile->compute_polls;
                if((status&3)==2) break;
                if(std::chrono::steady_clock::now()>=deadline)
                    throw std::runtime_error("matmul-soc timed out; hardware reset required (no abort register)");
            }
            timer.end(MatmulSocProfile::completion_wait);
            ++stats.submissions;stats.operand_bytes+=qa.size()+qb.size();stats.result_bytes+=output.size();
            io.read_buffer(0x9000,output.data(),output.size());
            timer.end(MatmulSocProfile::download);
            const double dequant=a.scale*b.scale*65536.0/scale;
            for(size_t i=0;i<mr;++i) for(size_t j=0;j<nc;++j)
                result[(r+i)*n+col+j]+=double(output[i*np+j])*dequant;
            timer.end(MatmulSocProfile::dequantize);
        }
    std::vector<float> staged(cc);
    for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j) {
        const double v=result[i*n+j]+(bias ? bias[j]:0.0);
        if(!std::isfinite(v) || std::abs(v)>std::numeric_limits<float>::max()) throw std::overflow_error("FP32 output overflow");
        staged[tc ? j*m+i:i*n+j]=float(v);
    }
    std::copy(staged.begin(),staged.end(),c);
    timer.end(MatmulSocProfile::output_convert);
    return stats;
}
MatmulSocStats matmul_soc_gemm_int8(MatmulSocIo& io,const MatmulSocOperand& a,
    const MatmulSocOperand& b,int8_t* c,const MatmulSocInt8Requantization& req,
    const MatmulSocPlan& plan,std::chrono::milliseconds timeout) {
    auto* profile=io.profile;ProfileSteps timer(profile);
    const size_t m=a.rows,n=b.cols,k=a.cols,cc=area(m,n);
    if(!c || a.cols!=b.rows || a.values.size()!=area(m,k) || b.values.size()!=area(k,n) ||
       req.multiplier.size()!=n || req.bias_q16.size()!=n || req.minimum< -128 || req.maximum>127 ||
       req.minimum>req.maximum || timeout.count()<=0 || plan.m!=m || plan.n!=n || plan.k!=k ||
       !fits(plan.tile_m,plan.tile_n,plan.tile_k)) throw std::invalid_argument("Invalid W8A8 GEMM");
    const size_t panels=(k-1)/plan.tile_k+1;
    const int64_t bias_limit=std::numeric_limits<int64_t>::max()/4;
    for(size_t j=0;j<n;++j) if(!std::isfinite(req.multiplier[j]) || req.multiplier[j]<=0 ||
        req.bias_q16[j]<-bias_limit || req.bias_q16[j]>bias_limit) throw std::invalid_argument("Invalid INT8 requantization");
    if(profile) {profile->tile_m=plan.tile_m;profile->tile_n=plan.tile_n;profile->tile_k=plan.tile_k;}
    std::vector<int64_t> sums(panels>1?cc:0,0);std::vector<int8_t> staged(cc);MatmulSocStats stats;
    timer.end(MatmulSocProfile::workspace);
    for(size_t r=0;r<m;r+=plan.tile_m) for(size_t col=0;col<n;col+=plan.tile_n)
        for(size_t p=0;p<k;p+=plan.tile_k) {
            const size_t mr=std::min(plan.tile_m,m-r),nc=std::min(plan.tile_n,n-col),kr=std::min(plan.tile_k,k-p);
            const size_t mp=padded(mr),np=padded(nc),kp=padded(kr),bands=kp/16;
            std::vector<int8_t> qa(mp*kp,0),qb(kp*np,0),out(mp*np);
            std::vector<int64_t> gains(nc);
            std::vector<int32_t> gains32(nc),bias32(nc);bool narrow=panels==1;
            for(size_t i=0;i<mr;++i) std::copy_n(a.values.begin()+(r+i)*k+p,kr,qa.begin()+i*kp);
            for(size_t t=0;t<kr;++t) std::copy_n(b.values.begin()+(p+t)*n+col,nc,qb.begin()+t*np);
            timer.end(MatmulSocProfile::pack_pad);
            unsigned row_bound=0,bpeak=0;
            for(size_t i=0;i<mr;++i) {unsigned sum=0;for(size_t t=0;t<kr;++t) sum+=unsigned(std::abs(int(qa[i*kp+t])));row_bound=std::max(row_bound,sum);}
            for(int v:qb) bpeak=std::max(bpeak,unsigned(std::abs(v)));
            const unsigned bound=row_bound*bpeak;
            const uint32_t scale=bound?std::min(65535u,unsigned((127-bands)*65536)/bound):65535u;
            if(!scale) throw std::overflow_error("INT8 output multiplier underflow");
            const double gain_limit=double(std::numeric_limits<int64_t>::max()/4/128/panels);
            for(size_t j=0;j<nc;++j) {
                const double gain=req.multiplier[col+j]*4294967296.0/scale;
                if(!std::isfinite(gain) || gain<0.5 || gain>gain_limit) throw std::overflow_error("INT8 fixed-point gain out of range");
                gains[j]=std::llround(gain);
                const auto bias=req.bias_q16[col+j];
                const uint64_t magnitude=uint64_t(gains[j])*128+uint64_t(bias<0?-bias:bias);
                if(magnitude>uint64_t(std::numeric_limits<int32_t>::max()-32768)) narrow=false;
                if(narrow) {gains32[j]=int32_t(gains[j]);bias32[j]=int32_t(bias);}
            }
            timer.end(MatmulSocProfile::scale_bound);
            if(io.read(0x50)&1) throw std::runtime_error("matmul-soc busy");
            io.write(0x10,unsigned(mp));io.write(0x20,unsigned(kp));io.write(0x30,unsigned(np));io.write(0x40,unsigned(np));
            io.write(0x60,scale);io.write(0x70,0);io.write(0x80,unsigned(np));timer.end(MatmulSocProfile::configure);
            io.write_buffer(0x1000,qa.data(),qa.size());io.write_buffer(0x5000,qb.data(),qb.size());timer.end(MatmulSocProfile::upload);
            (void)io.read(0x50);io.write(0,1);timer.end(MatmulSocProfile::launch);
            const auto deadline=std::chrono::steady_clock::now()+timeout;
            for(;;) {const auto status=io.read(0x50);if(profile) ++profile->compute_polls;
                if((status&3)==2) break;
                if(std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("INT8 matmul timed out; reset required");}
            timer.end(MatmulSocProfile::completion_wait);
            ++stats.submissions;stats.operand_bytes+=qa.size()+qb.size();stats.result_bytes+=out.size();
            io.read_buffer(0x9000,out.data(),out.size());timer.end(MatmulSocProfile::download);
            if(panels==1) {
                for(size_t i=0;i<mr;++i) {
                    if(narrow) {
                        requantize_int8(out.data()+i*np,staged.data()+(r+i)*n+col,gains32.data(),bias32.data(),nc,req.minimum,req.maximum);
#if defined(__ARM_NEON)
                        if(profile) profile->int8_vectorized_values+=nc/8*8;
#endif
                    }
                    else for(size_t j=0;j<nc;++j) {
                        const int64_t v=int64_t(out[i*np+j])*gains[j]+req.bias_q16[col+j];
                        const int64_t rounded=v>=0?(v+32768)>>16:-((-v+32768)>>16);
                        staged[(r+i)*n+col+j]=int8_t(std::max(int64_t(req.minimum),std::min(int64_t(req.maximum),rounded)));
                    }
                }
                timer.end(MatmulSocProfile::int8_requantize);
            } else {
                for(size_t i=0;i<mr;++i) for(size_t j=0;j<nc;++j) sums[(r+i)*n+col+j]+=int64_t(out[i*np+j])*gains[j];
                timer.end(MatmulSocProfile::int8_accumulate);
            }
        }
    if(panels>1) for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j) {
        const int64_t v=sums[i*n+j]+req.bias_q16[j];
        const int64_t rounded=v>=0?(v+32768)>>16:-((-v+32768)>>16);
        staged[i*n+j]=int8_t(std::max(int64_t(req.minimum),std::min(int64_t(req.maximum),rounded)));
    }
    std::copy(staged.begin(),staged.end(),c);timer.end(MatmulSocProfile::int8_requantize);
    return stats;
}
MatmulSocStats matmul_soc_gemm(MatmulSocIo& io,const float* a,const float* b,
    float* c,size_t m,size_t n,size_t k,bool ta,bool tb,bool tc,
    const float* bias,std::chrono::milliseconds timeout) {
    auto* profile=io.profile;
    const auto start=profile_now(profile);
    const auto qa=prepare_matmul_soc_operand(a,m,k,ta,profile);
    const auto qb=prepare_matmul_soc_operand(b,k,n,tb,profile);
    auto stats=matmul_soc_gemm_prepared(io,qa,qb,c,tc,bias,timeout);
    stats.quantized_values=qa.values.size()+qb.values.size();
    if(profile) {profile->jobs+=stats.submissions;profile->input_bytes+=stats.operand_bytes;
        profile->output_bytes+=stats.result_bytes;profile->quantized_values+=stats.quantized_values;}
    profile_add(profile,MatmulSocProfile::adapter_total,start);
    return stats;
}
}
