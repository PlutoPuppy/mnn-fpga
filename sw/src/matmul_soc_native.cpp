#include "matmul_soc_native.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fpga {
namespace {
size_t pad(size_t n){return (n+15)/16*16;}
void validate(const NativeInt8Weights& w) {
    if(!w.k || w.k>1024 || !w.n || w.n>4096 || w.kp!=pad(w.k) ||
       !w.tile_m || !w.tile_n || w.tile_m%16 || w.tile_n%16 || w.tile_n>128 ||
       (w.operand_b_bytes!=16384 && w.operand_b_bytes!=65536) ||
       w.tile_m*w.kp>16384 || w.tile_n*w.kp>w.operand_b_bytes || w.tile_m*w.tile_n>65536 ||
       w.weights.size()!=(w.n+w.tile_n-1)/w.tile_n || w.parameters.size()!=w.weights.size())
        throw std::invalid_argument("Invalid prepared native matrix plan");
    for(size_t tile=0;tile<w.weights.size();++tile) {
        const auto np=pad(std::min(w.tile_n,w.n-tile*w.tile_n));
        if(w.weights[tile].size!=w.kp*np || w.parameters[tile].size()!=np*8)
            throw std::invalid_argument("Invalid prepared native matrix image");
    }
}
void abi(MatmulSocIo& io) {
    if(io.read(0x90)!=native_int8_abi)throw std::runtime_error("NI82 native INT32 bitstream required");
    if(io.read(0x50)&1)throw std::runtime_error("Native accelerator already busy");
}
void parameters(MatmulSocIo& io,const std::vector<int8_t>& values) {
    for(size_t i=0;i<values.size();i+=16) {
        MatmulSocIo::Beat beat{};std::copy_n(values.data()+i,16,beat.begin());
        io.write_beat(0x19000+unsigned(i),beat);
    }
}
void config(MatmulSocIo& io,size_t m,size_t k,size_t n,size_t pitch,unsigned tile,unsigned mode) {
    const auto start=profile_now(io.profile);
    io.write(0x10,unsigned(m));io.write(0x20,unsigned(k));
    io.write(0x30,unsigned(n));io.write(0x40,unsigned(n));
    io.write(0x70,tile);io.write(0x80,unsigned(pitch));io.write(0xb0,mode);
    profile_add(io.profile,MatmulSocProfile::configure,start);
}
void execute(MatmulSocIo& io,MatmulSocStats& stats) {
    ProfileSteps timer(io.profile);
    io.write(0,1);timer.end(MatmulSocProfile::launch);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
    for(;;) {
        const auto s=io.read(0x50);if(io.profile)++io.profile->compute_polls;
        if((s&3)==2)break;
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("Native compute timed out; reset required");
    }
    timer.end(MatmulSocProfile::completion_wait);++stats.submissions;
}
void load(MatmulSocIo& io,const NativeInt8Weights& w,size_t tile,MatmulSocStats& stats) {
    ProfileSteps timer(io.profile);parameters(io,w.parameters.at(tile));
    stats.operand_bytes+=w.parameters[tile].size();
    timer.end(MatmulSocProfile::configure);
    io.write_cached(0x5000,w.weights.at(tile));stats.operand_bytes+=w.weights[tile].size;
    timer.end(MatmulSocProfile::upload);
}
}
NativeInt8Weights prepare_native_int8(MatmulSocIo& io,const int8_t* oi,size_t k,size_t n,
    const MatmulSocInt8Requantization& q) {
    if(!oi || !k || k>1024 || !n || n>4096 || q.multiplier.size()!=n || q.bias_q16.size()!=n ||
       q.minimum!=-127 || q.maximum!=127)throw std::invalid_argument("Unsupported native INT8 parameters");
    abi(io);
    NativeInt8Weights w;w.k=k;w.n=n;w.kp=pad(k);
    const auto format=io.read(0xa0);
    if((format&0x00ffffffu)!=0x00181080u || (format&0xfe000000u))
        throw std::invalid_argument("Unsupported NI82 format/capacity word");
    w.operand_b_bytes=(format&0x01000000)?65536:16384;
    w.tile_n=std::min({(w.operand_b_bytes/w.kp/16)*16,size_t(128),pad(n)});
    w.tile_m=std::min((16384/w.kp/16)*16,(65536/w.tile_n/16)*16);
    if(!w.tile_m)throw std::invalid_argument("Reduction exceeds operand capacity");
    for(size_t col=0;col<n;col+=w.tile_n) {
        const size_t nc=std::min(w.tile_n,n-col),np=pad(nc);
        std::vector<int8_t> b(w.kp*np,0),p(np*8,0);
        for(size_t j=0;j<nc;++j) {
            const auto jn=col+j;
            if(!std::isfinite(q.multiplier[jn]) || q.multiplier[jn]<=0 || q.multiplier[jn]>=1 ||
               q.bias_q16[jn]<std::numeric_limits<int32_t>::min() || q.bias_q16[jn]>std::numeric_limits<int32_t>::max())
                throw std::invalid_argument("Native coefficient/bias outside supported range");
            const uint32_t c=uint32_t(std::llround(q.multiplier[jn]*16777216.0));
            if(!c || c>=16777216)throw std::invalid_argument("Native Q24 coefficient out of range");
            const int32_t bias=int32_t(q.bias_q16[jn]);
            std::memcpy(p.data()+j*8,&c,4);std::memcpy(p.data()+j*8+4,&bias,4);
            for(size_t t=0;t<k;++t)b[t*np+j]=oi[jn*k+t];
        }
        w.weights.push_back(io.cache_buffer(b.data(),b.size()));w.parameters.push_back(std::move(p));
    }
    return w;
}
MatmulSocStats native_int8_linear(MatmulSocIo& io,const NativeInt8Weights& w,const int8_t* a,int8_t* c,size_t rows) {
    if(!rows || !a || !c || !w.k || !w.n)throw std::invalid_argument("Invalid native linear tensors");
    validate(w);abi(io);MatmulSocStats stats;
    for(size_t col=0,tile=0;col<w.n;col+=w.tile_n,++tile) {
        const size_t nc=std::min(w.tile_n,w.n-col),np=pad(nc);
        config(io,pad(std::min(w.tile_m,rows)),w.kp,np,np,0,0);
        load(io,w,tile,stats);
        for(size_t r=0;r<rows;r+=w.tile_m) {
            const size_t mr=std::min(w.tile_m,rows-r),mp=pad(mr);ProfileSteps timer(io.profile);
            config(io,mp,w.kp,np,np,0,0);timer.reset();
            io.write_nc4(0x1000,a,rows,w.k,r,mr,mp,w.kp);stats.operand_bytes+=mp*w.kp;
            timer.end(MatmulSocProfile::upload);execute(io,stats);
            timer.reset();io.read_nc4(0x9000,c,rows,w.n,r,col,mr,nc,mp,np);
            stats.result_bytes+=mp*np;timer.end(MatmulSocProfile::download);
        }
    }
    return stats;
}
NativePairStats native_int8_pair(MatmulSocIo& io,const NativeInt8Weights& f,const NativeInt8Weights& s,
    const std::array<int8_t,256>& lut,const int8_t* a,int8_t* c,size_t rows) {
    if(!a || !c || !rows || f.n!=s.k || f.n%16 || s.tile_m*s.kp>16384)
        throw std::invalid_argument("Unsupported resident MLP dimensions");
    validate(f);validate(s);abi(io);
    const size_t rm=std::min(f.tile_m,(61440/f.n/16)*16);
    if(!rm)throw std::invalid_argument("Hidden tensor exceeds FPGA capacity");
    for(size_t col=0;col<f.n;col+=f.tile_n)
        if(col%pad(std::min(f.tile_n,f.n-col)))throw std::invalid_argument("Unaligned resident FC1 column placement");
    for(size_t col=0;col<s.n;col+=s.tile_n)
        if(3840%pad(std::min(s.tile_n,s.n-col)))throw std::invalid_argument("Unsupported resident scratch pitch");
    const auto lut_start=profile_now(io.profile);
    for(unsigned i=0;i<256;i+=16) {MatmulSocIo::Beat beat{};std::copy_n(lut.data()+i,16,beat.begin());io.write_beat(0x19400+i,beat);}
    profile_add(io.profile,MatmulSocProfile::configure,lut_start);
    NativePairStats stats;
    stats.operand_bytes+=256;
    for(size_t row=0;row<rows;row+=rm) {
        const size_t mr=std::min(rm,rows-row),mp=pad(mr);ProfileSteps timer(io.profile);
        config(io,mp,f.kp,pad(std::min(f.tile_n,f.n)),f.n,0,1);
        timer.reset();
        io.write_nc4(0x1000,a,rows,f.k,row,mr,mp,f.kp);stats.operand_bytes+=mp*f.kp;
        timer.end(MatmulSocProfile::upload);
        for(size_t col=0,tile=0;col<f.n;col+=f.tile_n,++tile) {
            const size_t np=pad(std::min(f.tile_n,f.n-col));
            config(io,mp,f.kp,np,f.n,unsigned(col/np)<<15,1);load(io,f,tile,stats);execute(io,stats);
        }
        for(size_t col=0,tile=0;col<s.n;col+=s.tile_n,++tile) {
            const size_t nc=std::min(s.tile_n,s.n-col),np=pad(nc);
            const size_t row_tile=std::min(s.tile_m,(4096/np/16)*16);
            config(io,pad(std::min(row_tile,mr)),s.kp,np,np,0,2);load(io,s,tile,stats);
            for(size_t rr=0;rr<mr;rr+=row_tile) {
                const size_t count=std::min(row_tile,mr-rr),padded_rows=pad(count);
                timer.reset();io.write(0x80,unsigned(f.n));timer.end(MatmulSocProfile::configure);
                timer.reset();io.feedback(0x9000+unsigned(rr*f.n),0x1000,padded_rows*s.kp);
                stats.local_bytes+=padded_rows*s.kp;timer.end(MatmulSocProfile::local_feedback);
                config(io,padded_rows,s.kp,np,np,0,2);execute(io,stats);
                timer.reset();io.read_nc4(0x18000,c,rows,s.n,row+rr,col,count,nc,padded_rows,np);
                stats.result_bytes+=padded_rows*np;timer.end(MatmulSocProfile::download);
            }
        }
        stats.hidden_bytes_avoided+=2*mp*f.n;
    }
    const auto end_start=profile_now(io.profile);io.write(0xb0,0);
    profile_add(io.profile,MatmulSocProfile::configure,end_start);return stats;
}
}
