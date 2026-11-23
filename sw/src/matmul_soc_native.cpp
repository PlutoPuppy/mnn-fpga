#include "matmul_soc_native.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fpga {
namespace {
struct OutputReads {
    MatmulSocIo& io;
    ~OutputReads() { io.finish_output_reads(); }
};
size_t pad(size_t n){return (n+15)/16*16;}
void validate(const NativeInt8Weights& w) {
    if(!w.k || w.k>w.operand_a_bytes/16 || !w.n || w.n>4096 || w.kp!=pad(w.k) ||
       !w.tile_m || !w.tile_n || w.tile_m%16 || w.tile_n%16 || w.tile_n>128 ||
       (w.operand_b_bytes!=16384 && w.operand_b_bytes!=65536) ||
       (w.operand_a_bytes!=16384 && w.operand_a_bytes!=32768) ||
       w.operand_b_offset!=(w.operand_a_bytes==32768?0x9000u:0x5000u) ||
       w.tile_m*w.kp>w.operand_a_bytes || w.tile_n*w.kp>w.operand_b_bytes || w.tile_m*w.tile_n>65536 ||
       w.weights.size()!=(w.n+w.tile_n-1)/w.tile_n || w.parameters.size()!=w.weights.size())
        throw std::invalid_argument("Invalid prepared native matrix plan");
    for(size_t tile=0;tile<w.weights.size();++tile) {
        const auto np=pad(std::min(w.tile_n,w.n-tile*w.tile_n));
        if(w.weights[tile].size!=w.kp*np || w.parameters[tile].size()!=np*8)
            throw std::invalid_argument("Invalid prepared native matrix image");
    }
}
uint32_t abi(MatmulSocIo& io) {
    const auto value=io.read(0x90);
    if(value!=native_int8_abi && value!=native_int8_extended_abi)throw std::runtime_error("NI82/NI83 native INT32 bitstream required");
    if(io.read(0x50)&1)throw std::runtime_error("Native accelerator already busy");
    return value;
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
    unsigned polls=0;
    for(;;) {
        const auto s=io.read(0x50);if(io.profile)++io.profile->compute_polls;
        if((s&3)==2)break;
        if((!io.fast_poll || (++polls&63u)==0) && std::chrono::steady_clock::now()>=deadline)
            throw std::runtime_error("Native compute timed out; reset required");
    }
    timer.end(MatmulSocProfile::completion_wait);++stats.submissions;
}
void load(MatmulSocIo& io,const NativeInt8Weights& w,size_t tile,MatmulSocStats& stats) {
    ProfileSteps timer(io.profile);parameters(io,w.parameters.at(tile));
    stats.operand_bytes+=w.parameters[tile].size();
    timer.end(MatmulSocProfile::configure);
    io.write_cached(unsigned(w.operand_b_offset),w.weights.at(tile));stats.operand_bytes+=w.weights[tile].size;
    timer.end(MatmulSocProfile::upload);
}
}
void set_native_int8_parameters(NativeInt8Weights& w,const MatmulSocInt8Requantization& q) {
    if(!w.n || !w.tile_n || q.multiplier.size()!=w.n || q.bias_q16.size()!=w.n ||
       q.minimum!=-127 || q.maximum!=127)throw std::invalid_argument("Unsupported native INT8 epilogue");
    w.parameters.resize((w.n-1)/w.tile_n+1);
    for(size_t col=0,tile=0;col<w.n;col+=w.tile_n,++tile) {
        const size_t nc=std::min(w.tile_n,w.n-col);
        auto& p=w.parameters[tile];p.resize(pad(nc)*8);
        for(size_t j=0;j<nc;++j) {
            const auto jn=col+j;
            if(!std::isfinite(q.multiplier[jn]) || q.multiplier[jn]<=0 || q.multiplier[jn]>=1 ||
               q.bias_q16[jn]<std::numeric_limits<int32_t>::min() || q.bias_q16[jn]>std::numeric_limits<int32_t>::max())
                throw std::invalid_argument("Native coefficient/bias outside supported range");
            const uint32_t c=uint32_t(std::llround(q.multiplier[jn]*16777216.0));
            if(!c || c>=16777216)throw std::invalid_argument("Native Q24 coefficient out of range");
            const int32_t bias=int32_t(q.bias_q16[jn]);
            std::memcpy(p.data()+j*8,&c,4);std::memcpy(p.data()+j*8+4,&bias,4);
        }
    }
}
NativeInt8Weights prepare_native_int8(MatmulSocIo& io,const int8_t* oi,size_t k,size_t n,
    const MatmulSocInt8Requantization& q) {
    if(!oi || !k || k>2048 || !n || n>4096 || q.multiplier.size()!=n || q.bias_q16.size()!=n ||
       q.minimum!=-127 || q.maximum!=127)throw std::invalid_argument("Unsupported native INT8 parameters");
    const auto version=abi(io);
    NativeInt8Weights w;w.k=k;w.n=n;w.kp=pad(k);
    const auto format=io.read(0xa0);
    if((version==native_int8_extended_abi && format!=0x03181080u) ||
       (version==native_int8_abi && (format&0xfeffffffu)!=0x00181080u))
        throw std::invalid_argument("Unsupported NI82 format/capacity word");
    if(version==native_int8_extended_abi){w.operand_a_bytes=32768;w.operand_b_offset=0x9000;}
    w.operand_b_bytes=(format&0x01000000)?65536:16384;
    if(w.kp*16>w.operand_a_bytes)throw std::invalid_argument("Reduction exceeds native A capacity");
    w.tile_n=std::min({(w.operand_b_bytes/w.kp/16)*16,size_t(128),pad(n)});
    w.tile_m=std::min((w.operand_a_bytes/w.kp/16)*16,(65536/w.tile_n/16)*16);
    if(!w.tile_m)throw std::invalid_argument("Reduction exceeds operand capacity");
    set_native_int8_parameters(w,q);
    for(size_t col=0;col<n;col+=w.tile_n) {
        const size_t nc=std::min(w.tile_n,n-col),np=pad(nc);
        std::vector<int8_t> b(w.kp*np,0);
        for(size_t j=0;j<nc;++j) {
            const auto jn=col+j;
            for(size_t t=0;t<k;++t)b[t*np+j]=oi[jn*k+t];
        }
        w.weights.push_back(io.cache_buffer(b.data(),b.size()));
    }
    return w;
}
MatmulSocStats native_int8_linear(MatmulSocIo& io,const NativeInt8Weights& w,const int8_t* a,int8_t* c,size_t rows,bool row_major) {
    if(!rows || !a || !c || !w.k || !w.n || (row_major && (w.k%16 || w.n%16)))
        throw std::invalid_argument("Invalid native linear tensors");
    validate(w);
    if(abi(io)!=(w.operand_a_bytes==32768?native_int8_extended_abi:native_int8_abi))throw std::runtime_error("Native prepared ABI changed");
    const OutputReads drain{io};
    MatmulSocStats stats;
    const size_t row_tiles=(rows-1)/w.tile_m+1;
    const size_t column_tiles=w.weights.size();
    const uint64_t a_bytes=uint64_t(io.transfer_live_rows ? rows : pad(rows))*w.kp;
    // A is transposed and copied through noncached DDR at runtime. B is
    // already packed and staged. Count one extra unit per A byte for this
    // host work, and include the B tile's coefficient payload.
    const uint64_t a_cost=2*a_bytes;
    const uint64_t b_cost=uint64_t(pad(w.n))*(w.kp+8);
    auto upload_a=[&](size_t r,size_t mr,size_t mp) {
        ProfileSteps timer(io.profile);
        if(row_major)io.write_rows(0x1000,a+r*w.k,mr,w.k,mp);
        else io.write_nc4(0x1000,a,rows,w.k,r,mr,mp,w.kp);
        stats.operand_bytes+=(io.transfer_live_rows ? mr : mp)*w.kp;
        timer.end(MatmulSocProfile::upload);
    };
    auto run_tile=[&](size_t r,size_t mr,size_t mp,size_t col,size_t nc,size_t np) {
        execute(io,stats);
        ProfileSteps timer(io.profile);
        if(row_major)io.read_rows(0x9000,c+r*w.n+col,mr,nc,w.n,mp);
        else io.read_nc4(0x9000,c,rows,w.n,r,col,mr,nc,mp,np);
        stats.result_bytes+=(io.transfer_live_rows ? mr : mp)*np;
        timer.end(MatmulSocProfile::download);
    };
    if(io.reuse_native_activations && column_tiles>1) {
        // Reversing direction after each row/column keeps one more operand
        // tile resident at the turn. Start at the tail to favor retaining
        // the larger endpoint on the first turn.
        auto saved_at_turns=[](size_t turns,uint64_t full,uint64_t tail) {
            return (turns/2)*(full+tail)+(turns%2)*full;
        };
        const size_t last_m=rows-(row_tiles-1)*w.tile_m;
        const size_t last_n=pad(w.n-(column_tiles-1)*w.tile_n);
        const uint64_t row_tail=io.transfer_live_rows ? last_m : pad(last_m);
        const auto row_first=a_cost+b_cost*row_tiles-
            saved_at_turns(row_tiles-1,w.tile_n*(w.kp+8),last_n*(w.kp+8));
        const auto column_first=b_cost+a_cost*column_tiles-
            saved_at_turns(column_tiles-1,2*w.tile_m*w.kp,2*row_tail*w.kp);
        const bool keep_a=row_tiles==1 || row_first<column_first;
        const size_t outer=keep_a ? row_tiles : column_tiles;
        const size_t inner=keep_a ? column_tiles : row_tiles;
        size_t last_row=rows,last_tile=column_tiles,previous_mp=0,previous_np=0;
        for(size_t i=0;i<outer;++i) {
            const bool reverse=outer>1 && !(i%2);
            for(size_t j=0;j<inner;++j) {
                const size_t index=reverse ? inner-1-j : j;
                const size_t r=(keep_a ? i : index)*w.tile_m;
                const size_t tile=keep_a ? index : i;
                const size_t col=tile*w.tile_n;
                const size_t mr=std::min(w.tile_m,rows-r),mp=pad(mr);
                const size_t nc=std::min(w.tile_n,w.n-col),np=pad(nc);
                if(mp!=previous_mp || np!=previous_np) config(io,mp,w.kp,np,np,0,0);
                if(r!=last_row) upload_a(r,mr,mp);
                if(tile!=last_tile) load(io,w,tile,stats);
                run_tile(r,mr,mp,col,nc,np);
                last_row=r;
                last_tile=tile;
                previous_mp=mp;
                previous_np=np;
            }
        }
        return stats;
    }
    for(size_t col=0,tile=0;col<w.n;col+=w.tile_n,++tile) {
        const size_t nc=std::min(w.tile_n,w.n-col),np=pad(nc);
        config(io,pad(std::min(w.tile_m,rows)),w.kp,np,np,0,0);
        load(io,w,tile,stats);
        for(size_t r=0;r<rows;r+=w.tile_m) {
            const size_t mr=std::min(w.tile_m,rows-r),mp=pad(mr);
            if(r && mp!=pad(std::min(w.tile_m,rows))) config(io,mp,w.kp,np,np,0,0);
            upload_a(r,mr,mp);
            run_tile(r,mr,mp,col,nc,np);
        }
    }
    return stats;
}
NativePairStats native_int8_pair(MatmulSocIo& io,const NativeInt8Weights& f,const NativeInt8Weights& s,
    const std::array<int8_t,256>& lut,const int8_t* a,int8_t* c,size_t rows) {
    if(!a || !c || !rows || f.n!=s.k || f.n%16 || s.tile_m*s.kp>s.operand_a_bytes ||
       f.operand_a_bytes!=s.operand_a_bytes || f.operand_b_offset!=s.operand_b_offset)
        throw std::invalid_argument("Unsupported resident MLP dimensions");
    validate(f);validate(s);
    if(abi(io)!=(f.operand_a_bytes==32768?native_int8_extended_abi:native_int8_abi))throw std::runtime_error("Native pair ABI changed");
    const size_t rm=std::min(f.tile_m,(61440/f.n/16)*16);
    if(!rm)throw std::invalid_argument("Hidden tensor exceeds FPGA capacity");
    for(size_t col=0;col<f.n;col+=f.tile_n)
        if(col%pad(std::min(f.tile_n,f.n-col)))throw std::invalid_argument("Unaligned resident FC1 column placement");
    for(size_t col=0;col<s.n;col+=s.tile_n)
        if(3840%pad(std::min(s.tile_n,s.n-col)))throw std::invalid_argument("Unsupported resident scratch pitch");
    const OutputReads drain{io};
    const auto lut_start=profile_now(io.profile);
    for(unsigned i=0;i<256;i+=16) {MatmulSocIo::Beat beat{};std::copy_n(lut.data()+i,16,beat.begin());io.write_beat(0x19400+i,beat);}
    profile_add(io.profile,MatmulSocProfile::configure,lut_start);
    NativePairStats stats;
    stats.operand_bytes+=256;
    for(size_t row=0;row<rows;row+=rm) {
        const size_t mr=std::min(rm,rows-row),mp=pad(mr);ProfileSteps timer(io.profile);
        config(io,mp,f.kp,pad(std::min(f.tile_n,f.n)),f.n,0,1);
        timer.reset();
        io.write_nc4(0x1000,a,rows,f.k,row,mr,mp,f.kp);
        stats.operand_bytes+=(io.transfer_live_rows ? mr : mp)*f.kp;
        timer.end(MatmulSocProfile::upload);
        for(size_t col=0,tile=0;col<f.n;col+=f.tile_n,++tile) {
            const size_t np=pad(std::min(f.tile_n,f.n-col));
            config(io,mp,f.kp,np,f.n,unsigned(col/np)<<15,1);load(io,f,tile,stats);execute(io,stats);
        }
        const size_t second_tiles=s.weights.size();
        size_t previous_feedback_row=mr,previous_feedback_count=0;
        for(size_t i=0;i<second_tiles;++i) {
            // A retains the FC2 input across weight loads. Start with the
            // narrower output tile (its scratch allows more rows), then
            // alternate row traversal to reuse feedback at column turns.
            const size_t tile=io.reuse_native_activations ? second_tiles-1-i : i;
            const size_t col=tile*s.tile_n;
            const size_t nc=std::min(s.tile_n,s.n-col),np=pad(nc);
            const size_t row_tile=std::min(s.tile_m,(4096/np/16)*16);
            config(io,pad(std::min(row_tile,mr)),s.kp,np,np,0,2);load(io,s,tile,stats);
            const size_t subtiles=(mr-1)/row_tile+1;
            const bool reverse=io.reuse_native_activations && second_tiles>1 && !(i%2);
            for(size_t j=0;j<subtiles;++j) {
                const size_t rr=(reverse ? subtiles-1-j : j)*row_tile;
                const size_t count=std::min(row_tile,mr-rr),padded_rows=pad(count);
                const size_t transferred_rows=io.transfer_live_rows ? count : padded_rows;
                if(!io.reuse_native_activations || rr!=previous_feedback_row || transferred_rows>previous_feedback_count) {
                    timer.reset();io.write(0x80,unsigned(f.n));timer.end(MatmulSocProfile::configure);
                    timer.reset();io.feedback(0x9000+unsigned(rr*f.n),0x1000,transferred_rows*s.kp);
                    stats.local_bytes+=transferred_rows*s.kp;timer.end(MatmulSocProfile::local_feedback);
                    previous_feedback_row=rr;
                    previous_feedback_count=transferred_rows;
                }
                config(io,padded_rows,s.kp,np,np,0,2);execute(io,stats);
                timer.reset();io.read_nc4(0x18000,c,rows,s.n,row+rr,col,count,nc,padded_rows,np);
                stats.result_bytes+=transferred_rows*np;timer.end(MatmulSocProfile::download);
            }
        }
        stats.hidden_bytes_avoided+=2*(io.transfer_live_rows ? mr : mp)*f.n;
    }
    const auto end_start=profile_now(io.profile);io.write(0xb0,0);
    profile_add(io.profile,MatmulSocProfile::configure,end_start);return stats;
}
}
