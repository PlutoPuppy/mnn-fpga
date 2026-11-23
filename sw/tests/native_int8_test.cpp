#include "matmul_soc_native.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int8_t quant(int64_t acc,uint32_t coefficient,int32_t bias) {
    const int64_t scaled=acc*coefficient+int64_t(bias)*256;
    const auto rounded=scaled<0 ? -((-scaled+8388608)/16777216):(scaled+8388608)/16777216;
    return int8_t(std::clamp(rounded,int64_t(-127),int64_t(127)));
}
struct Mock:fpga::MatmulSocIo {
    std::map<unsigned,uint32_t> regs;
    std::vector<int8_t> a=std::vector<int8_t>(32768),b=std::vector<int8_t>(65536),c=std::vector<int8_t>(65536);
    std::array<int8_t,256> lut{};
    std::array<uint32_t,128> coefficient{};
    std::array<int32_t,128> bias{};
    size_t weight_loads=0,feedback_bytes=0;
    bool valid_abi=true;
    bool large_b=false,large_a=false;
    unsigned index(unsigned offset) {
        unsigned row=offset/regs[0x80],col=offset%regs[0x80];
        return row%16*4096+row/16*regs[0x80]+col;
    }
    uint32_t read(unsigned off) override {return off==0x90 ? (valid_abi?(large_a?fpga::native_int8_extended_abi:fpga::native_int8_abi):0):off==0x50?2:
        off==0xa0?(0x00181080u|(large_b?0x01000000u:0)|(large_a?0x02000000u:0)):regs[off];}
    void write(unsigned off,uint32_t value) override {
        regs[off]=value;if(off || value!=1)return;
        const unsigned m=regs[0x10],k=regs[0x20],n=regs[0x30],pitch=regs[0x80];
        require(m*k<=(large_a?32768u:16384u) && k*n<=(large_b?65536u:16384u) && n<=128 && m*n<=65536,"Invalid hardware capacity");
        for(unsigned i=0;i<m;++i)for(unsigned j=0;j<n;++j) {
            int64_t sum=0;for(unsigned t=0;t<k;++t)
                sum+=int(a[i%16*(large_a?2048:1024)+i/16*k+t])*int(b[t%16*(large_b?4096:1024)+t/16*n+j]);
            auto v=quant(sum,coefficient[j],bias[j]);if(regs[0xb0]&1)v=lut[uint8_t(v)];
            const unsigned target=(regs[0xb0]&2)?i%16*4096+3840+i/16*n+j:
                i%16*4096+(i/16)*pitch+((regs[0x70]>>15)&32767)*n+j;
            require(target<c.size(),"Canvas overflow");c[target]=v;
        }
    }
    void write_beat(unsigned off,const Beat& data) override {
        if(off>=0x19400){std::copy(data.begin(),data.end(),lut.begin()+off-0x19400);return;}
        if(off>=0x19000) {
            unsigned j=(off-0x19000)/8;
            std::memcpy(&coefficient[j],data.data(),4);std::memcpy(&bias[j],data.data()+4,4);
            std::memcpy(&coefficient[j+1],data.data()+8,4);std::memcpy(&bias[j+1],data.data()+12,4);return;
        }
        const bool first=off<(large_a?0x9000u:0x5000u);auto& dest=first?a:b;
        const unsigned start=off-(first?0x1000:(large_a?0x9000:0x5000)),pitch=regs[first?0x20:0x30];
        require(pitch>0,"Operand dimensions must be configured before upload");
        for(unsigned i=0;i<16;++i) {
            const auto row=(start+i)/pitch,col=(start+i)%pitch;
            dest.at(row%16*(first?(large_a?2048:1024):(large_b?4096:1024))+row/16*pitch+col)=data[i];
        }
    }
    Beat read_beat(unsigned off) override {Beat v{};for(unsigned i=0;i<16;++i)v[i]=c.at(index(off-0x9000+i));return v;}
    void write_cached(unsigned off,const CachedBuffer& value) override {
        require(value.size==regs[0x20]*regs[0x30],"Cached operand uploaded with stale dimensions");
        ++weight_loads;MatmulSocIo::write_cached(off,value);
    }
    void feedback(unsigned source,unsigned destination,size_t bytes) override {
        require(destination==0x1000,"Wrong feedback destination");feedback_bytes+=bytes;
        const auto pitch=regs[0x20];require(pitch>0,"Feedback dimensions must be configured before DMA");
        for(unsigned i=0;i<bytes;++i) {
            const auto row=i/pitch,col=i%pitch;
            a.at(row%16*(large_a?2048:1024)+row/16*pitch+col)=c.at(index(source-0x9000+i));
        }
    }
};
std::vector<int8_t> weights(size_t k,size_t n) {
    std::vector<int8_t> x(k*n);
    for(size_t i=0;i<x.size();++i)x[i]=int8_t(int((i*1337+59)%17)-8);
    return x;
}
fpga::MatmulSocInt8Requantization requant(size_t n) {
    fpga::MatmulSocInt8Requantization q;q.multiplier.resize(n);q.bias_q16.resize(n);
    for(size_t j=0;j<n;++j){q.multiplier[j]=double(1+j%3)/256;q.bias_q16[j]=(int(j%5)-2)*32768;}
    return q;
}
std::vector<int8_t> pack(const std::vector<int8_t>& x,size_t m,size_t k) {
    std::vector<int8_t> y(m*((k+3)/4)*4,99);
    for(size_t i=0;i<m;++i)for(size_t j=0;j<k;++j)y[(j/4*m+i)*4+j%4]=x[i*k+j];
    return y;
}
std::vector<int8_t> reference(const std::vector<int8_t>& a,const std::vector<int8_t>& w,size_t m,size_t k,size_t n) {
    auto q=requant(n);std::vector<int8_t> out(m*n);
    for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j){
        int64_t acc=0;for(size_t t=0;t<k;++t)acc+=int(a[i*k+t])*int(w[j*k+t]);
        out[i*n+j]=quant(acc,uint32_t(q.multiplier[j]*16777216),int32_t(q.bias_q16[j]));
    }
    return out;
}
}
int main() {
    try {
        size_t cases=0;
        for(bool large:{false,true})for(size_t k:{size_t(17),size_t(128),size_t(160),size_t(512),size_t(640),size_t(1024)})
            for(size_t n:{size_t(19),size_t(128),size_t(160)}) {
                const size_t m=37;Mock io;io.large_b=large;auto w=weights(k,n),a=weights(k,m),ai=pack(a,m,k);
                auto prepared=fpga::prepare_native_int8(io,w.data(),k,n,requant(n));
                std::vector<int8_t> output(m*((n+3)/4)*4,99);
                auto stats=fpga::native_int8_linear(io,prepared,ai.data(),output.data(),m);
                require(output==pack(reference(a,w,m,k,n),m,n),"Linear oracle mismatch");
                require(io.weight_loads==(n+prepared.tile_n-1)/prepared.tile_n,"Weights reloaded for every M tile");
                require(stats.operand_bytes>=k*n,"Missing transfer accounting");++cases;
            }
        for(bool large:{false,true}) {
            const size_t m=1105,k=16,n=128;Mock io;io.large_b=large;
            auto a=weights(k,m),w=weights(k,n),ai=pack(a,m,k);std::vector<int8_t> output(m*n);
            auto prepared=fpga::prepare_native_int8(io,w.data(),k,n,requant(n));
            auto stats=fpga::native_int8_linear(io,prepared,ai.data(),output.data(),m);
            require(prepared.tile_m*prepared.tile_n<=65536 && stats.submissions==3,"Output capacity ignored for small K");
            require(output==pack(reference(a,w,m,k,n),m,n),"Large-M/small-K oracle mismatch");++cases;
        }
        for(bool large:{false,true})for(size_t k:{size_t(128),size_t(160)}) {
            const size_t m=135,h=k*4,n=k;Mock io;io.large_b=large;auto a=weights(k,m),w=weights(k,h),v=weights(h,n),ai=pack(a,m,k);
            auto first=fpga::prepare_native_int8(io,w.data(),k,h,requant(h));
            auto second=fpga::prepare_native_int8(io,v.data(),h,n,requant(n));
            std::array<int8_t,256> lut{};for(unsigned i=0;i<256;++i)lut[i]=int8_t(int(int8_t(i))/2);
            std::vector<int8_t> output(m*n);
            auto stats=fpga::native_int8_pair(io,first,second,lut,ai.data(),output.data(),m);
            auto hidden=reference(a,w,m,k,h);for(auto& x:hidden)x=lut[uint8_t(x)];
            require(output==pack(reference(hidden,v,m,h,n),m,n),"Resident pair oracle mismatch");
            require(stats.local_bytes==io.feedback_bytes && stats.hidden_bytes_avoided>=m*h*2,"Missing resident transfer accounting");++cases;
        }
        for(size_t k:{size_t(128),size_t(160)}) {
            const size_t m=k==128?784:196,h=k*4,n=k;Mock io;io.large_b=true;
            auto a=weights(k,m),w=weights(k,h),v=weights(h,n),ai=pack(a,m,k);
            auto first=fpga::prepare_native_int8(io,w.data(),k,h,requant(h));
            auto second=fpga::prepare_native_int8(io,v.data(),h,n,requant(n));
            std::array<int8_t,256> lut{};for(unsigned i=0;i<256;++i)lut[i]=int8_t(int(int8_t(i))/2);
            std::vector<int8_t> output(m*n);
            const auto stats=fpga::native_int8_pair(io,first,second,lut,ai.data(),output.data(),m);
            auto hidden=reference(a,w,m,k,h);for(auto& x:hidden)x=lut[uint8_t(x)];
            require(output==pack(reference(hidden,v,m,h,n),m,n),"TinyViT resident pair oracle mismatch");
            require(stats.submissions==(k==128?56u:41u),"Unexpected TinyViT resident job count");
            require(stats.operand_bytes==(k==128?1053952u:667136u),"Unexpected resident host upload payload");
            require(stats.result_bytes==(k==128?100352u:33280u),"Unexpected resident host download payload");
            require(stats.local_bytes==(k==128?401408u:266240u),"Unexpected resident local DMA payload");++cases;
        }
        for(size_t k:{size_t(1280),size_t(2048)})for(size_t n:{size_t(160),size_t(320)}) {
            const size_t m=37;Mock io;io.large_a=true;io.large_b=true;
            auto a=weights(k,m),w=weights(k,n),ai=pack(a,m,k);std::vector<int8_t> output(m*n);
            auto prepared=fpga::prepare_native_int8(io,w.data(),k,n,requant(n));
            const auto stats=fpga::native_int8_linear(io,prepared,ai.data(),output.data(),m);
            require(prepared.operand_a_bytes==32768 && prepared.operand_b_offset==0x9000,"NI83 map mismatch");
            require(stats.submissions>0 && output==pack(reference(a,w,m,k,n),m,n),"NI83 large-K oracle mismatch");++cases;
        }
        for(size_t m:{size_t(1),size_t(3136)}) {
            const size_t k=m==1?320:64,n=m==1?1000:256;Mock io;io.large_a=true;io.large_b=true;
            auto a=weights(k,m),w=weights(k,n),ai=pack(a,m,k);std::vector<int8_t> output(m*n);
            auto prepared=fpga::prepare_native_int8(io,w.data(),k,n,requant(n));
            fpga::native_int8_linear(io,prepared,ai.data(),output.data(),m);
            require(output==pack(reference(a,w,m,k,n),m,n),"Spatial/head oracle mismatch");++cases;
        }
        for(size_t m:{size_t(49),size_t(196),size_t(784)}) {
            const size_t k=160,n=480;
            Mock io;io.large_a=true;io.large_b=true;
            auto a=weights(k,m),w=weights(k,n),ai=pack(a,m,k);
            auto prepared=fpga::prepare_native_int8(io,w.data(),k,n,requant(n));
            std::vector<int8_t> baseline(m*n),reused(m*n);
            auto old_stats=fpga::native_int8_linear(io,prepared,ai.data(),baseline.data(),m);
            io.reuse_native_activations=true;
            auto stats=fpga::native_int8_linear(io,prepared,ai.data(),reused.data(),m);
            require(reused==baseline,"Resident activation scheduling changed output");
            require(stats.submissions==old_stats.submissions && stats.operand_bytes<old_stats.operand_bytes,
                    "Activation scheduling did not reduce operand traffic");
            ++cases;
            fpga::native_int8_linear(io,prepared,a.data(),reused.data(),m,true);
            require(reused==reference(a,w,m,k,n),"Row-major native matrix oracle mismatch");
            ++cases;
        }
        {
            const size_t m=49,k=320,h=1280,n=320;Mock io;io.large_a=true;io.large_b=true;
            auto a=weights(k,m),w=weights(k,h),v=weights(h,n),ai=pack(a,m,k);
            auto first=fpga::prepare_native_int8(io,w.data(),k,h,requant(h));
            auto second=fpga::prepare_native_int8(io,v.data(),h,n,requant(n));
            std::array<int8_t,256> lut{};for(unsigned i=0;i<256;++i)lut[i]=int8_t(int(int8_t(i))/2);
            std::vector<int8_t> output(m*n);const auto stats=fpga::native_int8_pair(io,first,second,lut,ai.data(),output.data(),m);
            auto hidden=reference(a,w,m,k,h);for(auto& value:hidden)value=lut[uint8_t(value)];
            require(stats.submissions==48 && output==pack(reference(hidden,v,m,h,n),m,n),"Stage3 resident pair oracle mismatch");++cases;
            io.reuse_native_activations=true;
            std::vector<int8_t> reused(m*n);
            const auto reduced=fpga::native_int8_pair(io,first,second,lut,ai.data(),reused.data(),m);
            require(reused==output && reduced.submissions==stats.submissions && reduced.local_bytes<stats.local_bytes,
                    "Resident FC2 feedback reuse failed");
            ++cases;
        }
        Mock bad;bad.valid_abi=false;bool rejected=false;auto w=weights(16,16);
        try{(void)fpga::prepare_native_int8(bad,w.data(),16,16,requant(16));}catch(const std::exception&){rejected=true;}
        require(rejected,"Legacy ABI accepted");
        std::cout<<"PASS native INT8 driver: "<<cases<<" matrix/pair cases; tails, K17..2048, weight reuse, resident feedback, wrong ABI\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
