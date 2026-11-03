#include "fpga_gemm.hpp"
#include "matmul_soc.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
struct Soc : fpga::MatmulSocIo {
    std::map<unsigned,uint32_t> regs;
    std::array<int8_t,256> a{},b{},c{};
    size_t jobs=0,fail_at=0;
    bool busy=false;
    uint32_t read(unsigned off) override {return off==0x50 ? (busy ? 1u : 2u) : regs[off];}
    void write(unsigned off,uint32_t v) override {
        regs[off]=v;
        if(off) return;
        require(v==1 && regs[0x10]==16 && regs[0x20]==16 && regs[0x30]==16 &&
            regs[0x40]==16 && regs[0x70]==0 && regs[0x80]==16,"wrong register programming");
        ++jobs;
        if(fail_at==jobs) throw std::runtime_error("injected device failure");
        for(unsigned i=0;i<16;++i) for(unsigned j=0;j<16;++j) {
            int64_t sum=0;
            for(unsigned p=0;p<16;++p) sum+=int(a[i*16+p])*int(b[p*16+j]);
            // Explicit floor division, matching RTL arithmetic shift for negatives.
            const int64_t prod=sum*regs[0x60];
            const int scaled=int(prod>=0 ? prod/65536 : -((-prod+65535)/65536));
            require(scaled>=-128 && scaled<=127,"requantization saturated");
            c[i*16+j]=int8_t(scaled);
        }
    }
    void write_beat(unsigned off,const Beat& data) override {
        require(regs[0x20]==16 && regs[0x30]==16,"dimensions must precede buffers");
        require(off%16==0,"unaligned beat");
        auto& dst=off<0x5000 ? a:b;
        const unsigned idx=off-(off<0x5000 ? 0x1000:0x5000);
        require(idx<256,"buffer overrun");std::copy(data.begin(),data.end(),dst.begin()+idx);
    }
    Beat read_beat(unsigned off) override {
        require(off>=0x9000 && off<0x9100 && off%16==0,"bad output access");
        Beat out{};std::copy_n(c.begin()+off-0x9000,16,out.begin());return out;
    }
};
struct Legacy : fpga::RegisterIo {
    fpga::Input a{},b{};fpga::Output c{};
    size_t jobs=0;
    uint32_t read(unsigned off) override {
        using namespace fpga;
        if(off==reg::Id) return reg::Magic;
        if(off==reg::Version) return reg::Abi;
        if(off==reg::Format) return reg::FormatValue;
        if(off==reg::Status) return reg::Done;
        if(off==reg::Cycles) return JobCycles;
        require(off>=reg::C && off<Span,"bad legacy read");return uint32_t(c[(off-reg::C)/4]);
    }
    void write(unsigned off,uint32_t v) override {
        using namespace fpga;
        if(off==reg::Control) {if(v==reg::Start) {c=reference(a,b);++jobs;}return;}
        const unsigned idx=(off-(off<reg::B ? reg::A:reg::B))/4;
        (off<reg::B ? a:b)[idx]=static_cast<int8_t>(v<=127 ? int(v):int(v)-256);
    }
};
std::vector<float> stored(const std::vector<float>& x,size_t rows,size_t cols,bool transpose) {
    if(!transpose) return x;
    std::vector<float> out(x.size());
    for(size_t i=0;i<rows;++i) for(size_t j=0;j<cols;++j) out[j*rows+i]=x[i*cols+j];
    return out;
}
void problem(size_t m,size_t n,size_t k,bool ta,bool tb,bool tc) {
    std::vector<float> a(m*k),b(k*n),c(m*n,-999),exact(m*n),bias(n);
    for(size_t i=0;i<a.size();++i) a[i]=float(int(i*7%13)-6);
    for(size_t i=0;i<b.size();++i) b[i]=float(int(i*11%9)-4);
    for(size_t j=0;j<n;++j) bias[j]=float(j%3)*0.25f;
    for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j)
        for(size_t p=0;p<k;++p) exact[i*n+j]+=a[i*k+p]*b[p*n+j];
    const auto aa=stored(a,m,k,ta),bb=stored(b,k,n,tb);
    const size_t jobs=((m+15)/16)*((n+15)/16)*((k+15)/16);
    Legacy legacy;size_t submitted=0;
    require(fpga::exact_float_gemm(legacy,aa.data(),bb.data(),c.data(),m,n,k,ta,tb,tc,&submitted),"exact GEMM rejected");
    require(submitted==jobs && legacy.jobs==jobs,"wrong legacy submission count");
    for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j)
        require(c[tc ? j*m+i:i*n+j]==exact[i*n+j],"legacy result mismatch");
    Soc soc;
    const auto stats=fpga::matmul_soc_gemm(soc,aa.data(),bb.data(),c.data(),m,n,k,ta,tb,tc,bias.data());
    require(stats.submissions==jobs && soc.jobs==jobs,"wrong soc submission count");
    double max_error=0;
    for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j)
        max_error=std::max(max_error,std::abs(double(c[tc ? j*m+i:i*n+j])-exact[i*n+j]-bias[j]));
    // Conservative absolute bound for these bounded test operands, including
    // per-block floor error and input quantization. This is not a model accuracy threshold.
    require(max_error<2.5*((k+15)/16),"soc approximation outside error bound");
}
int main() {
    try {
        for(bool ta:{false,true}) for(bool tb:{false,true}) for(bool tc:{false,true}) {
            problem(16,16,16,ta,tb,tc);problem(32,48,64,ta,tb,tc);problem(17,31,33,ta,tb,tc);
        }
        problem(272,16,16,false,false,false); // exceeds the physical canvas; host tiles it
        Soc soc;std::vector<float> a(256,0),b(256,0),c(256,-99);
        fpga::matmul_soc_gemm(soc,a.data(),b.data(),c.data(),16,16,16);
        for(float v:c) require(v==0,"zero GEMM mismatch");
        a.back()=std::numeric_limits<float>::quiet_NaN();
        const auto before=soc.jobs;
        try {fpga::matmul_soc_gemm(soc,a.data(),b.data(),c.data(),16,16,16);throw 1;}
        catch(const std::invalid_argument&) {}
        require(soc.jobs==before,"nonfinite input touched hardware");
        Legacy legacy;
        require(!fpga::exact_float_gemm(legacy,a.data(),b.data(),c.data(),16,16,16) && !legacy.jobs,"legacy nonfinite submitted");
        a.assign(512,1);b.assign(512,1);c.assign(256,-99);
        Soc failing;failing.fail_at=2;
        try {fpga::matmul_soc_gemm(failing,a.data(),b.data(),c.data(),16,16,32);throw 1;}
        catch(const std::runtime_error&) {}
        for(float v:c) require(v==-99,"failure partially published output");
        Soc busy;busy.busy=true;
        try {fpga::matmul_soc_gemm(busy,a.data(),b.data(),c.data(),16,16,32);throw 1;}
        catch(const std::runtime_error&) {}
        require(!busy.jobs,"busy device submitted");
        std::cout<<"PASS: tiling, transpose, tails, >256 rows, quantized hardware model, zero/nonfinite, failure atomicity, busy\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
