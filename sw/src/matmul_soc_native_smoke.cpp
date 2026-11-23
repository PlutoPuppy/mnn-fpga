#include "matmul_soc_native.hpp"
#include "matmul_soc_uio.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
std::vector<int8_t> pack(const std::vector<int8_t>& a,size_t m,size_t k) {
    std::vector<int8_t> p(m*k);
    for(size_t i=0;i<m;++i)for(size_t j=0;j<k;++j)p[(j/4*m+i)*4+j%4]=a[i*k+j];
    return p;
}
std::vector<int8_t> oracle(const std::vector<int8_t>& a,const std::vector<int8_t>& w,size_t m,size_t k,size_t n) {
    std::vector<int8_t> o(m*n);
    for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j) {
        int64_t acc=0;for(size_t t=0;t<k;++t)acc+=int(a[i*k+t])*int(w[j*k+t]);
        const int64_t scaled=acc*65536+(int(j%7)-3)*8388608;
        const auto rounded=scaled>=0?(scaled+8388608)/16777216:-((-scaled+8388608)/16777216);
        o[i*n+j]=int8_t(std::clamp(rounded,int64_t(-127),int64_t(127)));
    }
    return o;
}
fpga::MatmulSocInt8Requantization quant(size_t n) {
    fpga::MatmulSocInt8Requantization q;q.multiplier.assign(n,1.0/256);q.bias_q16.resize(n);
    for(size_t j=0;j<n;++j)q.bias_q16[j]=(int(j%7)-3)*32768;
    return q;
}
void compare(const std::vector<int8_t>& a,const std::vector<int8_t>& b) {
    for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]) {
        std::cerr<<"Mismatch at "<<i<<": FPGA="<<int(a[i])<<" oracle="<<int(b[i])<<'\n';
        throw std::runtime_error("Physical native matrix verification failed");
    }
}
}
int main() {
    try {
        fpga::MatmulSocUio io("/dev/mem",true,true);std::mt19937 rng(20261114);
        auto random=[&](size_t n,int range) {std::vector<int8_t> a(n);for(auto& x:a)x=int8_t(int(rng()%unsigned(range*2+1))-range);return a;};
        {
            const size_t m=1105,k=16,n=128;auto a=random(m*k,31),w=random(k*n,31),input=pack(a,m,k);
            auto prepared=fpga::prepare_native_int8(io,w.data(),k,n,quant(n));std::vector<int8_t> output(m*n);
            const auto stats=fpga::native_int8_linear(io,prepared,input.data(),output.data(),m);
            compare(output,pack(oracle(a,w,m,k,n),m,n));
            if(stats.submissions!=3)throw std::runtime_error("Small-K output capacity not respected");
            std::cout<<"PASS maximum 64KB output canvas with M1105 K16 N128, jobs="<<stats.submissions<<std::endl;
        }
        std::vector<size_t> reductions{32,64,128,160,512,640,1024};
        const bool extended=io.read(0x90)==fpga::native_int8_extended_abi;
        if(extended){reductions.push_back(1280);reductions.push_back(2048);}
        for(size_t k:reductions) {
            const size_t m=37,n=160;auto a=random(m*k,63),w=random(k*n,63),input=pack(a,m,k);std::vector<int8_t> output(m*n);
            const auto prepared=fpga::prepare_native_int8(io,w.data(),k,n,quant(n));
            const auto start=std::chrono::steady_clock::now();const auto stats=fpga::native_int8_linear(io,prepared,input.data(),output.data(),m);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            compare(output,pack(oracle(a,w,m,k,n),m,n));
            std::cout<<"PASS native "<<m<<'x'<<k<<" x "<<k<<'x'<<n<<" jobs="<<stats.submissions
                     <<" tile="<<prepared.tile_m<<'x'<<prepared.kp<<'x'<<prepared.tile_n
                     <<" B_capacity="<<prepared.operand_b_bytes<<" ms="<<ms<<std::endl;
        }
        for(size_t k:extended?std::vector<size_t>{128,160,320}:std::vector<size_t>{128,160}) {
            const size_t m=135,h=k*4,n=k;auto a=random(m*k,7),w=random(k*h,7),v=random(h*n,7),input=pack(a,m,k);
            auto first=fpga::prepare_native_int8(io,w.data(),k,h,quant(h));auto second=fpga::prepare_native_int8(io,v.data(),h,n,quant(n));
            std::array<int8_t,256> lut{};for(unsigned i=0;i<256;++i)lut[i]=int8_t(int(int8_t(i))/2);
            std::vector<int8_t> output(m*n);auto stats=fpga::native_int8_pair(io,first,second,lut,input.data(),output.data(),m);
            auto hidden=oracle(a,w,m,k,h);for(auto& x:hidden)x=lut[uint8_t(x)];
            compare(output,pack(oracle(hidden,v,m,h,n),m,n));
            std::cout<<"PASS resident MLP "<<m<<'x'<<k<<" -> "<<h<<" -> "<<n<<" jobs="<<stats.submissions
                     <<" local_bytes="<<stats.local_bytes<<" avoided="<<stats.hidden_bytes_avoided<<std::endl;
        }
        std::cout<<"PASS physical native INT32 matrix and FPGA-resident MLP checks\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
