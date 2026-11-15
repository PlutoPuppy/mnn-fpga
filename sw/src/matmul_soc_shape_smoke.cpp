#include "matmul_soc_uio.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

int main() {
    try {
        fpga::MatmulSocUio io("/dev/mem",true,true);
        for(const auto& shape:std::vector<std::tuple<size_t,size_t,size_t>>{{128,128,128},{64,64,256},{784,512,128}}) {
            size_t m,n,k;std::tie(m,n,k)=shape;
            fpga::MatmulSocOperand a,b;a.rows=m;a.cols=k;b.rows=k;b.cols=n;
            a.values.resize(m*k);b.values.resize(k*n);
            uint32_t state=1234567;
            for(auto* values:{&a.values,&b.values}) for(auto& v:*values) {
                state=state*1664525u+1013904223u;v=(state>>31)?1:-1;
            }
            std::vector<float> result(m*n),bias(n);
            for(size_t j=0;j<n;++j) bias[j]=float(j%5)*0.125f;
            const auto plan=fpga::plan_matmul_soc(m,n,k,fpga::MatmulSocTilePolicy::Capacity);
            const auto stats=fpga::matmul_soc_gemm_prepared(io,a,b,result.data(),false,bias.data(),std::chrono::milliseconds(1000),&plan);
            double worst=0;
            for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j) {
                double expected=bias[j];
                for(size_t p=0;p<k;p+=plan.tile_k) {
                    const auto kr=std::min(plan.tile_k,k-p),bands=(kr+15)/16;
                    // All operands have |q|=1, so the exact programmed bound
                    // and multiplier have this closed form for every tile.
                    const auto scale=std::min(size_t(65535),(127-bands)*65536/kr);
                    int quantized=0;
                    for(size_t band=0;band<kr;band+=16) {
                        int64_t dot=0;
                        for(size_t t=band;t<std::min(kr,band+16);++t) dot+=int(a.values[i*k+p+t])*int(b.values[(p+t)*n+j]);
                        const int64_t product=dot*int64_t(scale);
                        const int value=int(product>=0 ? product/65536:-((-product+65535)/65536));
                        quantized=std::max(-128,std::min(127,quantized+std::max(-128,std::min(127,value))));
                    }
                    expected+=double(quantized)*65536/scale;
                }
                worst=std::max(worst,std::abs(result[i*n+j]-expected));
            }
            if(worst>1e-3 || stats.submissions!=plan.jobs || stats.operand_bytes!=plan.operand_bytes || stats.result_bytes!=plan.result_bytes)
                throw std::runtime_error("Capacity DMA result/traffic differs from independent band reference");
            std::cout<<"PASS "<<m<<'x'<<k<<" times "<<k<<'x'<<n<<" tile="<<plan.tile_m<<'x'<<plan.tile_k<<'x'<<plan.tile_n
                     <<" jobs="<<stats.submissions<<" input="<<stats.operand_bytes<<" output="<<stats.result_bytes<<" max_error="<<worst<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
