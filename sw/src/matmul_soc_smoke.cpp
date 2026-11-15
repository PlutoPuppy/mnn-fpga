#include "matmul_soc_uio.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>

int main(int argc,char** argv) {
    try {
        if(argc<2 || argc>3) throw std::invalid_argument("Usage: matmul_soc_smoke /dev/uioN | --devmem [--dma]");
        const bool physical=std::string(argv[1])=="--devmem";
        const bool dma=argc==3 && std::string(argv[2])=="--dma";
        if(argc==3 && !dma) throw std::invalid_argument("Unknown transport option");
        fpga::MatmulSocUio io(physical ? "/dev/mem":argv[1],physical,dma);
        size_t jobs=0;
        const std::vector<std::tuple<size_t,size_t,size_t>> cases={
            {17,31,16},{17,31,32},{17,31,33},{17,31,65},
            {49,49,32},{256,256,64},{272,17,16}};
        for(const auto& shape:cases) for(bool signed_pattern:{false,true}) {
            size_t m,n,k;std::tie(m,n,k)=shape;
            std::vector<float> a(m*k,1),b(k*n,1),c(m*n,-1);
            if(signed_pattern) {
                for(size_t i=0;i<m;++i) for(size_t p=0;p<k;++p) a[i*k+p]=float(int((i*3+p*7)%5)-2);
                for(size_t p=0;p<k;++p) for(size_t j=0;j<n;++j) b[p*n+j]=float(int((p*11+j*3)%5)-2);
            }
            const auto qa=fpga::prepare_matmul_soc_operand(a.data(),m,k);
            const auto qb=fpga::prepare_matmul_soc_operand(b.data(),k,n);
            const auto stats=fpga::matmul_soc_gemm_prepared(io,qa,qb,c.data());
            if(stats.quantized_values || stats.submissions!=((m+255)/256)*((n+255)/256)*((k+63)/64))
                throw std::runtime_error("Prepared INT8 batching count mismatch");
            double worst=0;
            for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j) {
                double expected=0;
                for(size_t p=0;p<k;++p) expected+=double(a[i*k+p])*b[p*n+j];
                worst=std::max(worst,std::abs(double(c[i*n+j])-expected));
            }
            if(worst>2.5*((k+15)/16)) throw std::runtime_error("FPGA result outside smoke-test error bound");
            jobs+=stats.submissions;
            std::cout<<"PASS "<<m<<'x'<<k<<" * "<<k<<'x'<<n<<(signed_pattern ? " signed pattern": " ones")<<": "<<stats.submissions<<" submissions, max error "<<worst<<", in="<<stats.operand_bytes<<" B, out="<<stats.result_bytes<<" B\n";
        }
        std::cout<<"Completed FPGA submissions: "<<jobs<<'\n';
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
