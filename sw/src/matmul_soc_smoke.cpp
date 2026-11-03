#include "matmul_soc_uio.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::invalid_argument("Usage: matmul_soc_smoke /dev/uioN | --devmem");
        const bool physical=std::string(argv[1])=="--devmem";
        fpga::MatmulSocUio io(physical ? "/dev/mem":argv[1],physical);
        size_t jobs=0;
        for(const auto k:{16u,32u,33u}) for(bool signed_pattern:{false,true}) {
            const size_t m=17,n=31;
            std::vector<float> a(m*k,1),b(k*n,1),c(m*n,-1);
            if(signed_pattern) {
                for(size_t i=0;i<m;++i) for(size_t p=0;p<k;++p) a[i*k+p]=float(int((i*3+p*7)%5)-2);
                for(size_t p=0;p<k;++p) for(size_t j=0;j<n;++j) b[p*n+j]=float(int((p*11+j*3)%5)-2);
            }
            const auto stats=fpga::matmul_soc_gemm(io,a.data(),b.data(),c.data(),m,n,k);
            double worst=0;
            for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j) {
                double expected=0;
                for(size_t p=0;p<k;++p) expected+=double(a[i*k+p])*b[p*n+j];
                worst=std::max(worst,std::abs(double(c[i*n+j])-expected));
            }
            if(worst>2.5*((k+15)/16)) throw std::runtime_error("FPGA result outside smoke-test error bound");
            jobs+=stats.submissions;
            std::cout<<"PASS "<<m<<'x'<<k<<" * "<<k<<'x'<<n<<(signed_pattern ? " signed pattern": " ones")<<": "<<stats.submissions<<" submissions, max error "<<worst<<'\n';
        }
        std::cout<<"Completed FPGA submissions: "<<jobs<<'\n';
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
