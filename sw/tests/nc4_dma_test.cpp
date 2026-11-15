#include "nc4_dma.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        size_t cases=0;
        for(size_t m:{size_t(1),size_t(5),size_t(37),size_t(49),size_t(196),size_t(784)})
        for(size_t k:{size_t(4),size_t(20),size_t(128),size_t(160),size_t(512),size_t(640)})
        for(size_t first:{size_t(0),m/2}) {
            const size_t count=std::min(size_t(37),m-first),mp=(count+15)/16*16,kp=(k+15)/16*16;
            std::vector<int8_t> input(m*k),dense(mp*kp,99),oracle(mp*kp,0);
            for(size_t i=0;i<m;++i)for(size_t j=0;j<k;++j)input[(j/4*m+i)*4+j%4]=int8_t((i*31+j*13)%255-127);
            for(size_t i=0;i<count;++i)for(size_t j=0;j<k;++j)oracle[i*kp+j]=input[(j/4*m+first+i)*4+j%4];
            fpga::nc4_dma::pack(dense.data(),input.data(),m,k,first,count,mp,kp);
            if(dense!=oracle)throw std::runtime_error("NC4 pack mismatch");
            std::vector<int8_t> output(m*(k+8),99),expected=output;
            fpga::nc4_dma::unpack(output.data(),dense.data(),m,first,4,count,k,kp);
            for(size_t i=0;i<count;++i)for(size_t j=0;j<k;++j)expected[((4+j)/4*m+first+i)*4+j%4]=oracle[i*kp+j];
            if(output!=expected)throw std::runtime_error("NC4 unpack mismatch");
            ++cases;
        }
        std::cout<<"PASS NC4 DMA layout: "<<cases<<" cases, nonzero row/channel offsets, padding, all row tails; ISA="
#if defined(__arm__) && defined(__ARM_NEON)
                 <<"AArch32 NEON\n";
#else
                 <<"scalar\n";
#endif
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
