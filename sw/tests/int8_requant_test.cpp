#include "int8_requant.hpp"
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>
int main() {
    try {
        for(size_t count:{size_t(1),size_t(7),size_t(8),size_t(9),size_t(127),size_t(256),size_t(257),size_t(401408)}) {
            std::vector<int8_t> codes(count),scalar(count+8,99),simd(count+8,99);
            std::vector<int32_t> gain(count),bias(count);
            for(size_t i=0;i<count;++i) {codes[i]=int8_t(int(i%256)-128);
                gain[i]=i%3==0?32768:i%3==1?8000000:int32_t(i%17+1)*49152;
                bias[i]=int32_t(i%5)*32768-65536;}
            for(int minimum:{-128,-127,-20}) {
                fpga::requantize_int8(codes.data(),scalar.data(),gain.data(),bias.data(),count,minimum,127,false);
                fpga::requantize_int8(codes.data(),simd.data(),gain.data(),bias.data(),count,minimum,127,true);
                if(scalar!=simd) throw std::runtime_error("SIMD/scalar mismatch or tail overwrite");
                for(size_t i=0;i<count;++i) {
                    const int64_t v=int64_t(codes[i])*gain[i]+bias[i];
                    const int64_t rounded=v>=0?(v+32768)/65536:-((-v+32768)/65536);
                    const int expected=std::max(minimum,std::min(127,int(rounded)));
                    if(simd[i]!=expected) throw std::runtime_error("Q16 rounding/clamp differs from INT64 oracle");
                }
            }
        }
#if defined(__ARM_NEON)
        std::puts("PASS ARM NEON and scalar requantization: all INT8 codes, signed ties, clamps, tails, full fc1 length");
#else
        std::puts("PASS scalar requantization oracle; ARM NEON unavailable on this host");
#endif
    } catch(const std::exception& e) {std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
