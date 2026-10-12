#pragma once
#include "fpga_gemm.hpp"
#include <iostream>
#include <stdexcept>
namespace fpga {
inline void run_vectors(RegisterIo& io) {
    probe(io);
    io.write(reg::Scratch, 0xa55a1234);
    if (io.read(reg::Scratch) != 0xa55a1234) throw std::runtime_error("Scratch mismatch");
    uint32_t rng=0x496;
    for (unsigned test=0; test<12; ++test) {
        Input a{}, b{};
        for (unsigned i=0; i<Elements; ++i) {
            rng = rng*1664525u+1013904223u;
            a[i]=static_cast<int8_t>(int(rng >> 24)-128);
            rng = rng*1664525u+1013904223u;
            b[i]=static_cast<int8_t>(int(rng >> 24)-128);
        }
        if (test==0) { // Identity, including signed values and k=0.
            b.fill(0); for (unsigned i=0; i<N; ++i) b[i*N+i]=1;
        }
        if (test==1) b.fill(0);
        if (test==2) { a.fill(-128); b.fill(-128); }
        if (test==3) { a.fill(-128); b.fill(127); }
        if (test==4) { a.fill(0); b.fill(0); a[15*N]=7; b[15]=-9; }
        // Also check operand readback before the first submission.
        if (test==0) for (unsigned i=0; i<Elements; ++i) {
            io.write(reg::A+4*i, static_cast<uint8_t>(a[i]));
            io.write(reg::B+4*i, static_cast<uint8_t>(b[i]));
            if (io.read(reg::A+4*i) != static_cast<uint32_t>(static_cast<int32_t>(a[i])) ||
                io.read(reg::B+4*i) != static_cast<uint32_t>(static_cast<int32_t>(b[i])))
                throw std::runtime_error("Operand readback mismatch");
        }
        Output actual{};
        auto cycles=gemm(io,a,b,actual);
        const auto expected=reference(a,b);
        for (unsigned i=0; i<Elements; ++i)
            if (actual[i] != expected[i]) throw std::runtime_error("Vector "+std::to_string(test)+
                " index "+std::to_string(i)+" expected "+std::to_string(expected[i])+
                " got "+std::to_string(actual[i]));
        if (cycles != JobCycles) throw std::runtime_error("Unexpected job cycle count");
        std::cout << "PASS vector " << test << ": 256 outputs, " << cycles << " FPGA cycles\n";
    }
}
}
