#pragma once
#include <array>
#include <chrono>
#include <cstdint>

namespace fpga {
constexpr unsigned N = 16, Elements = N * N, Span = 4096;
constexpr unsigned JobCycles = 384; // Four 8x8 quadrants, each reducing over K=16.
namespace reg {
constexpr unsigned Id=0x00, Version=0x04, Control=0x08, Status=0x0c;
constexpr unsigned Cycles=0x10, Scratch=0x14, Format=0x18;
constexpr unsigned A=0x400, B=0x800, C=0xc00;
constexpr uint32_t Magic=0x53413136, Abi=0x00010000, FormatValue=0x20101008;
constexpr uint32_t Start=1, Ack=2, Abort=4, Busy=1, Done=2, Error=4;
}
using Input = std::array<int8_t, Elements>;
using Output = std::array<int32_t, Elements>;
// Offsets are BYTES. Implementations must provide ordered device accesses.
class RegisterIo {
public:
    virtual ~RegisterIo() = default;
    virtual uint32_t read(unsigned offset) = 0;
    virtual void write(unsigned offset, uint32_t value) = 0;
};
void probe(RegisterIo& io);
uint32_t gemm(RegisterIo& io, const Input& a, const Input& b, Output& c,
              std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));
Output reference(const Input& a, const Input& b);
// Exact subset for the first MNN demo: finite, integral INT8-valued floats.
// Returns false without submitting anything for unsupported values.
bool exact_float_tile(RegisterIo& io, const float* a, const float* b, float* c);
}
