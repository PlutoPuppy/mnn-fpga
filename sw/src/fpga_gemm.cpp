#include "fpga_gemm.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>

namespace fpga {
void probe(RegisterIo& io) {
    if (io.read(reg::Id) != reg::Magic || io.read(reg::Version) != reg::Abi ||
        io.read(reg::Format) != reg::FormatValue)
        throw std::runtime_error("Accelerator ID, ABI, or numeric format mismatch");
}
uint32_t gemm(RegisterIo& io, const Input& a, const Input& b, Output& c,
              std::chrono::milliseconds timeout) {
    if (timeout.count() <= 0) throw std::invalid_argument("Timeout must be positive");
    probe(io);
    if (io.read(reg::Status) & reg::Busy) throw std::runtime_error("Accelerator already busy");
    io.write(reg::Control, reg::Ack);
    for (unsigned i=0; i<Elements; ++i) {
        io.write(reg::A+4*i, static_cast<uint8_t>(a[i]));
        io.write(reg::B+4*i, static_cast<uint8_t>(b[i]));
    }
    // RegisterIo orders operand writes before the START doorbell.
    io.write(reg::Control, reg::Start);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        const auto status = io.read(reg::Status);
        if (status & reg::Error) {
            io.write(reg::Control, reg::Abort);
            throw std::runtime_error("Accelerator reported a protocol error");
        }
        if ((status & (reg::Busy | reg::Done)) == reg::Done) break;
        if (std::chrono::steady_clock::now() >= deadline) {
            io.write(reg::Control, reg::Abort);
            throw std::runtime_error("Accelerator timed out; abort requested");
        }
    }
    for (unsigned i=0; i<Elements; ++i) {
        uint32_t raw = io.read(reg::C+4*i);
        // Defined signed decoding, including negative results.
        c[i] = raw <= INT32_MAX ? static_cast<int32_t>(raw)
                               : static_cast<int32_t>(static_cast<int64_t>(raw)-(INT64_C(1)<<32));
    }
    return io.read(reg::Cycles);
}
Output reference(const Input& a, const Input& b) {
    Output c{};
    for (unsigned i=0; i<N; ++i)
        for (unsigned j=0; j<N; ++j)
            for (unsigned k=0; k<N; ++k) c[i*N+j] += int32_t(a[i*N+k])*int32_t(b[k*N+j]);
    return c;
}
bool exact_float_tile(RegisterIo& io, const float* a, const float* b, float* c) {
    Input qa{}, qb{};
    for (unsigned i=0; i<Elements; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
            a[i] < -128 || a[i] > 127 || b[i] < -128 || b[i] > 127 ||
            std::trunc(a[i]) != a[i] || std::trunc(b[i]) != b[i]) return false;
        qa[i] = static_cast<int8_t>(a[i]); qb[i] = static_cast<int8_t>(b[i]);
    }
    Output out{};
    gemm(io, qa, qb, out);
    for (unsigned i=0; i<Elements; ++i) c[i] = static_cast<float>(out[i]);
    return true;
}
bool exact_float_gemm(RegisterIo& io,const float* a,const float* b,float* c,
    size_t m,size_t n,size_t k,bool ta,bool tb,bool tc,size_t* submissions) {
    if (submissions) *submissions=0;
    const auto limit=std::numeric_limits<size_t>::max();
    if (!a || !b || !c || !m || !n || !k || k>1024 || m>limit/k || k>limit/n || m>limit/n)
        return false;
    auto eligible=[](const float* x,size_t count) {
        for(size_t i=0;i<count;++i)
            if(!std::isfinite(x[i]) || x[i]<-128 || x[i]>127 || std::trunc(x[i])!=x[i]) return false;
        return true;
    };
    if(!eligible(a,m*k) || !eligible(b,k*n)) return false;
    std::vector<int32_t> result(m*n,0);
    for(size_t r=0;r<m;r+=N) for(size_t col=0;col<n;col+=N) for(size_t p=0;p<k;p+=N) {
        Input qa{},qb{};Output out{};
        for(size_t i=0;i<N && r+i<m;++i) for(size_t t=0;t<N && p+t<k;++t)
            qa[i*N+t]=static_cast<int8_t>(a[ta ? (p+t)*m+r+i : (r+i)*k+p+t]);
        for(size_t t=0;t<N && p+t<k;++t) for(size_t j=0;j<N && col+j<n;++j)
            qb[t*N+j]=static_cast<int8_t>(b[tb ? (col+j)*k+p+t : (p+t)*n+col+j]);
        gemm(io,qa,qb,out);
        if(submissions) ++*submissions;
        for(size_t i=0;i<N && r+i<m;++i) for(size_t j=0;j<N && col+j<n;++j)
            result[(r+i)*n+col+j]+=out[i*N+j];
    }
    for(size_t i=0;i<m;++i) for(size_t j=0;j<n;++j)
        c[tc ? j*m+i : i*n+j]=static_cast<float>(result[i*n+j]);
    return true;
}
}
