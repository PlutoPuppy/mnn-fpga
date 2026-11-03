#include "matmul_soc.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fpga {
namespace {
size_t area(size_t x, size_t y) {
    if (!x || !y || x > std::numeric_limits<size_t>::max()/y)
        throw std::invalid_argument("Invalid matrix dimensions");
    return x*y;
}
double quantize(const std::array<float,256>& x, std::array<int8_t,256>& q) {
    double peak=0;
    for (float v:x) peak=std::max(peak,std::abs(double(v)));
    const double scale=peak ? peak/127.0 : 1.0;
    for (size_t i=0;i<x.size();++i) q[i]=static_cast<int8_t>(std::round(x[i]/scale));
    return scale;
}
}
MatmulSocStats matmul_soc_gemm(MatmulSocIo& io, const float* a, const float* b,
    float* c, size_t m, size_t n, size_t k, bool ta, bool tb, bool tc,
    const float* bias, std::chrono::milliseconds timeout) {
    const auto ac=area(m,k),bc=area(k,n),cc=area(m,n);
    if (!a || !b || !c || timeout.count()<=0) throw std::invalid_argument("Invalid GEMM arguments");
    // Validate the entire operation before touching hardware (including tails).
    for (size_t i=0;i<ac;++i) if (!std::isfinite(a[i])) throw std::invalid_argument("Nonfinite A");
    for (size_t i=0;i<bc;++i) if (!std::isfinite(b[i])) throw std::invalid_argument("Nonfinite B");
    if (bias) for (size_t j=0;j<n;++j) if (!std::isfinite(bias[j])) throw std::invalid_argument("Nonfinite bias");
    std::vector<double> result(cc,0.0);
    MatmulSocStats stats;
    for (size_t r=0;r<m;r+=16) for (size_t col=0;col<n;col+=16)
        for (size_t p=0;p<k;p+=16) {
            std::array<float,256> at{},bt{};
            std::array<int8_t,256> qa{},qb{};
            for (size_t i=0;i<16 && r+i<m;++i) for (size_t t=0;t<16 && p+t<k;++t)
                at[i*16+t]=a[ta ? (p+t)*m+r+i : (r+i)*k+p+t];
            for (size_t t=0;t<16 && p+t<k;++t) for (size_t j=0;j<16 && col+j<n;++j)
                bt[t*16+j]=b[tb ? (col+j)*k+p+t : (p+t)*n+col+j];
            const double sa=quantize(at,qa),sb=quantize(bt,qb);
            // Bound every dot product to avoid hardware's INT8 saturation.
            unsigned row_bound=0,bpeak=0;
            for (size_t i=0;i<16;++i) {
                unsigned sum=0;
                for (size_t t=0;t<16;++t) sum+=unsigned(std::abs(int(qa[i*16+t])));
                row_bound=std::max(row_bound,sum);
            }
            for (int v:qb) bpeak=std::max(bpeak,unsigned(std::abs(v)));
            const unsigned bound=row_bound*bpeak;
            const uint32_t scale=bound ? std::min(65535u,(127u*65536u)/bound) : 65535u;
            if (io.read(0x50)&1) throw std::runtime_error("matmul-soc busy; reset hardware before retry");
            // Dimensions MUST precede buffer writes: they control address remapping.
            io.write(0x10,16); io.write(0x20,16); io.write(0x30,16); io.write(0x40,16);
            io.write(0x60,scale); io.write(0x70,0); io.write(0x80,16);
            for (unsigned i=0;i<16;++i) {
                MatmulSocIo::Beat wa{},xb{};
                std::copy_n(qa.begin()+16*i,16,wa.begin());
                std::copy_n(qb.begin()+16*i,16,xb.begin());
                io.write_beat(0x1000+16*i,wa); io.write_beat(0x5000+16*i,xb);
            }
            (void)io.read(0x50); // done is read-to-clear
            io.write(0,1);
            const auto deadline=std::chrono::steady_clock::now()+timeout;
            for (;;) {
                const auto status=io.read(0x50);
                if ((status&3)==2) break;
                if (std::chrono::steady_clock::now()>=deadline)
                    throw std::runtime_error("matmul-soc timed out; hardware reset required (no abort register)");
            }
            ++stats.submissions;
            const double dequant=sa*sb*65536.0/scale;
            for (unsigned i=0;i<16;++i) {
                const auto out=io.read_beat(0x9000+16*i);
                for (size_t j=0;j<16 && col+j<n && r+i<m;++j)
                    result[(r+i)*n+col+j]+=double(out[j])*dequant;
            }
        }
    std::vector<float> staged(cc);
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) {
        const double v=result[i*n+j]+(bias ? bias[j] : 0.0);
        if (!std::isfinite(v) || std::abs(v)>std::numeric_limits<float>::max())
            throw std::overflow_error("FP32 output overflow");
        staged[tc ? j*m+i : i*n+j]=static_cast<float>(v);
    }
    std::copy(staged.begin(),staged.end(),c);
    return stats;
}
}
