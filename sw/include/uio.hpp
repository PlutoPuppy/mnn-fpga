#pragma once
#include "fpga_gemm.hpp"
#include <string>
namespace fpga {
// Dedicated, page-aligned 4 KiB UIO map0; exclusive cooperative flock.
class Uio final : public RegisterIo {
public:
    explicit Uio(const std::string& path);
    ~Uio() override;
    Uio(const Uio&) = delete;
    Uio& operator=(const Uio&) = delete;
    uint32_t read(unsigned offset) override;
    void write(unsigned offset, uint32_t value) override;
private:
    int fd_ = -1;
    volatile uint32_t* base_ = nullptr;
};
}
