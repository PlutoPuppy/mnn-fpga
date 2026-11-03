#pragma once
#include "matmul_soc.hpp"
#include <string>
namespace fpga {
class MatmulSocUio final : public MatmulSocIo {
public:
    // devmem=true is an explicit bring-up fallback for a board without UIO;
    // maps only the upstream fixed 0xc0000000 aperture and requires root.
    explicit MatmulSocUio(const std::string& path, bool devmem=false);
    ~MatmulSocUio() override;
    MatmulSocUio(const MatmulSocUio&)=delete;
    MatmulSocUio& operator=(const MatmulSocUio&)=delete;
    uint32_t read(unsigned offset) override;
    void write(unsigned offset,uint32_t value) override;
    void write_beat(unsigned offset,const Beat& data) override;
    Beat read_beat(unsigned offset) override;
private:
    int fd_=-1;
    volatile uint8_t* base_=nullptr;
};
}
