#pragma once
#include "matmul_soc.hpp"
#include <memory>
#include <string>
namespace fpga {
class MatmulSocUio final : public MatmulSocIo {
public:
    // devmem=true is an explicit bring-up fallback for a board without UIO;
    // maps the upstream fixed 0xc0000000 aperture and requires root.
    // dma=true additionally maps LW DMA CSRs and the board-specific excluded
    // DDR window at 0x30000000, after checking /proc/iomem for overlaps.
    explicit MatmulSocUio(const std::string& path, bool devmem=false, bool dma=false);
    ~MatmulSocUio() override;
    MatmulSocUio(const MatmulSocUio&)=delete;
    MatmulSocUio& operator=(const MatmulSocUio&)=delete;
    uint32_t read(unsigned offset) override;
    void write(unsigned offset,uint32_t value) override;
    void write_beat(unsigned offset,const Beat& data) override;
    Beat read_beat(unsigned offset) override;
    void write_buffer(unsigned offset,const int8_t* data,size_t bytes) override;
    void read_buffer(unsigned offset,int8_t* data,size_t bytes) override;
    CachedBuffer cache_buffer(const int8_t* data,size_t bytes) override;
    void write_cached(unsigned offset,const CachedBuffer& data) override;
    void write_nc4(unsigned offset,const int8_t* data,size_t rows,size_t channels,
        size_t first,size_t count,size_t padded_rows,size_t padded_channels) override;
    void read_nc4(unsigned offset,int8_t* data,size_t rows,size_t channels,
        size_t first,size_t column,size_t count,size_t columns,size_t padded_rows,size_t padded_columns) override;
    void finish_output_reads() noexcept override;
    void feedback(unsigned source,unsigned destination,size_t bytes) override;
    void write_rows(unsigned offset,const int8_t* data,size_t rows,size_t width,size_t padded_rows) override;
    void read_rows(unsigned offset,int8_t* data,size_t rows,size_t width,size_t stride,size_t padded_rows) override;
    bool uses_dma() const { return dma_; }
private:
    struct OutputReader;
    void enable_dma();
    void collect_output_copy() noexcept;
    void dma_transfer(unsigned engine,uint32_t source,uint32_t destination,size_t bytes);
    int fd_=-1;
    volatile uint8_t* base_=nullptr;
    bool dma_=false;
    volatile uint32_t* dma_regs_=nullptr;
    volatile uint8_t* staging_=nullptr;
    size_t staging_span_=0x20000,cache_next_=0x20000;
    bool native_checked_=false;
    bool stream_stage_=false;
    bool batch_dma_poll_=false;
    bool chain_requested_=false, chain_enabled_=false;
    bool direct_a_=false, direct_c_=false;
    std::vector<int8_t> stage_cache_;
    std::unique_ptr<OutputReader> output_reader_;
    size_t native_b_capacity_=16384;
    size_t native_a_capacity_=16384,native_b_offset_=0x5000;
    void native_check();
};
}
