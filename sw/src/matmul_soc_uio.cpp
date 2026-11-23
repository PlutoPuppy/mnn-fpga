#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#include "matmul_soc_uio.hpp"
#include "nc4_dma.hpp"
#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__arm__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
namespace fpga {
namespace {
constexpr size_t span=0x20000;
void barrier() {
#if defined(__arm__)
    asm volatile("dmb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}
void check(unsigned off,bool buffer) {
    if (buffer ? (off%16 || off<0x1000 || off>=0x19500)
               : (off%16 || off>0xb0)) throw std::out_of_range("Invalid matmul-soc offset");
}
}
struct MatmulSocUio::OutputReader {
    struct Tile {
        int8_t* target;
        const volatile int8_t* source;
        size_t rows, first, column, count, columns, pitch;
        bool measure;
    };

    OutputReader() : worker([this] { copy_loop(); }) {}

    ~OutputReader() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping=true;
        }
        scheduled.notify_one();
        worker.join();
    }

    void submit(const Tile& tile) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            work=tile;
            pending=true;
        }
        scheduled.notify_one();
    }

    uint64_t finish() {
        std::unique_lock<std::mutex> lock(mutex);
        completed.wait(lock,[this] { return !pending; });
        const uint64_t elapsed=copy_ns;
        copy_ns=0;
        return elapsed;
    }

private:
    void copy_loop() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            scheduled.wait(lock,[this] { return pending || stopping; });
            if (stopping && !pending) return;
            const Tile tile=work;
            lock.unlock();
            const auto start=tile.measure ? ProfileClock::now() : ProfileClock::time_point{};
            barrier();
            nc4_dma::unpack(tile.target,tile.source,tile.rows,tile.first,tile.column,
                            tile.count,tile.columns,tile.pitch);
            barrier();
            const uint64_t elapsed=tile.measure ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                ProfileClock::now()-start).count() : 0;
            lock.lock();
            copy_ns=elapsed;
            pending=false;
            completed.notify_one();
        }
    }

    std::mutex mutex;
    std::condition_variable scheduled, completed;
    Tile work{};
    uint64_t copy_ns=0;
    bool pending=false, stopping=false;
    std::thread worker;
};

MatmulSocUio::MatmulSocUio(const std::string& path,bool devmem,bool dma) {
    const auto* fast=std::getenv("MNN_MATMUL_SOC_FAST_POLL");
    fast_poll=fast && std::strcmp(fast,"1")==0;
    const auto* stream=std::getenv("MNN_MATMUL_SOC_STREAM_STAGE");
    stream_stage_=stream && std::strcmp(stream,"1")==0;
    const auto* reuse=std::getenv("MNN_MATMUL_SOC_REUSE_A");
    reuse_native_activations=reuse && std::strcmp(reuse,"1")==0;
    const auto* live=std::getenv("MNN_MATMUL_SOC_LIVE_ROWS");
    transfer_live_rows=live && std::strcmp(live,"1")==0;
    const auto* batch=std::getenv("MNN_MATMUL_SOC_DMA_POLL_BATCH");
    batch_dma_poll_=batch && std::strcmp(batch,"1")==0;
    const auto* chain=std::getenv("MNN_MATMUL_SOC_DMA_CHAIN");
    chain_requested_=chain && std::strcmp(chain,"1")==0;
    const auto* direct_a=std::getenv("MNN_MATMUL_SOC_DIRECT_A");
    const auto* direct_c=std::getenv("MNN_MATMUL_SOC_DIRECT_C");
    direct_a_=direct_a && std::strcmp(direct_a,"1")==0;
    direct_c_=direct_c && std::strcmp(direct_c,"1")==0;
    if(stream_stage_)stage_cache_.resize(65536+15);
#if !defined(__arm__) || !defined(__ARM_NEON)
    (void)path;(void)devmem;(void)dma;
    throw std::runtime_error("matmul-soc PIO requires ARMv7 NEON 128-bit device accesses");
#else
    if (devmem && path!="/dev/mem") throw std::invalid_argument("Physical transport requires /dev/mem");
    if (dma && !devmem) throw std::invalid_argument("DMA prototype requires explicit /dev/mem mode");
    const auto name=path.substr(path.find_last_of('/')+1);
    if (!devmem) {
        if (name.size()<=3 || name.substr(0,3)!="uio" || name.find_first_not_of("0123456789",3)!=std::string::npos)
            throw std::invalid_argument("Expected /dev/uioN");
        unsigned long size=0,offset=0;
        std::ifstream sz("/sys/class/uio/"+name+"/maps/map0/size");
        std::ifstream of("/sys/class/uio/"+name+"/maps/map0/offset");
        if (!(sz>>std::hex>>size) || !(of>>std::hex>>offset) || size<span || offset)
            throw std::runtime_error("matmul-soc requires page-aligned 128 KB UIO map0");
    }
    fd_=open(path.c_str(),O_RDWR|O_CLOEXEC|O_SYNC);
    if (fd_<0) throw std::runtime_error("Cannot open matmul-soc UIO");
    if (flock(fd_,LOCK_EX|LOCK_NB)<0) {close(fd_);fd_=-1;throw std::runtime_error("matmul-soc UIO locked");}
    void* p=mmap(nullptr,span,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,devmem ? 0xc0000000ULL : 0);
    if (p==MAP_FAILED) {close(fd_);fd_=-1;throw std::runtime_error("Cannot map matmul-soc UIO");}
    base_=static_cast<volatile uint8_t*>(p);
    try {
        if(dma) enable_dma();
#if defined(__arm__) && defined(__ARM_NEON)
        const auto* pipeline=std::getenv("MNN_MATMUL_SOC_PIPELINE_C");
        if(dma_ && !direct_c_ && !stream_stage_ && pipeline && std::strcmp(pipeline,"1")==0)
            output_reader_.reset(new OutputReader);
#endif
    }
    catch(...) {
        if(dma_regs_) munmap(const_cast<uint32_t*>(dma_regs_),4096);
        if(staging_) munmap(const_cast<uint8_t*>(staging_),staging_span_);
        munmap(const_cast<uint8_t*>(base_),span);base_=nullptr;
        close(fd_);fd_=-1;throw;
    }
#endif
}
MatmulSocUio::~MatmulSocUio() {
    output_reader_.reset();
    if(dma_) {
        dma_regs_[6]=0;dma_regs_[14]=0;if(native_checked_)dma_regs_[22]=0;barrier();
    }
    if(dma_regs_) munmap(const_cast<uint32_t*>(dma_regs_),4096);
    if(staging_) munmap(const_cast<uint8_t*>(staging_),staging_span_);
    if (base_) munmap(const_cast<uint8_t*>(base_),span);
    if (fd_>=0) close(fd_);
}

void MatmulSocUio::collect_output_copy() noexcept {
    const uint64_t ns=output_reader_->finish();
    if(profile)profile->ns[MatmulSocProfile::output_stage_copy]+=ns;
}

void MatmulSocUio::finish_output_reads() noexcept {
    if(!output_reader_)return;
    const auto start=profile_now(profile);
    collect_output_copy();
    profile_add(profile,MatmulSocProfile::download,start);
}
void MatmulSocUio::enable_dma() {
    // Board-specific opt-in: Linux must exclude the entire staging window.
    // Never point DMA at malloc memory or at a hard-coded address in System RAM.
    std::ifstream map("/proc/iomem");
    if(!map) throw std::runtime_error("Cannot validate DMA staging memory");
    std::string line;bool system_ram=false;
    while(std::getline(map,line)) {
        unsigned long long lo=0,hi=0;
        if(std::sscanf(line.c_str()," %llx-%llx",&lo,&hi)!=2) continue;
        if(line.find("System RAM")!=std::string::npos && hi>lo) system_ram=true;
        if(lo<=0x30ffffffULL && hi>=0x30000000ULL)
            throw std::runtime_error("DMA staging window overlaps a Linux memory resource");
    }
    if(!system_ram) throw std::runtime_error("Cannot establish Linux RAM map for DMA");
    void* regs=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0xff200000ULL);
    if(regs==MAP_FAILED) throw std::runtime_error("Cannot map DMA CSRs");
    dma_regs_=static_cast<volatile uint32_t*>(regs);
    barrier();
    if((dma_regs_[0]|dma_regs_[8])&2u) throw std::runtime_error("DMA engine busy; reset before retry");
    staging_span_=0x1000000;
    void* stage=mmap(nullptr,staging_span_,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0x30000000ULL);
    if(stage==MAP_FAILED) throw std::runtime_error("Cannot map excluded DDR staging window");
    staging_=static_cast<volatile uint8_t*>(stage);
    dma_=true;
}
void MatmulSocUio::dma_transfer(unsigned engine,uint32_t source,uint32_t destination,size_t bytes) {
    auto* regs=dma_regs_+8*engine;
    // This DMA emits ONE burst per descriptor. Its burstcount register is
    // generated from maxBurstSize, and length/16 is assigned without a clamp.
    // Exceeding that limit can wrap the counter to zero and wedge a transfer.
    const size_t descriptor_bytes=chain_enabled_ ? 65536 : native_checked_ ? (engine==2?256:2048):8192;
    const uint32_t control=(1u<<3)|(1u<<7)|(1u<<11)|(chain_enabled_ ? (1u<<13) : 0);
    const bool fast=fast_poll && native_checked_;
    const bool batch_poll=fast && batch_dma_poll_;
    const auto transfer_deadline=batch_poll ? std::chrono::steady_clock::now()+std::chrono::seconds(1)
                                            : std::chrono::steady_clock::time_point{};
    if(fast){regs[6]=0;barrier();}
    for(size_t off=0;off<bytes;off+=descriptor_bytes) {
        ProfileSteps timer(profile);
        const auto count=uint32_t(std::min(descriptor_bytes,bytes-off));
        if(!fast){regs[6]=0;regs[3]=0;}
        regs[0]=0;barrier();
        regs[1]=source+uint32_t(off);regs[2]=destination+uint32_t(off);regs[3]=count;
        barrier();regs[6]=control;barrier();
        timer.end(engine==2 ? MatmulSocProfile::local_dma_setup:
                  engine ? MatmulSocProfile::output_dma_setup:MatmulSocProfile::input_dma_setup);
        const auto deadline=batch_poll ? transfer_deadline : std::chrono::steady_clock::now()+std::chrono::seconds(1);
        unsigned polls=0;
        for(;;) {
            // GO is ordered before polling, and stop/barrier orders the
            // completed transfer before CPU or accelerator access to data.
            if(!batch_poll) barrier();
            const auto status=regs[0];
            if(!batch_poll) barrier();
            if(profile) {if(engine==2)++profile->local_polls;else if(engine) ++profile->output_polls;else ++profile->input_polls;}
            if(chain_enabled_ && (status&4u)) {
                regs[6]=0;barrier();
                throw std::runtime_error("DMA chain rejected transfer address or length");
            }
            if(status&1u) break;
            if((!fast || (++polls&63u)==0) && std::chrono::steady_clock::now()>=deadline) {
                std::ostringstream message;
                message<<"DMA"<<engine<<" timeout: source=0x"<<std::hex<<(source+off)
                       <<" destination=0x"<<(destination+off)<<std::dec<<" bytes="<<count
                       <<" status="<<status<<" remaining="<<regs[3]<<"; reset hardware before retry";
                regs[6]=0;barrier();
                throw std::runtime_error(message.str());
            }
        }
        regs[6]=0;barrier();
        timer.end(engine==2 ? MatmulSocProfile::local_dma_wait:
                  engine ? MatmulSocProfile::output_dma_wait:MatmulSocProfile::input_dma_wait);
        if(profile) {if(engine==2){++profile->local_descriptors;profile->local_bytes+=count;}
            else if(engine) ++profile->output_descriptors;else ++profile->input_descriptors;}
    }
}
uint32_t MatmulSocUio::read(unsigned off) {
    check(off,false);barrier();const auto v=*reinterpret_cast<volatile uint32_t*>(base_+off);barrier();return v;
}
void MatmulSocUio::write(unsigned off,uint32_t v) {
    check(off,false);barrier();*reinterpret_cast<volatile uint32_t*>(base_+off)=v;barrier();
}
void MatmulSocUio::write_beat(unsigned off,const Beat& data) {
    check(off,true);
    if (off>=0x9000 && off<0x19000) throw std::out_of_range("Output buffer is read-only");
    if (off>=0x19000) native_check();
#if defined(__arm__) && defined(__ARM_NEON)
    const auto v=vld1q_s8(data.data());barrier();
    asm volatile("vst1.8 {%e0-%f0}, [%1 :128]" :: "w"(v),"r"(base_+off):"memory");
    barrier();
#else
    (void)data;throw std::runtime_error("128-bit ARM NEON access required");
#endif
}
MatmulSocIo::Beat MatmulSocUio::read_beat(unsigned off) {
    check(off,true);
    if (off<0x9000) throw std::out_of_range("Input buffers are write-only");
#if defined(__arm__) && defined(__ARM_NEON)
    int8x16_t v;barrier();
    asm volatile("vld1.8 {%e0-%f0}, [%1 :128]" : "=w"(v):"r"(base_+off):"memory");
    barrier();Beat out{};vst1q_s8(out.data(),v);return out;
#else
    throw std::runtime_error("128-bit ARM NEON access required");
#endif
}
void MatmulSocUio::write_buffer(unsigned off,const int8_t* data,size_t bytes) {
    check(off,true);
    const unsigned end=off<0x5000 ? 0x5000:0x9000;
    if(!data || !bytes || bytes%16 || off>=0x9000 || bytes>end-off)
        throw std::out_of_range("Invalid operand image");
#if defined(__arm__) && defined(__ARM_NEON)
    const unsigned stage=off<0x5000 ? 0:0x8000;
    auto* target=dma_ ? staging_+stage:base_+off;
    const auto copy_start=profile_now(profile);
    barrier();
    for(size_t i=0;i<bytes;i+=16) {
        const auto v=vld1q_s8(data+i);
        asm volatile("vst1.8 {%e0-%f0}, [%1 :128]" :: "w"(v),"r"(target+i):"memory");
    }
    barrier();
    profile_add(profile,MatmulSocProfile::input_stage_copy,copy_start);
    if(dma_) dma_transfer(0,0x30000000+stage,off,bytes);
#else
    throw std::runtime_error("128-bit ARM NEON access required");
#endif
}
void MatmulSocUio::read_buffer(unsigned off,int8_t* data,size_t bytes) {
    check(off,true);
    if(!data || !bytes || bytes%16 || off<0x9000 || bytes>0x19000-off)
        throw std::out_of_range("Invalid result image");
#if defined(__arm__) && defined(__ARM_NEON)
    if(dma_) dma_transfer(1,off,0x30010000,bytes);
    const auto* source=dma_ ? staging_+0x10000:base_+off;
    const auto copy_start=profile_now(profile);
    barrier();
    for(size_t i=0;i<bytes;i+=16) {
        int8x16_t v;
        asm volatile("vld1.8 {%e0-%f0}, [%1 :128]" : "=w"(v):"r"(source+i):"memory");
        vst1q_s8(data+i,v);
    }
    barrier();
    profile_add(profile,MatmulSocProfile::output_stage_copy,copy_start);
#else
    throw std::runtime_error("128-bit ARM NEON access required");
#endif
}
void MatmulSocUio::native_check() {
    if(native_checked_)return;
    const auto version=read(0x90);
    if(!dma_ || (version!=0x4e493832 && version!=0x4e493833))throw std::runtime_error("Native NI82/NI83 DMA ABI required");
    const auto format=read(0xa0);
    if((version==0x4e493833 && format!=0x03181080u) ||
       (version==0x4e493832 && (format&0xfeffffffu)!=0x00181080u))throw std::runtime_error("Unsupported native format word");
    if(version==0x4e493833){native_a_capacity_=32768;native_b_offset_=0x9000;}
    native_b_capacity_=(format&0x01000000)?65536:16384;
    if(dma_regs_[16]&2u)throw std::runtime_error("Native local DMA busy");
    if(chain_requested_) {
        barrier();
        for(unsigned engine=0;engine<3;++engine)
            if(dma_regs_[8*engine+4]!=0x444d4331u)
                throw std::runtime_error("DMA chain requested but FPGA is missing DMC1 capability");
        barrier();
        chain_enabled_=true;
    }
    native_checked_=true;
}
MatmulSocIo::CachedBuffer MatmulSocUio::cache_buffer(const int8_t* data,size_t bytes) {
    native_check();
    if(!data || !bytes || bytes%16 || bytes>staging_span_-cache_next_)
        throw std::invalid_argument("Native excluded-DDR weight cache exhausted");
    CachedBuffer c;c.owner=this;c.physical=0x30000000+uint32_t(cache_next_);c.size=bytes;
#if defined(__arm__) && defined(__ARM_NEON)
    barrier();
    for(size_t i=0;i<bytes;i+=16) {
        const auto v=vld1q_s8(data+i);
        asm volatile("vst1.8 {%e0-%f0}, [%1 :128]" :: "w"(v),"r"(staging_+cache_next_+i):"memory");
    }
    barrier();
#endif
    cache_next_+=bytes;return c;
}
void MatmulSocUio::write_cached(unsigned off,const CachedBuffer& c) {
    native_check();
    if(c.owner!=this || c.physical<0x30020000 || !c.size || c.size%16 ||
       uint64_t(c.physical)+c.size>0x30000000ULL+cache_next_ ||
       off!=native_b_offset_ || c.size>native_b_capacity_)throw std::invalid_argument("Invalid native weight cache descriptor");
    dma_transfer(0,c.physical,off,c.size);
}
void MatmulSocUio::write_nc4(unsigned off,const int8_t* data,size_t rows,size_t channels,
    size_t first,size_t count,size_t mp,size_t kp) {
    native_check();
    if(off!=0x1000 || !data || !count || first+count>rows || count>mp || channels>kp ||
       channels%4 || mp%16 || kp%16 || mp*kp>native_a_capacity_)throw std::invalid_argument("Invalid native NC4 input");
#if defined(__arm__) && defined(__ARM_NEON)
    // Padded M rows only contribute to their own discarded output rows.
    // Keep all K lanes of every live row, including the required zero tail.
    const size_t transfer_rows=transfer_live_rows ? count : mp;
    const auto start=profile_now(profile);barrier();
    auto* target=reinterpret_cast<volatile int8_t*>(direct_a_ ? base_+off : staging_);
    if(stream_stage_) {
        auto* cached=reinterpret_cast<int8_t*>((reinterpret_cast<uintptr_t>(stage_cache_.data())+15)&~uintptr_t(15));
        nc4_dma::pack(cached,data,rows,channels,first,count,transfer_rows,kp);
        for(size_t i=0;i<transfer_rows*kp;i+=16)
            nc4_dma::store(target+i,vld1q_u32(reinterpret_cast<uint32_t*>(cached+i)));
    } else nc4_dma::pack(target,data,rows,channels,first,count,transfer_rows,kp);
    barrier();profile_add(profile,MatmulSocProfile::input_stage_copy,start);
    if(!direct_a_) dma_transfer(0,0x30000000,off,transfer_rows*kp);
#else
    MatmulSocIo::write_nc4(off,data,rows,channels,first,count,mp,kp);
#endif
}
void MatmulSocUio::read_nc4(unsigned off,int8_t* data,size_t rows,size_t channels,
    size_t first,size_t column,size_t count,size_t columns,size_t mp,size_t np) {
    native_check();
    if(!data || !count || !columns || first+count>rows || column+columns>channels || count>mp || columns>np ||
       column%4 || columns%4 || !mp || !np || mp%16 || np%16 || off%16 || off<0x9000 || off>=0x19000 || mp*np>0x19000-off)
        throw std::invalid_argument("Invalid native NC4 output");
#if defined(__arm__) && defined(__ARM_NEON)
    // The previous tile can unpack during the following upload and GEMM.
    // Wait before reusing the DMA destination at the next readback.
    if(output_reader_)collect_output_copy();
    if(!direct_c_) dma_transfer(1,off,0x30010000,(transfer_live_rows ? count : mp)*np);
    const auto* source=reinterpret_cast<const volatile int8_t*>(direct_c_ ? base_+off : staging_+0x10000);
    if(output_reader_) {
        output_reader_->submit({data,source,rows,first,column,count,columns,np,profile!=nullptr});
        return;
    }
    const auto start=profile_now(profile);barrier();
    if(stream_stage_) {
        auto* cached=reinterpret_cast<int8_t*>((reinterpret_cast<uintptr_t>(stage_cache_.data())+15)&~uintptr_t(15));
        for(size_t i=0;i<mp*np;i+=16)
            vst1q_u32(reinterpret_cast<uint32_t*>(cached+i),nc4_dma::load(source+i));
        nc4_dma::unpack(data,cached,rows,first,column,count,columns,np);
    } else nc4_dma::unpack(data,source,rows,first,column,count,columns,np);
    barrier();profile_add(profile,MatmulSocProfile::output_stage_copy,start);
#else
    MatmulSocIo::read_nc4(off,data,rows,channels,first,column,count,columns,mp,np);
#endif
}
void MatmulSocUio::feedback(unsigned source,unsigned destination,size_t bytes) {
    native_check();
    if(source<0x9000 || source%16 || destination!=0x1000 || !bytes || bytes%16 ||
       bytes>native_a_capacity_ || source+bytes>0x18000)throw std::invalid_argument("Invalid resident feedback span");
    dma_transfer(2,source,destination,bytes);
}
void MatmulSocUio::write_rows(unsigned off,const int8_t* data,size_t rows,size_t width,size_t padded) {
    native_check();
    if(off!=0x1000 || !data || !rows || rows>padded || !width || width%16 || padded>native_a_capacity_/width)
        throw std::invalid_argument("Invalid native row-major input");
#if defined(__arm__) && defined(__ARM_NEON)
    const auto start=profile_now(profile);barrier();
    auto* target=reinterpret_cast<volatile int8_t*>(direct_a_ ? base_+off : staging_);
    const size_t active_bytes=rows*width,total_bytes=(transfer_live_rows ? rows : padded)*width;
    for(size_t i=0;i<active_bytes;i+=16)
        nc4_dma::store(target+i,vreinterpretq_u32_s8(vld1q_s8(data+i)));
    for(size_t i=active_bytes;i<total_bytes;i+=16)nc4_dma::store(target+i,vdupq_n_u32(0));
    barrier();profile_add(profile,MatmulSocProfile::input_stage_copy,start);
    if(!direct_a_)dma_transfer(0,0x30000000,off,total_bytes);
#else
    MatmulSocIo::write_rows(off,data,rows,width,padded);
#endif
}
void MatmulSocUio::read_rows(unsigned off,int8_t* data,size_t rows,size_t width,size_t stride,size_t padded) {
    native_check();
    if(off!=0x9000 || !data || !rows || rows>padded || !width || width%16 || stride<width || padded>65536/width)
        throw std::invalid_argument("Invalid native row-major output");
#if defined(__arm__) && defined(__ARM_NEON)
    if(!direct_c_)dma_transfer(1,off,0x30010000,(transfer_live_rows ? rows : padded)*width);
    const auto start=profile_now(profile);barrier();
    const auto* source=reinterpret_cast<const volatile int8_t*>(direct_c_ ? base_+off : staging_+0x10000);
    for(size_t r=0;r<rows;++r)for(size_t j=0;j<width;j+=16)
        vst1q_s8(data+r*stride+j,vreinterpretq_s8_u32(nc4_dma::load(source+r*width+j)));
    barrier();profile_add(profile,MatmulSocProfile::output_stage_copy,start);
#else
    MatmulSocIo::read_rows(off,data,rows,width,stride,padded);
#endif
}
}
