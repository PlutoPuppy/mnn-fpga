#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#include "matmul_soc_uio.hpp"
#include "nc4_dma.hpp"
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <stdexcept>
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
MatmulSocUio::MatmulSocUio(const std::string& path,bool devmem,bool dma) {
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
    try { if(dma) enable_dma(); }
    catch(...) {
        if(dma_regs_) munmap(const_cast<uint32_t*>(dma_regs_),4096);
        if(staging_) munmap(const_cast<uint8_t*>(staging_),staging_span_);
        munmap(const_cast<uint8_t*>(base_),span);base_=nullptr;
        close(fd_);fd_=-1;throw;
    }
#endif
}
MatmulSocUio::~MatmulSocUio() {
    if(dma_) {
        dma_regs_[6]=0;dma_regs_[14]=0;if(native_checked_)dma_regs_[22]=0;barrier();
    }
    if(dma_regs_) munmap(const_cast<uint32_t*>(dma_regs_),4096);
    if(staging_) munmap(const_cast<uint8_t*>(staging_),staging_span_);
    if (base_) munmap(const_cast<uint8_t*>(base_),span);
    if (fd_>=0) close(fd_);
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
    const size_t descriptor_bytes=native_checked_ ? (engine==2?256:2048):8192;
    for(size_t off=0;off<bytes;off+=descriptor_bytes) {
        ProfileSteps timer(profile);
        const auto count=uint32_t(std::min(descriptor_bytes,bytes-off));
        regs[6]=0;regs[3]=0;regs[0]=0;barrier();
        regs[1]=source+uint32_t(off);regs[2]=destination+uint32_t(off);regs[3]=count;
        barrier();regs[6]=(1u<<3)|(1u<<7)|(1u<<11);barrier();
        timer.end(engine==2 ? MatmulSocProfile::local_dma_setup:
                  engine ? MatmulSocProfile::output_dma_setup:MatmulSocProfile::input_dma_setup);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(1000);
        for(;;) {
            barrier();const auto status=regs[0];barrier();
            if(profile) {if(engine==2)++profile->local_polls;else if(engine) ++profile->output_polls;else ++profile->input_polls;}
            if(status&1u) break;
            if(std::chrono::steady_clock::now()>=deadline) {
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
    if(!dma_ || read(0x90)!=0x4e493832)throw std::runtime_error("Native NI82 DMA ABI required");
    const auto format=read(0xa0);
    if((format&0x00ffffffu)!=0x00181080u || (format&0xfe000000u))throw std::runtime_error("Unsupported NI82 format word");
    native_b_capacity_=(format&0x01000000)?65536:16384;
    if(dma_regs_[16]&2u)throw std::runtime_error("Native local DMA busy");
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
       off!=0x5000 || c.size>native_b_capacity_)throw std::invalid_argument("Invalid native weight cache descriptor");
    dma_transfer(0,c.physical,off,c.size);
}
void MatmulSocUio::write_nc4(unsigned off,const int8_t* data,size_t rows,size_t channels,
    size_t first,size_t count,size_t mp,size_t kp) {
    native_check();
    if(off!=0x1000 || !data || !count || first+count>rows || count>mp || channels>kp ||
       channels%4 || mp%16 || kp%16 || mp*kp>16384)throw std::invalid_argument("Invalid native NC4 input");
#if defined(__arm__) && defined(__ARM_NEON)
    const auto start=profile_now(profile);barrier();
    nc4_dma::pack(reinterpret_cast<volatile int8_t*>(staging_),data,rows,channels,first,count,mp,kp);
    barrier();profile_add(profile,MatmulSocProfile::input_stage_copy,start);
    dma_transfer(0,0x30000000,off,mp*kp);
#else
    MatmulSocIo::write_nc4(off,data,rows,channels,first,count,mp,kp);
#endif
}
void MatmulSocUio::read_nc4(unsigned off,int8_t* data,size_t rows,size_t channels,
    size_t first,size_t column,size_t count,size_t columns,size_t mp,size_t np) {
    native_check();
    if(!data || first+count>rows || column+columns>channels || count>mp || columns>np ||
       column%4 || columns%4 || !mp || !np || mp%16 || np%16 || off%16 || off<0x9000 || off>=0x19000 || mp*np>0x19000-off)
        throw std::invalid_argument("Invalid native NC4 output");
#if defined(__arm__) && defined(__ARM_NEON)
    dma_transfer(1,off,0x30010000,mp*np);
    const auto start=profile_now(profile);barrier();
    nc4_dma::unpack(data,reinterpret_cast<const volatile int8_t*>(staging_+0x10000),rows,first,column,count,columns,np);
    barrier();profile_add(profile,MatmulSocProfile::output_stage_copy,start);
#else
    MatmulSocIo::read_nc4(off,data,rows,channels,first,column,count,columns,mp,np);
#endif
}
void MatmulSocUio::feedback(unsigned source,unsigned destination,size_t bytes) {
    native_check();
    if(source<0x9000 || source%16 || destination!=0x1000 || !bytes || bytes%16 ||
       bytes>16384 || source+bytes>0x18000)throw std::invalid_argument("Invalid resident feedback span");
    dma_transfer(2,source,destination,bytes);
}
}
