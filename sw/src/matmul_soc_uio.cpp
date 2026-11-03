#include "matmul_soc_uio.hpp"
#include <fstream>
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
    if (buffer ? (off%16 || !((off>=0x1000 && off<0x9000) || (off>=0x9000 && off<0x19000)))
               : (off%16 || off>0x80)) throw std::out_of_range("Invalid matmul-soc offset");
}
}
MatmulSocUio::MatmulSocUio(const std::string& path,bool devmem) {
#if !defined(__arm__) || !defined(__ARM_NEON)
    (void)path;(void)devmem;
    throw std::runtime_error("matmul-soc PIO requires ARMv7 NEON 128-bit device accesses");
#else
    if (devmem && path!="/dev/mem") throw std::invalid_argument("Physical transport requires /dev/mem");
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
#endif
}
MatmulSocUio::~MatmulSocUio() {
    if (base_) munmap(const_cast<uint8_t*>(base_),span);
    if (fd_>=0) close(fd_);
}
uint32_t MatmulSocUio::read(unsigned off) {
    check(off,false);barrier();const auto v=*reinterpret_cast<volatile uint32_t*>(base_+off);barrier();return v;
}
void MatmulSocUio::write(unsigned off,uint32_t v) {
    check(off,false);barrier();*reinterpret_cast<volatile uint32_t*>(base_+off)=v;barrier();
}
void MatmulSocUio::write_beat(unsigned off,const Beat& data) {
    check(off,true);
    if (off>=0x9000) throw std::out_of_range("Output buffer is read-only");
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
}
