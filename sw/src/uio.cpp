#include "uio.hpp"
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sys/file.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace fpga {
namespace {
void barrier() {
#if defined(__arm__) || defined(__aarch64__)
    asm volatile("dmb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}
unsigned long metadata(const std::string& name, const char* field) {
    std::ifstream in("/sys/class/uio/"+name+"/maps/map0/"+field);
    unsigned long value = 0;
    if (!(in >> std::hex >> value)) throw std::runtime_error("Cannot read UIO map0 metadata");
    return value;
}
void check_offset(unsigned offset) {
    if (offset >= Span || offset % 4) throw std::out_of_range("Invalid MMIO offset");
}
}
Uio::Uio(const std::string& path) {
    auto name = path.substr(path.find_last_of('/')+1);
    if (name.size() <= 3 || name.substr(0,3) != "uio" ||
        name.find_first_not_of("0123456789",3) != std::string::npos)
        throw std::invalid_argument("Use /dev/uioN for the dedicated accelerator");
    if (metadata(name,"size") < Span || metadata(name,"offset") != 0)
        throw std::runtime_error("UIO map0 must be page-aligned and at least 4096 bytes");
    fd_ = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error(std::string("open UIO: ")+std::strerror(errno));
    if (flock(fd_, LOCK_EX | LOCK_NB) < 0) {
        close(fd_); fd_=-1; throw std::runtime_error("UIO is already locked by another process");
    }
    void* p = mmap(nullptr, Span, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
        close(fd_); fd_=-1; throw std::runtime_error("mmap UIO failed");
    }
    base_ = static_cast<volatile uint32_t*>(p);
}
Uio::~Uio() {
    if (base_) munmap(const_cast<uint32_t*>(base_), Span);
    if (fd_ >= 0) close(fd_);
}
uint32_t Uio::read(unsigned offset) {
    check_offset(offset); barrier(); uint32_t value=base_[offset/4]; barrier(); return value;
}
void Uio::write(unsigned offset, uint32_t value) {
    check_offset(offset); barrier(); base_[offset/4]=value; barrier();
}
}
