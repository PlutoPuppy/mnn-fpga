#include "fpga_bridge.hpp"
#include "uio.hpp"
#include "matmul_soc_uio.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace {
std::atomic<unsigned long> completed{0};
std::mutex device_mutex;
std::unique_ptr<fpga::Uio> device;
std::unique_ptr<fpga::MatmulSocUio> soc_device;
std::atomic<bool> selected{false};
bool soc_failed=false;
}
extern "C" void mnn_fpga_select_operation(bool active) { selected.store(active); }
bool mnn_fpga_try_matmul(const float* a,const float* b,float* c,
    size_t m,size_t n,size_t k,bool ta,bool tb,bool tc,const float* bias) {
    const char* path=std::getenv("MNN_MATMUL_SOC_UIO");
    const char* physical=std::getenv("MNN_MATMUL_SOC_DEVMEM");
    const bool devmem=physical && std::string(physical)=="1";
    if (devmem && path && *path) return false;
    if (devmem) path="/dev/mem";
    if (!selected.load()) return false;
    if (!path || !*path) {
        path=std::getenv("MNN_FPGA_UIO");
        if (!path || !*path || bias) return false;
        std::lock_guard<std::mutex> lock(device_mutex);
        try {
            if (!device) device.reset(new fpga::Uio(path));
            size_t jobs=0;
            if (!fpga::exact_float_gemm(*device,a,b,c,m,n,k,ta,tb,tc,&jobs)) return false;
            completed.fetch_add(jobs);return true;
        } catch (const std::exception& e) {
            std::fprintf(stderr,"MNN FPGA: %s; using CPU\n",e.what());device.reset();return false;
        }
    }
    std::lock_guard<std::mutex> lock(device_mutex);
    if (soc_failed) return false;
    try {
        if (!soc_device) soc_device.reset(new fpga::MatmulSocUio(path,devmem));
        auto stats=fpga::matmul_soc_gemm(*soc_device,a,b,c,m,n,k,ta,tb,tc,bias);
        completed.fetch_add(stats.submissions);
        std::fprintf(stderr,"MNN matmul-soc: %zux%zu x %zux%zu, %zu FPGA submissions (approximate INT8)\n",m,k,k,n,stats.submissions);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"MNN matmul-soc: %s; disabling device for this process, using CPU\n",e.what());
        soc_failed=true;
        soc_device.reset();
        return false;
    }
}
extern "C" unsigned long mnn_fpga_completed_jobs() { return completed.load(); }
bool mnn_fpga_try_tile(const float* a, const float* b, float* c) {
    const char* path = std::getenv("MNN_FPGA_UIO");
    if (!path || !*path) return false;
    std::lock_guard<std::mutex> lock(device_mutex);
    try {
        if (!device) device.reset(new fpga::Uio(path));
        if (!fpga::exact_float_tile(*device,a,b,c)) return false;
        ++completed;
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"MNN FPGA: %s; using CPU for this operation\n",e.what());
        device.reset();
        return false;
    }
}
