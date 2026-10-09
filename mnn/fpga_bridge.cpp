#include "fpga_bridge.hpp"
#include "uio.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace {
std::atomic<unsigned long> completed{0};
std::mutex device_mutex;
std::unique_ptr<fpga::Uio> device;
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
