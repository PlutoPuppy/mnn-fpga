#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
// Returns false for unsupported values, disabled offload, or device errors.
bool mnn_fpga_try_tile(const float* a, const float* b, float* c);
extern "C" __attribute__((visibility("default"))) unsigned long mnn_fpga_completed_jobs();
// Callback-scoped selection of one named runtime operation. Enable only around
// synchronous execution; separate concurrent sessions are not supported.
extern "C" __attribute__((visibility("default"))) void mnn_fpga_select_operation(bool active);
// One synchronous session at a time; returned JSON storage lasts until next call.
extern "C" __attribute__((visibility("default"))) void mnn_fpga_profile_reset(bool enabled);
extern "C" __attribute__((visibility("default"))) const char* mnn_fpga_profile_json();
// Static FP32 OI weights are prepared once before inference. Linear execution
// replaces one callback-selected, dense H=W=1 convolution; errors return false.
bool mnn_fpga_prepare_linear(const float* weights,const float* bias,size_t m,size_t k,size_t n);
bool mnn_fpga_try_linear(const float* input,float* output,size_t m,size_t k,size_t n);
const char* mnn_fpga_linear_setup_json();
// Application tensor conversion is outside adapter/bridge timing.
void mnn_fpga_record_linear_layout(uint64_t allocation_ns,uint64_t input_ns,uint64_t output_ns);
bool mnn_fpga_prepare_linear_int8(const int8_t* weights_oi,const float* weight_scales,
    const float* bias,size_t m,size_t k,size_t n,float input_scale,float output_scale,int minimum,int maximum);
bool mnn_fpga_linear_int8_active();
// Native MNN NC4HW4 bytes, pack=4, H=W=1. 1=completed, -1=device/contract error.
int mnn_fpga_try_linear_int8(const int8_t* input,int8_t* output,size_t m,size_t k,size_t n,
    float input_scale,float output_scale,float input_zero,float output_zero);
unsigned long mnn_fpga_completed_int8_operations();
// Multi-operation NI82 interface. Preparation and cached-weight upload happen
// before inference. Selection is callback-scoped, synchronous and fail-closed.
void mnn_fpga_register_native(const std::string& name,const int8_t* weights,const float* scales,
    const float* bias,size_t k,size_t n,float input_scale,float output_scale);
void mnn_fpga_register_native_pair(const std::string& first,const std::string& second,
    const std::array<int8_t,256>& table,const std::vector<std::string>& skipped);
bool mnn_fpga_select_native(const std::string& name);
bool mnn_fpga_native_skip(const std::string& name);
size_t mnn_fpga_native_count();
const char* mnn_fpga_native_json();
uint64_t mnn_fpga_cast_profile_start();
void mnn_fpga_cast_profile_end(uint64_t start,bool quantize,size_t values);
// Optional diagnostic dumps; disabled unless selected and MNN_INT8_TRACE is set.
// Captures native INT8 buffers inside the executor, after runtime casts.
struct MnnFpgaInt8Trace {
    const char* prefix=nullptr;const int8_t* output=nullptr;size_t m=0,n=0;
    MnnFpgaInt8Trace(const int8_t* input,const int8_t* out,size_t rows,size_t k,size_t cols,bool valid);
    ~MnnFpgaInt8Trace();
};
std::uint64_t mnn_fpga_cpu_profile_start();
// CPU execute durations are summed across both MNN threads; these scopes
// overlap and their sum is CPU thread-time, NOT selected-operation wall time.
void mnn_fpga_cpu_profile_end(std::uint64_t start);
void mnn_fpga_cpu_kernel_profile_end(std::uint64_t start);
struct MnnFpgaCpuProfileScope {
    std::uint64_t start=mnn_fpga_cpu_profile_start();
    ~MnnFpgaCpuProfileScope() {mnn_fpga_cpu_profile_end(start);}
};
struct MnnFpgaCpuKernelProfileScope {
    std::uint64_t start=mnn_fpga_cpu_profile_start();
    ~MnnFpgaCpuKernelProfileScope() {mnn_fpga_cpu_kernel_profile_end(start);}
};
bool mnn_fpga_try_matmul(const float* a,const float* b,float* c,
    size_t m,size_t n,size_t k,bool ta,bool tb,bool tc,const float* bias);
