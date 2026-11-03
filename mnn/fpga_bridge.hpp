#pragma once
#include <cstddef>
// Returns false for unsupported values, disabled offload, or device errors.
bool mnn_fpga_try_tile(const float* a, const float* b, float* c);
extern "C" __attribute__((visibility("default"))) unsigned long mnn_fpga_completed_jobs();
// Callback-scoped selection of one named runtime operation. Enable only around
// synchronous execution; separate concurrent sessions are not supported.
extern "C" __attribute__((visibility("default"))) void mnn_fpga_select_operation(bool active);
bool mnn_fpga_try_matmul(const float* a,const float* b,float* c,
    size_t m,size_t n,size_t k,bool ta,bool tb,bool tc,const float* bias);
