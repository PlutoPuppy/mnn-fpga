#pragma once
// Returns false for unsupported values, disabled offload, or device errors.
bool mnn_fpga_try_tile(const float* a, const float* b, float* c);
extern "C" __attribute__((visibility("default"))) unsigned long mnn_fpga_completed_jobs();
