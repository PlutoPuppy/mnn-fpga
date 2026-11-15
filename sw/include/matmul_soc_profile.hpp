#pragma once
#include <array>
#include <chrono>
#include <cstdint>

namespace fpga {
// Optional host wall timers. Transfer subphases are nested in upload/download.
// completion_wait includes MMIO polling and scheduling, not just fabric cycles.
struct MatmulSocProfile {
    enum Phase { peak_scan, quantize, workspace, pack_pad, scale_bound,
        configure, upload, launch, completion_wait, download, dequantize,
        output_convert, adapter_total, device_init, bridge_total,
        input_stage_copy, input_dma_setup, input_dma_wait,
        output_dma_setup, output_dma_wait, output_stage_copy, tile_plan,
        layout_allocate, input_layout, output_layout, int8_accumulate,
        int8_requantize, local_feedback, local_dma_setup, local_dma_wait, phase_count };
    std::array<uint64_t,phase_count> ns{};
    uint64_t jobs=0, input_bytes=0, output_bytes=0, quantized_values=0;
    uint64_t input_descriptors=0, output_descriptors=0;
    uint64_t input_polls=0, output_polls=0, compute_polls=0;
    uint64_t tile_m=0,tile_n=0,tile_k=0;
    uint64_t int8_vectorized_values=0;
    uint64_t local_descriptors=0,local_polls=0,local_bytes=0;
};
using ProfileClock=std::chrono::steady_clock;
inline ProfileClock::time_point profile_now(MatmulSocProfile* p) {
    return p ? ProfileClock::now():ProfileClock::time_point{};
}
inline void profile_add(MatmulSocProfile* p,MatmulSocProfile::Phase phase,
                        ProfileClock::time_point start) {
    if(p) p->ns[phase]+=std::chrono::duration_cast<std::chrono::nanoseconds>(ProfileClock::now()-start).count();
}
// Consecutive checkpoints define disjoint phases inside one scope.
class ProfileSteps {
    MatmulSocProfile* p_; ProfileClock::time_point last_;
public:
    explicit ProfileSteps(MatmulSocProfile* p):p_(p),last_(profile_now(p)) {}
    void reset() {last_=profile_now(p_);}
    void end(MatmulSocProfile::Phase phase) {
        if(p_) {const auto now=ProfileClock::now();p_->ns[phase]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-last_).count();last_=now;}
    }
};
}
