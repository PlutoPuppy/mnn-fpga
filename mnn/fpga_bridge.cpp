#include "fpga_bridge.hpp"
#include "uio.hpp"
#include "matmul_soc_uio.hpp"
#include "matmul_soc_native.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <map>
#include <set>

namespace {
std::atomic<unsigned long> completed{0};
std::mutex device_mutex;
std::unique_ptr<fpga::Uio> device;
std::unique_ptr<fpga::MatmulSocUio> soc_device;
std::atomic<bool> selected{false};
bool soc_failed=false;
fpga::MatmulSocProfile profile;
bool profiling=false;
fpga::MatmulSocOperand linear_weights;
fpga::MatmulSocPlan linear_plan;
std::vector<float> linear_bias;
fpga::MatmulSocProfile weight_profile;
double weight_setup_ms=0,weight_plan_ms=0;
bool linear_ready=false;
bool linear_int8=false;
float linear_input_scale=0,linear_output_scale=0;
fpga::MatmulSocInt8Requantization linear_requant;
std::atomic<unsigned long> completed_int8{0};
struct CastProfile {
    uint64_t quantize_ns=0,dequantize_ns=0,quantize_calls=0,dequantize_calls=0;
    uint64_t quantized_values=0,dequantized_values=0,selected_quantize_ns=0,selected_dequantize_ns=0;
} cast_profile;
std::atomic<std::uint64_t> cpu_execute_ns{0},cpu_execute_calls{0};
struct CpuIntervals {
    std::array<std::pair<uint64_t,uint64_t>,4096> spans{};
    std::atomic<unsigned> count{0};
    void add(uint64_t start,uint64_t end) {
        const auto i=count.fetch_add(1);
        if(i<spans.size()) spans[i]={start,end};
    }
    // Union gives elapsed wall time with at least one CPU thread in this scope.
    std::pair<uint64_t,uint64_t> durations() const {
        const auto n=std::min(size_t(count.load()),spans.size());
        std::vector<std::pair<uint64_t,uint64_t>> sorted(spans.begin(),spans.begin()+n);
        std::sort(sorted.begin(),sorted.end());
        uint64_t sum=0,wall=0,end=0;
        for(const auto& s:sorted) {sum+=s.second-s.first;wall+=s.second>std::max(end,s.first) ? s.second-std::max(end,s.first):0;end=std::max(end,s.second);}
        return {sum,wall};
    }
} cpu_execute_intervals,cpu_kernel_intervals;
struct NativeContext {
    std::string name;
    fpga::NativeInt8Weights weights;
    float input_scale=0,output_scale=0;
    int second=-1;
    std::array<int8_t,256> lut{};
    std::set<std::string> skipped;
};
struct NativeState {
    std::vector<NativeContext> contexts;
    std::map<std::string,int> names;
    int active=-1,pending=-1;
    size_t pending_rows=0;
    std::vector<int8_t> output;
    uint64_t local_bytes=0,hidden_bytes_avoided=0;
    uint64_t fused_pairs=0,skipped_ops=0;
};
NativeState& natives(){static NativeState state;return state;}
}
std::uint64_t mnn_fpga_cpu_profile_start() {
    if(!profiling || !selected.load()) return 0;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(fpga::ProfileClock::now().time_since_epoch()).count();
}
void mnn_fpga_cpu_profile_end(std::uint64_t start) {
    if(!start) return;
    const auto end=std::chrono::duration_cast<std::chrono::nanoseconds>(fpga::ProfileClock::now().time_since_epoch()).count();
    cpu_execute_ns.fetch_add(end-start);cpu_execute_calls.fetch_add(1);
    cpu_execute_intervals.add(start,end);
}
void mnn_fpga_cpu_kernel_profile_end(std::uint64_t start) {
    if(!start) return;
    const auto end=std::chrono::duration_cast<std::chrono::nanoseconds>(fpga::ProfileClock::now().time_since_epoch()).count();
    cpu_kernel_intervals.add(start,end);
}
extern "C" void mnn_fpga_profile_reset(bool enabled) {
    std::lock_guard<std::mutex> lock(device_mutex);
    profile={};cast_profile={};profiling=enabled;
    natives().local_bytes=0;natives().hidden_bytes_avoided=0;natives().fused_pairs=0;natives().skipped_ops=0;
    cpu_execute_ns=0;cpu_execute_calls=0;
    cpu_execute_intervals.count=0;cpu_kernel_intervals.count=0;
    if(soc_device) soc_device->profile=enabled ? &profile:nullptr;
}
extern "C" const char* mnn_fpga_profile_json() {
    std::lock_guard<std::mutex> lock(device_mutex);
    static std::string result;
    static const char* names[]={"peak_scan","quantize","workspace","pack_pad","scale_bound",
        "configure","upload","launch","completion_wait","download","dequantize",
        "output_convert","adapter_total","device_init","bridge_total",
        "input_stage_copy","input_dma_setup","input_dma_wait","output_dma_setup",
        "output_dma_wait","output_stage_copy","tile_plan","layout_allocate","input_layout","output_layout",
        "int8_accumulate","int8_requantize","local_feedback","local_dma_setup","local_dma_wait"};
    static_assert(sizeof(names)/sizeof(names[0])==fpga::MatmulSocProfile::phase_count,"profile names");
    std::ostringstream out;out<<std::setprecision(12)<<"{\"ms\":{";
    for(size_t i=0;i<profile.ns.size();++i) {if(i) out<<',';out<<'"'<<names[i]<<"\":"<<double(profile.ns[i])/1e6;}
    out<<"},\"jobs\":"<<profile.jobs<<",\"input_bytes\":"<<profile.input_bytes
       <<",\"output_bytes\":"<<profile.output_bytes<<",\"quantized_values\":"<<profile.quantized_values
       <<",\"input_descriptors\":"<<profile.input_descriptors<<",\"output_descriptors\":"<<profile.output_descriptors
       <<",\"input_polls\":"<<profile.input_polls<<",\"output_polls\":"<<profile.output_polls
       <<",\"compute_polls\":"<<profile.compute_polls
       <<",\"local_descriptors\":"<<profile.local_descriptors<<",\"local_polls\":"<<profile.local_polls
       <<",\"local_bytes\":"<<profile.local_bytes
       <<",\"tile_m\":"<<profile.tile_m<<",\"tile_n\":"<<profile.tile_n<<",\"tile_k\":"<<profile.tile_k
       <<",\"int8_vectorized_values\":"<<profile.int8_vectorized_values
       <<",\"cpu_execute_ms\":"<<double(cpu_execute_ns.load())/1e6
       <<",\"cpu_execute_calls\":"<<cpu_execute_calls.load()
       <<",\"cpu_execute_union_ms\":"<<double(cpu_execute_intervals.durations().second)/1e6
       <<",\"cpu_kernel_thread_ms\":"<<double(cpu_kernel_intervals.durations().first)/1e6
       <<",\"cpu_kernel_union_ms\":"<<double(cpu_kernel_intervals.durations().second)/1e6
       <<",\"cpu_kernel_calls\":"<<cpu_kernel_intervals.count.load()
       <<",\"cpu_interval_overflow\":"<<((cpu_execute_intervals.count>4096 || cpu_kernel_intervals.count>4096)?1:0)
       <<",\"framework_casts\":{\"quantize_ms\":"<<double(cast_profile.quantize_ns)/1e6
       <<",\"dequantize_ms\":"<<double(cast_profile.dequantize_ns)/1e6
       <<",\"quantize_calls\":"<<cast_profile.quantize_calls<<",\"dequantize_calls\":"<<cast_profile.dequantize_calls
       <<",\"quantized_values\":"<<cast_profile.quantized_values<<",\"dequantized_values\":"<<cast_profile.dequantized_values
       <<",\"selected_quantize_ms\":"<<double(cast_profile.selected_quantize_ns)/1e6
       <<",\"selected_dequantize_ms\":"<<double(cast_profile.selected_dequantize_ns)/1e6<<"}}";
    result=out.str();return result.c_str();
}
extern "C" void mnn_fpga_select_operation(bool active) { selected.store(active); }
uint64_t mnn_fpga_cast_profile_start() {
    if(!profiling) return 0;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(fpga::ProfileClock::now().time_since_epoch()).count();
}
void mnn_fpga_cast_profile_end(uint64_t start,bool quantize,size_t values) {
    if(!start) return;
    const auto end=std::chrono::duration_cast<std::chrono::nanoseconds>(fpga::ProfileClock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(device_mutex);const uint64_t elapsed=end-start;
    if(quantize) {cast_profile.quantize_ns+=elapsed;++cast_profile.quantize_calls;cast_profile.quantized_values+=values;
        if(selected.load()) cast_profile.selected_quantize_ns+=elapsed;}
    else {cast_profile.dequantize_ns+=elapsed;++cast_profile.dequantize_calls;cast_profile.dequantized_values+=values;
        if(selected.load()) cast_profile.selected_dequantize_ns+=elapsed;}
}
void mnn_fpga_record_linear_layout(uint64_t allocation_ns,uint64_t input_ns,uint64_t output_ns) {
    std::lock_guard<std::mutex> lock(device_mutex);
    if(!profiling) return;
    profile.ns[fpga::MatmulSocProfile::layout_allocate]+=allocation_ns;
    profile.ns[fpga::MatmulSocProfile::input_layout]+=input_ns;
    profile.ns[fpga::MatmulSocProfile::output_layout]+=output_ns;
}
bool mnn_fpga_prepare_linear(const float* weights,const float* bias,size_t m,size_t k,size_t n) {
    std::lock_guard<std::mutex> lock(device_mutex);
    try {
        const auto start=fpga::ProfileClock::now();weight_profile={};linear_ready=false;linear_int8=false;
        linear_weights=fpga::prepare_matmul_soc_operand(weights,k,n,true,&weight_profile);
        linear_bias.assign(n,0);if(bias) std::copy_n(bias,n,linear_bias.begin());
        const char* mode=std::getenv("MNN_MATMUL_SOC_TILE_POLICY");
        const auto policy=mode && std::string(mode)=="fixed64" ? fpga::MatmulSocTilePolicy::Fixed64:fpga::MatmulSocTilePolicy::Capacity;
        const auto plan_start=fpga::ProfileClock::now();
        linear_plan=fpga::plan_matmul_soc(m,n,k,policy);
        weight_plan_ms=std::chrono::duration<double,std::milli>(fpga::ProfileClock::now()-plan_start).count();
        weight_setup_ms=std::chrono::duration<double,std::milli>(fpga::ProfileClock::now()-start).count();
        linear_ready=true;return true;
    } catch(const std::exception& e) {std::fprintf(stderr,"MNN linear prepare: %s\n",e.what());return false;}
}
const char* mnn_fpga_linear_setup_json() {
    static std::string result;std::ostringstream out;
    out<<std::setprecision(12)<<"{\"ready\":"<<(linear_ready?"true":"false")<<",\"weight_setup_ms\":"<<weight_setup_ms
       <<",\"weight_peak_scan_ms\":"<<double(weight_profile.ns[fpga::MatmulSocProfile::peak_scan])/1e6
       <<",\"weight_quantize_ms\":"<<double(weight_profile.ns[fpga::MatmulSocProfile::quantize])/1e6
       <<",\"plan_selection_ms\":"<<weight_plan_ms<<",\"weight_values\":"<<linear_weights.values.size()
       <<",\"M\":"<<linear_plan.m<<",\"N\":"<<linear_plan.n<<",\"K\":"<<linear_plan.k
       <<",\"tile_m\":"<<linear_plan.tile_m<<",\"tile_n\":"<<linear_plan.tile_n<<",\"tile_k\":"<<linear_plan.tile_k
       <<",\"jobs\":"<<linear_plan.jobs<<",\"operand_bytes\":"<<linear_plan.operand_bytes<<",\"result_bytes\":"<<linear_plan.result_bytes
       <<",\"accumulated_values\":"<<linear_plan.accumulated_values<<",\"cost_proxy_ns\":"<<linear_plan.cost_proxy_ns
       <<",\"policy\":\""<<(linear_plan.policy==fpga::MatmulSocTilePolicy::Capacity?"capacity":"fixed64")<<"\""
       <<",\"operand_format\":\""<<(linear_int8?"native_w8a8":"fp32_adapter")<<"\""
       <<",\"input_scale\":"<<linear_input_scale<<",\"output_scale\":"<<linear_output_scale
       <<",\"per_channel_weight_scales\":"<<(linear_int8?linear_plan.n:0)<<'}';
    result=out.str();return result.c_str();
}
bool mnn_fpga_try_linear(const float* input,float* output,size_t m,size_t k,size_t n) {
    if(!selected.load()) return false;
    std::lock_guard<std::mutex> lock(device_mutex);
    auto* timing=profiling ? &profile:nullptr;
    const auto bridge_start=fpga::profile_now(timing);
    try {
        if(!linear_ready || linear_int8 || linear_plan.m!=m || linear_plan.k!=k || linear_plan.n!=n || soc_failed)
            throw std::runtime_error("Linear shape is not prepared");
        const char* physical=std::getenv("MNN_MATMUL_SOC_DEVMEM");
        const char* dma_mode=std::getenv("MNN_MATMUL_SOC_DMA");
        const bool devmem=physical && std::string(physical)=="1";
        const bool dma=dma_mode && std::string(dma_mode)=="1";
        if(!devmem || !dma) throw std::runtime_error("Linear experiment requires explicit devmem and DMA");
        if(!soc_device) {
            const auto start=fpga::profile_now(timing);
            soc_device.reset(new fpga::MatmulSocUio("/dev/mem",true,true));
            fpga::profile_add(timing,fpga::MatmulSocProfile::device_init,start);
        }
        soc_device->profile=timing;
        const auto adapter_start=fpga::profile_now(timing);
        const auto a=fpga::prepare_matmul_soc_operand(input,m,k,false,timing);
        auto stats=fpga::matmul_soc_gemm_prepared(*soc_device,a,linear_weights,output,false,linear_bias.data(),
                                               std::chrono::milliseconds(1000),&linear_plan);
        stats.quantized_values=a.values.size();
        if(timing) {timing->jobs+=stats.submissions;timing->input_bytes+=stats.operand_bytes;
            timing->output_bytes+=stats.result_bytes;timing->quantized_values+=stats.quantized_values;}
        fpga::profile_add(timing,fpga::MatmulSocProfile::adapter_total,adapter_start);
        completed.fetch_add(stats.submissions);
        std::fprintf(stderr,"MNN linear FPGA: %zux%zu x %zux%zu; %zu jobs; tile=%zux%zux%zu; input=%zu B output=%zu B\n",
                     m,k,k,n,stats.submissions,linear_plan.tile_m,linear_plan.tile_k,linear_plan.tile_n,stats.operand_bytes,stats.result_bytes);
        fpga::profile_add(timing,fpga::MatmulSocProfile::bridge_total,bridge_start);return true;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"MNN linear FPGA failed: %s\n",e.what());soc_failed=true;soc_device.reset();return false;
    }
}
bool mnn_fpga_prepare_linear_int8(const int8_t* weights,const float* scales,const float* bias,
    size_t m,size_t k,size_t n,float si,float so,int minimum,int maximum) {
    std::lock_guard<std::mutex> lock(device_mutex);
    try {
        const auto start=fpga::ProfileClock::now();linear_ready=false;linear_int8=false;weight_profile={};
        if(!weights || !scales || !bias || m!=784 || k!=128 || n!=512 ||
            !std::isfinite(si) || !std::isfinite(so) || si<=0 || so<=0 || minimum< -128 || maximum>127 || minimum>maximum)
            throw std::invalid_argument("Unsupported native fc1 quantization");
        linear_weights.rows=k;linear_weights.cols=n;linear_weights.scale=1;linear_weights.values.resize(k*n);
        for(size_t p=0;p<k;++p) for(size_t j=0;j<n;++j) linear_weights.values[p*n+j]=weights[j*k+p];
        linear_requant.multiplier.resize(n);linear_requant.bias_q16.resize(n);
        linear_requant.minimum=minimum;linear_requant.maximum=maximum;
        for(size_t j=0;j<n;++j) {
            if(!std::isfinite(scales[j]) || scales[j]<=0 || !std::isfinite(bias[j]))
                throw std::invalid_argument("Invalid per-channel native quantization");
            linear_requant.multiplier[j]=double(si)*scales[j]/so;
            const double b=double(bias[j])/so*65536.0;
            if(std::abs(b)>double(std::numeric_limits<int64_t>::max()/4)) throw std::overflow_error("Native bias overflow");
            linear_requant.bias_q16[j]=std::llround(b);
        }
        const auto plan_start=fpga::ProfileClock::now();
        const char* mode=std::getenv("MNN_MATMUL_SOC_TILE_POLICY");
        linear_plan=fpga::plan_matmul_soc(m,n,k,mode && std::string(mode)=="fixed64"?
            fpga::MatmulSocTilePolicy::Fixed64:fpga::MatmulSocTilePolicy::Capacity);
        weight_plan_ms=std::chrono::duration<double,std::milli>(fpga::ProfileClock::now()-plan_start).count();
        weight_setup_ms=std::chrono::duration<double,std::milli>(fpga::ProfileClock::now()-start).count();
        linear_input_scale=si;linear_output_scale=so;linear_int8=true;linear_ready=true;
        if(const char* prefix=std::getenv("MNN_INT8_TRACE")) {
            const std::string base=prefix;
            std::ofstream w(base+".weights.bin",std::ios::binary),g(base+".multipliers.f64",std::ios::binary),b(base+".biasq16.i64",std::ios::binary);
            w.write(reinterpret_cast<const char*>(linear_weights.values.data()),linear_weights.values.size());
            g.write(reinterpret_cast<const char*>(linear_requant.multiplier.data()),n*sizeof(double));
            b.write(reinterpret_cast<const char*>(linear_requant.bias_q16.data()),n*sizeof(int64_t));
            if(!w || !g || !b) throw std::runtime_error("Cannot save native INT8 diagnostic parameters");
        }
        return true;
    } catch(const std::exception& e) {std::fprintf(stderr,"MNN native INT8 prepare: %s\n",e.what());return false;}
}
bool mnn_fpga_linear_int8_active() {return selected.load() && ((linear_ready && linear_int8) || natives().active>=0);}
unsigned long mnn_fpga_completed_int8_operations() {return completed_int8.load();}
void mnn_fpga_register_native(const std::string& name,const int8_t* weights,const float* scales,
    const float* bias,size_t k,size_t n,float si,float so) {
    std::lock_guard<std::mutex> lock(device_mutex);
    auto& state=natives();
    if(name.empty() || state.names.count(name) || !weights || !scales || !bias ||
       !std::isfinite(si) || !std::isfinite(so) || si<=0 || so<=0 || k%4 || n%4)
        throw std::invalid_argument("Invalid/duplicate native operation");
    const char* d=std::getenv("MNN_MATMUL_SOC_DMA"),*p=std::getenv("MNN_MATMUL_SOC_DEVMEM");
    if(!d || std::string(d)!="1" || !p || std::string(p)!="1")throw std::runtime_error("Native NI82 requires explicit DMA/devmem");
    if(!soc_device)soc_device.reset(new fpga::MatmulSocUio("/dev/mem",true,true));
    fpga::MatmulSocInt8Requantization q;q.multiplier.resize(n);q.bias_q16.resize(n);
    for(size_t j=0;j<n;++j) {
        if(!std::isfinite(bias[j]) || !std::isfinite(scales[j]) || scales[j]<=0)
            throw std::invalid_argument("Invalid native channel parameters");
        q.multiplier[j]=double(si)*scales[j]/so;
        const auto b=double(bias[j])/so*65536;
        if(!std::isfinite(b) || std::abs(b)>2147483647.0)throw std::invalid_argument("Native Q16 bias overflow");
        q.bias_q16[j]=std::llround(b);
    }
    NativeContext context;context.name=name;context.input_scale=si;context.output_scale=so;
    context.weights=fpga::prepare_native_int8(*soc_device,weights,k,n,q);
    state.names[name]=int(state.contexts.size());state.contexts.push_back(std::move(context));
}
void mnn_fpga_register_native_pair(const std::string& first,const std::string& second,
    const std::array<int8_t,256>& lut,const std::vector<std::string>& skipped) {
    auto& state=natives();const auto f=state.names.at(first),s=state.names.at(second);
    if(f>=s || state.contexts[f].weights.n!=state.contexts[s].weights.k || skipped.empty())
        throw std::invalid_argument("Invalid native resident pair");
    state.contexts[f].second=s;state.contexts[f].lut=lut;
    state.contexts[f].skipped.insert(skipped.begin(),skipped.end());
}
bool mnn_fpga_select_native(const std::string& name) {
    auto& state=natives();const auto it=state.names.find(name);
    state.active=it==state.names.end()?-1:it->second;selected.store(state.active>=0);return state.active>=0;
}
bool mnn_fpga_native_skip(const std::string& name) {
    auto& state=natives();
    if(state.pending<0)return false;
    for(const auto& c:state.contexts)if(c.second==state.pending)
        for(const auto& skip:c.skipped)if(name==skip || name.rfind(skip+"_raster_",0)==0){++state.skipped_ops;return true;}
    return false;
}
size_t mnn_fpga_native_count(){return natives().contexts.size();}
const char* mnn_fpga_native_json() {
    auto& state=natives();static std::string result;std::ostringstream out;
    out<<"{\"abi\":\"NI82\",\"operations\":"<<state.contexts.size()<<",\"local_feedback_bytes\":"<<state.local_bytes
       <<",\"hidden_roundtrip_bytes_avoided\":"<<state.hidden_bytes_avoided
       <<",\"fused_pairs\":"<<state.fused_pairs<<",\"skipped_runtime_operations\":"<<state.skipped_ops
       <<",\"pending\":"<<state.pending<<",\"layers\":[";
    for(size_t i=0;i<state.contexts.size();++i) {
        const auto& c=state.contexts[i];const auto& w=c.weights;if(i)out<<',';
        out<<"{\"name\":\""<<c.name<<"\",\"K\":"<<w.k<<",\"N\":"<<w.n
           <<",\"tile_m\":"<<w.tile_m<<",\"tile_n\":"<<w.tile_n<<",\"tile_k\":"<<w.kp
           <<",\"operand_b_bytes\":"<<w.operand_b_bytes<<",\"fused_second\":"<<c.second<<'}';
    }
    out<<"]}";result=out.str();return result.c_str();
}
namespace {
void trace_tensor(const char* prefix,const char* suffix,const int8_t* packed,size_t m,size_t channels) {
    std::vector<int8_t> dense(m*channels);
    for(size_t p=0;p<channels;p+=4) for(size_t i=0;i<m;++i)
        std::copy_n(packed+(p/4*m+i)*4,4,dense.data()+i*channels+p);
    std::ofstream file(std::string(prefix)+suffix,std::ios::binary);
    file.write(reinterpret_cast<const char*>(dense.data()),dense.size());
    if(!file) std::fprintf(stderr,"INT8 diagnostic dump failed\n");
}
}
MnnFpgaInt8Trace::MnnFpgaInt8Trace(const int8_t* input,const int8_t* out,size_t rows,size_t k,size_t cols,bool valid) {
    if(!valid || !selected.load() || rows!=784 || k!=128 || cols!=512) return;
    prefix=std::getenv("MNN_INT8_TRACE");if(!prefix || !*prefix) {prefix=nullptr;return;}
    output=out;m=rows;n=cols;trace_tensor(prefix,".input.bin",input,m,k);
}
MnnFpgaInt8Trace::~MnnFpgaInt8Trace() {if(prefix) trace_tensor(prefix,".output.bin",output,m,n);}
int mnn_fpga_try_linear_int8(const int8_t* input,int8_t* output,size_t m,size_t k,size_t n,
    float si,float so,float zi,float zo) {
    std::lock_guard<std::mutex> lock(device_mutex);
    auto* timing=profiling?&profile:nullptr;
    try {
        auto& state=natives();
        if(state.active>=0) {
            const auto start=fpga::profile_now(timing);auto& context=state.contexts.at(size_t(state.active));
            if(!selected.load() || soc_failed || !soc_device || !input || !output || !m ||
               k!=context.weights.k || n!=context.weights.n || si!=context.input_scale || so!=context.output_scale || zi || zo)
                throw std::runtime_error("Native multi-operation tensor contract mismatch");
            soc_device->profile=timing;fpga::MatmulSocStats stats;
            if(state.pending>=0) {
                if(state.active!=state.pending || m!=state.pending_rows || state.output.size()!=m*n)
                    throw std::runtime_error("Resident FC2 order/shape mismatch");
                const auto copy_start=fpga::profile_now(timing);
                std::copy(state.output.begin(),state.output.end(),output);state.pending=-1;
                fpga::profile_add(timing,fpga::MatmulSocProfile::output_layout,copy_start);
            } else if(context.second>=0) {
                const auto& second=state.contexts.at(size_t(context.second));
                const auto allocation_start=fpga::profile_now(timing);state.output.resize(m*second.weights.n);
                fpga::profile_add(timing,fpga::MatmulSocProfile::layout_allocate,allocation_start);
                const auto p=fpga::native_int8_pair(*soc_device,context.weights,second.weights,context.lut,input,state.output.data(),m);
                stats=p;state.local_bytes+=p.local_bytes;state.hidden_bytes_avoided+=p.hidden_bytes_avoided;
                ++state.fused_pairs;
                state.pending=context.second;state.pending_rows=m;
            } else stats=fpga::native_int8_linear(*soc_device,context.weights,input,output,m);
            if(timing){timing->jobs+=stats.submissions;timing->input_bytes+=stats.operand_bytes;timing->output_bytes+=stats.result_bytes;}
            fpga::profile_add(timing,fpga::MatmulSocProfile::adapter_total,start);
            completed.fetch_add(stats.submissions);completed_int8.fetch_add(1);return 1;
        }
        if(!selected.load() || !linear_ready || !linear_int8 || soc_failed || !input || !output ||
            m!=linear_plan.m || k!=linear_plan.k || n!=linear_plan.n || si!=linear_input_scale || so!=linear_output_scale || zi!=0 || zo!=0)
            throw std::runtime_error("Native fc1 tensor/scale contract mismatch");
        const auto allocated_start=fpga::profile_now(timing);
        fpga::MatmulSocOperand a;a.rows=m;a.cols=k;a.scale=1;a.values.resize(m*k);
        std::vector<int8_t> dense_output(m*n);
        fpga::profile_add(timing,fpga::MatmulSocProfile::layout_allocate,allocated_start);
        const auto unpack_start=fpga::profile_now(timing);
        for(size_t p=0;p<k;p+=4) for(size_t i=0;i<m;++i)
            std::copy_n(input+(p/4*m+i)*4,4,a.values.data()+i*k+p);
        fpga::profile_add(timing,fpga::MatmulSocProfile::input_layout,unpack_start);
        const auto bridge_start=fpga::profile_now(timing);
        const char* devmem=std::getenv("MNN_MATMUL_SOC_DEVMEM");const char* dma=std::getenv("MNN_MATMUL_SOC_DMA");
        if(!devmem || std::string(devmem)!="1" || !dma || std::string(dma)!="1")
            throw std::runtime_error("Native INT8 offload requires explicit devmem/DMA");
        if(!soc_device) {const auto start=fpga::profile_now(timing);soc_device.reset(new fpga::MatmulSocUio("/dev/mem",true,true));
            fpga::profile_add(timing,fpga::MatmulSocProfile::device_init,start);}
        soc_device->profile=timing;const auto adapter_start=fpga::profile_now(timing);
        const auto stats=fpga::matmul_soc_gemm_int8(*soc_device,a,linear_weights,dense_output.data(),linear_requant,linear_plan);
        if(timing) {timing->jobs+=stats.submissions;timing->input_bytes+=stats.operand_bytes;timing->output_bytes+=stats.result_bytes;}
        fpga::profile_add(timing,fpga::MatmulSocProfile::adapter_total,adapter_start);
        std::fprintf(stderr,"MNN native INT8 FPGA: %zux%zu x %zux%zu; %zu jobs; no FP32 activation/output adapter\n",m,k,k,n,stats.submissions);
        fpga::profile_add(timing,fpga::MatmulSocProfile::bridge_total,bridge_start);
        const auto pack_start=fpga::profile_now(timing);
        for(size_t j=0;j<n;j+=4) for(size_t i=0;i<m;++i)
            std::copy_n(dense_output.data()+i*n+j,4,output+(j/4*m+i)*4);
        fpga::profile_add(timing,fpga::MatmulSocProfile::output_layout,pack_start);
        completed.fetch_add(stats.submissions);completed_int8.fetch_add(1);return 1;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"MNN native INT8 FPGA failed: %s\n",e.what());soc_failed=true;soc_device.reset();return -1;
    }
}
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
    auto* timing=profiling ? &profile:nullptr;
    const auto bridge_start=fpga::profile_now(timing);
    try {
        const char* mode=std::getenv("MNN_MATMUL_SOC_DMA");
        const bool dma=mode && std::string(mode)=="1";
        if (!soc_device) {
            const auto init_start=fpga::profile_now(timing);
            soc_device.reset(new fpga::MatmulSocUio(path,devmem,dma));
            fpga::profile_add(timing,fpga::MatmulSocProfile::device_init,init_start);
        }
        soc_device->profile=timing;
        auto stats=fpga::matmul_soc_gemm(*soc_device,a,b,c,m,n,k,ta,tb,tc,bias);
        completed.fetch_add(stats.submissions);
        std::fprintf(stderr,"MNN matmul-soc: %zux%zu x %zux%zu, %zu FPGA submissions (prepared INT8, %s, quantized=%zu, in=%zu B, out=%zu B)\n",
            m,k,k,n,stats.submissions,soc_device->uses_dma()?"DMA":"PIO",stats.quantized_values,stats.operand_bytes,stats.result_bytes);
        fpga::profile_add(timing,fpga::MatmulSocProfile::bridge_total,bridge_start);
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
