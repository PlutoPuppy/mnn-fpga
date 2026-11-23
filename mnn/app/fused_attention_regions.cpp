#include "fused_attention_regions.hpp"
#include "fused_attention_kernel.hpp"
#include "fpga_bridge.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

struct FusedAttentionRegions::State {
    struct Region {
        std::string source,target;size_t heads=0,tokens=0,dim=0,rows=0;
        float scale=0;std::vector<float> bias,result,scratch;
        std::set<std::string> skip;
        uint64_t calls=0,skipped=0,checked=0,over=0;
        MnnCpuPoolInfo pool;int workers=1;size_t query_slices=1;
        double compute_ms=0,copy_ms=0,max_error=0,square_error=0,square_reference=0;
    };
    std::vector<Region> regions;std::map<std::string,size_t> sources;
    int pending=-1,threads=1;bool verify=false;
};
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
}
FusedAttentionRegions::FusedAttentionRegions(const MNN::Interpreter& model,int threads,bool verify):state_(new State) {
    auto& state=*state_;state.threads=threads;state.verify=verify;
    auto buffer=model.getModelBuffer();const auto* graph=MNN::GetNet(buffer.first);
    std::map<std::string,const MNN::Op*> names;std::map<std::string,size_t> positions;
    std::map<int,const MNN::Op*> producers;
    for(size_t i=0;i<graph->oplists()->size();++i) {
        const auto* op=graph->oplists()->Get(unsigned(i));if(!op->name())continue;
        names[op->name()->str()]=op;positions[op->name()->str()]=i;
        if(op->outputIndexes())for(auto index:*op->outputIndexes())producers[index]=op;
    }
    for(const auto* op:*graph->oplists()) {
        if(!op->name() || op->type()!=MNN::OpType_Convolution)continue;const auto name=op->name()->str();
        const auto marker=name.find("/attn/qkv/");
        if(marker==std::string::npos || name.find("__matmul_converted")==std::string::npos)continue;
        const auto prefix=name.substr(0,marker)+"/attn/";
        const auto target=prefix+"proj/Add_output_0__matmul_converted";
        if(!names.count(target) || !names.count(prefix+"Mul_output_0") || !names.count(prefix+"Add_output_0"))
            throw std::runtime_error("Unsupported TinyViT attention constants: "+name);
        const auto* mul=names.at(prefix+"Mul_output_0");const auto* add=names.at(prefix+"Add_output_0");
        if(!mul->inputIndexes() || mul->inputIndexes()->size()!=2 || !add->inputIndexes() || add->inputIndexes()->size()!=2)
            throw std::runtime_error("Invalid attention scale/bias connectivity");
        const auto* bias=producers.at(add->inputIndexes()->Get(1))->main_as_Blob();
        const auto* scale=producers.at(mul->inputIndexes()->Get(1))->main_as_Blob();
        const auto* conv=op->main_as_Convolution2D();
        if(!bias || !bias->dims() || bias->dims()->size()!=3 || !bias->float32s() || !scale ||
           !scale->float32s() || scale->float32s()->size()!=1 || !conv || !conv->common())
            throw std::runtime_error("Invalid TinyViT attention parameters");
        State::Region r;r.source=name;r.target=names.count(target+"/quantize")?target+"/quantize":target;
        r.heads=size_t(bias->dims()->Get(0));r.tokens=size_t(bias->dims()->Get(1));
        if(!r.heads || !r.tokens || bias->dims()->Get(2)!=int(r.tokens) ||
           size_t(conv->common()->outputCount())% (r.heads*3))throw std::runtime_error("Unsupported QKV dimensions");
        r.dim=size_t(conv->common()->outputCount())/r.heads/3;
        if(r.dim%4 || !r.dim || r.dim>128 || r.tokens>256 || r.heads>32 ||
           bias->float32s()->size()!=r.heads*r.tokens*r.tokens)throw std::runtime_error("Unsupported attention dimensions");
        r.scale=scale->float32s()->Get(0);if(!std::isfinite(r.scale) || r.scale<=0)throw std::runtime_error("Invalid attention scale");
        r.bias.assign(bias->float32s()->begin(),bias->float32s()->end());
        for(size_t i=positions.at(name)+1;i<positions.at(r.target);++i) {
            const auto* skipped=graph->oplists()->Get(unsigned(i));if(skipped->name())r.skip.insert(skipped->name()->str());
        }
        r.skip.insert(r.target);r.skip.insert(target);
        state.sources.emplace(name,state.regions.size());state.regions.push_back(std::move(r));
    }
    if(state.regions.size()!=10)throw std::runtime_error("Expected ten TinyViT attention regions");
}
FusedAttentionRegions::~FusedAttentionRegions()=default;
bool FusedAttentionRegions::before(const std::string& name,const std::vector<MNN::Tensor*>& inputs) {
    auto& state=*state_;if(state.pending<0)return true;
    auto& r=state.regions.at(size_t(state.pending));
    if(name==r.target) {
        const auto start=Clock::now();
        if(inputs.size()!=1 || !inputs[0]->host<float>() || inputs[0]->getType().bits!=32 ||
           size_t(inputs[0]->batch())!=r.rows || size_t(inputs[0]->channel())!=r.heads*r.dim ||
           inputs[0]->height()!=1 || inputs[0]->width()!=1)throw std::runtime_error("Attention target must be FP32 MxCx1x1");
        const auto* desc=MNN::TensorUtils::getDescribe(inputs[0]);
        const bool packed=desc->dimensionFormat==MNN::MNN_DATA_FORMAT_NC4HW4;
        if(!packed && desc->dimensionFormat!=MNN::MNN_DATA_FORMAT_NCHW)throw std::runtime_error("Unsupported attention target layout");
        const size_t channels=r.heads*r.dim;auto* target=inputs[0]->host<float>();
        if(state.verify) {
            for(size_t row=0;row<r.rows;++row)for(size_t c=0;c<channels;++c) {
                const size_t nc4=(c/4*r.rows+row)*4+c%4;
                const float reference=target[packed?nc4:row*channels+c],actual=r.result[nc4];
                const double error=std::abs(double(actual)-reference);
                ++r.checked;r.over+=!std::isfinite(error) || error>1e-4+1e-4*std::abs(reference);
                r.max_error=std::max(r.max_error,error);r.square_error+=error*error;r.square_reference+=double(reference)*reference;
            }
        } else if(packed)std::memcpy(target,r.result.data(),r.result.size()*sizeof(float));
        else for(size_t row=0;row<r.rows;++row)for(size_t c=0;c<channels;++c)
            target[row*channels+c]=r.result[(c/4*r.rows+row)*4+c%4];
        r.copy_ms+=elapsed(start);state.pending=-1;return true;
    }
    if(state.verify)return true;
    const auto raster=name.find("_raster_");
    if(r.skip.count(name) || (raster!=std::string::npos && r.skip.count(name.substr(0,raster)))) {++r.skipped;return false;}
    return true;
}
void FusedAttentionRegions::after(const std::string& name,const std::vector<MNN::Tensor*>& outputs) {
    auto& state=*state_;const auto it=state.sources.find(name);if(it==state.sources.end())return;
    if(state.pending>=0)throw std::runtime_error("Attention region did not reach its target");
    auto& r=state.regions[it->second];const auto start=Clock::now();
    if(outputs.size()!=1)throw std::runtime_error("Invalid QKV output count");
    const auto* t=outputs[0];const auto* desc=MNN::TensorUtils::getDescribe(t);
    if(!t->host<float>() || desc->dimensionFormat!=MNN::MNN_DATA_FORMAT_NC4HW4 ||
       t->height()!=1 || t->width()!=1 || size_t(t->channel())!=r.heads*r.dim*3 || t->batch()<=0 || size_t(t->batch())%r.tokens)
        throw std::runtime_error("Unsupported QKV storage/layout");
    const bool quantized=t->getType().bits==8;
    if(!quantized && t->getType()!=halide_type_of<float>())throw std::runtime_error("Unsupported QKV type");
    float scale=1;
    if(quantized) {
        if(!desc->quantAttr || desc->quantAttr->zero!=0 || desc->quantAttr->scale<=0)throw std::runtime_error("Invalid QKV quantization");
        scale=desc->quantAttr->scale;
    }
    r.rows=size_t(t->batch());const int tasks=mnn_cpu_parallel_threads(t,state.threads);
    const size_t block=(r.tokens+8)*(r.dim*3+4);r.scratch.resize(block*size_t(tasks));r.result.resize(r.rows*r.heads*r.dim);
    const size_t slices=tasks==2 && r.rows==r.tokens && r.heads%2 && r.tokens>=8?2:1;
    const size_t work=r.rows/r.tokens*r.heads*slices;
    r.pool=mnn_cpu_pool_info(t);r.workers=tasks;r.query_slices=slices;
    mnn_cpu_parallel(t,tasks,[&](int id) {
        float* q=r.scratch.data()+size_t(id)*block;float* k=q+(r.tokens+8)*r.dim;
        float* v=k+(r.tokens+8)*r.dim;float* scores=v+(r.tokens+8)*r.dim;
        for(size_t job=size_t(id);job<work;job+=size_t(tasks)) {
            const size_t item=job/slices,window=item/r.heads,h=item%r.heads;
            const size_t middle=std::min(r.tokens,((r.tokens+3)/4+1)/2*4);
            const size_t first=slices==1 || job%2==0?0:middle;
            const size_t last=slices==1 || job%2?r.tokens:middle;
            tinyvit_attention::head(t->host<void>(),quantized,scale,r.bias.data()+h*r.tokens*r.tokens,r.scale,
                r.rows,r.tokens,r.heads,r.dim,window,h,q,k,v,scores,r.result.data(),mnn_cpu_softmax,first,last);
        }
    });
    ++r.calls;r.compute_ms+=elapsed(start);state.pending=int(it->second);
}
void FusedAttentionRegions::reset() {
    auto& s=*state_;if(s.pending>=0)throw std::runtime_error("Pending attention at reset");
    for(auto& r:s.regions){r.calls=r.skipped=r.checked=r.over=0;r.compute_ms=r.copy_ms=r.max_error=r.square_error=r.square_reference=0;}
}
bool FusedAttentionRegions::valid(unsigned runs) const {
    const auto& s=*state_;return s.pending<0 && std::all_of(s.regions.begin(),s.regions.end(),[&](const State::Region& r){return r.calls==runs && (!s.verify || (r.checked>0 && !r.over));});
}
std::string FusedAttentionRegions::json() const {
    const auto& s=*state_;std::ostringstream o;o.precision(12);o<<"{\"verify\":"<<(s.verify?"true":"false")<<",\"pending\":"<<s.pending<<",\"regions\":[";
    for(size_t i=0;i<s.regions.size();++i){const auto& r=s.regions[i];if(i)o<<',';
        o<<"{\"name\":\""<<r.source<<"\",\"rows\":"<<r.rows<<",\"tokens\":"<<r.tokens<<",\"heads\":"<<r.heads
         <<",\"dim\":"<<r.dim<<",\"calls\":"<<r.calls<<",\"skipped\":"<<r.skipped<<",\"compute_ms\":"<<r.compute_ms
         <<",\"copy_or_verify_ms\":"<<r.copy_ms<<",\"checked\":"<<r.checked<<",\"over_tolerance\":"<<r.over
         <<",\"owner_threads\":"<<r.pool.configured_threads<<",\"pool_capacity\":"<<r.pool.available_workers
         <<",\"owner_task_index\":"<<r.pool.task_index<<",\"kernel_workers\":"<<r.workers<<",\"query_slices\":"<<r.query_slices
         <<",\"maximum_error\":"<<r.max_error<<",\"relative_l2\":"<<std::sqrt(r.square_error/std::max(r.square_reference,1e-30))<<'}';}
    o<<"]}";return o.str();
}
