#include "gelu_lookup.hpp"
#include "gelu_lookup_kernel.hpp"
#include "fpga_bridge.hpp"
#include "MNN_generated.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
struct GeluLookup::State {
    struct Row {uint64_t calls=0,values=0,over=0;double ms=0,maximum=0,squared=0;MnnCpuPoolInfo pool;int workers=1;};
    std::map<std::string,Row> layers;std::vector<float> table,scratch,saved;
    const MNN::Tensor* input=nullptr;std::string pending;int threads=1;bool verify=false;Mode mode=Mode::Lookup;
};
GeluLookup::GeluLookup(const MNN::Interpreter& net,int threads,bool verify,Mode mode):state_(new State) {
    auto& s=*state_;s.threads=threads;s.verify=verify;s.mode=mode;
    auto buffer=net.getModelBuffer();for(const auto* op:*MNN::GetNet(buffer.first)->oplists())
        if(op->name() && op->type()==MNN::OpType_UnaryOp && op->main_as_UnaryOp()->opType()==MNN::UnaryOpOperation_GELU)s.layers.emplace(op->name()->str(),State::Row{});
    if(s.layers.empty())throw std::runtime_error("No supported GELU operators");
    if(s.mode!=Mode::Lookup)return;
    std::vector<float> inputs(tinyvit_gelu::intervals+1),outputs(inputs.size());
    for(size_t i=0;i<inputs.size();++i)inputs[i]=float(i)/256.0f-8.0f;
    mnn_cpu_gelu(outputs.data(),inputs.data(),inputs.size());s.table.resize(tinyvit_gelu::intervals*2);
    tinyvit_gelu::coefficients(s.table.data(),outputs.data());
}
GeluLookup::~GeluLookup()=default;
bool GeluLookup::before(const std::string& name,const std::vector<MNN::Tensor*>& input) {
    auto& s=*state_;if(!s.layers.count(name))return true;
    if(s.input || input.size()!=1 || input[0]->getType()!=halide_type_of<float>())throw std::runtime_error("Invalid FP32 GELU input");
    s.pending=name;s.input=input[0];
    if(s.verify)s.saved.assign(input[0]->host<float>(),input[0]->host<float>()+input[0]->elementSize());
    return s.verify;
}
void GeluLookup::after(const std::string& name,const std::vector<MNN::Tensor*>& output) {
    auto& s=*state_;if(!s.input || name!=s.pending)return;
    const auto start=std::chrono::steady_clock::now();
    if(output.size()!=1 || output[0]->getType()!=halide_type_of<float>() || output[0]->elementSize()!=s.input->elementSize())throw std::runtime_error("Invalid GELU output");
    const size_t count=size_t(s.input->elementSize());auto& record=s.layers.at(name);
    float* destination=output[0]->host<float>();
    if(s.verify){s.scratch.resize(count);destination=s.scratch.data();}
    const float* source=s.input->host<float>();const int tasks=mnn_cpu_parallel_threads(s.input,s.threads);
    record.pool=mnn_cpu_pool_info(s.input);record.workers=tasks;
    // In verification, the original op may have reused its input allocation.
    // A saved input copy is prepared by before() for this diagnostic mode.
    if(s.verify)source=s.saved.data();
    mnn_cpu_parallel(s.input,tasks,[&](int id) {
        const size_t unit=(count+size_t(tasks)*4-1)/(size_t(tasks)*4)*4;
        const size_t first=std::min(count,size_t(id)*unit),last=std::min(count,first+unit);
        switch(s.mode) {
            case Mode::Lookup: tinyvit_gelu::apply(destination+first,source+first,last-first,s.table.data());break;
            case Mode::Reciprocal: tinyvit_gelu::reciprocal(destination+first,source+first,last-first);break;
            case Mode::Polynomial: tinyvit_gelu::polynomial(destination+first,source+first,last-first);break;
        }
    });
    if(s.verify)for(size_t i=0;i<count;++i) {
        const float reference=output[0]->host<float>()[i];const double error=std::abs(double(destination[i])-reference);
        record.over+=!std::isfinite(error) || error>5e-5+5e-5*std::abs(reference);record.maximum=std::max(record.maximum,error);record.squared+=error*error;
    }
    ++record.calls;record.values+=count;record.ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();s.input=nullptr;s.pending.clear();
}
void GeluLookup::reset(){auto& s=*state_;if(s.input)throw std::runtime_error("Pending GELU at reset");for(auto& p:s.layers)p.second={};}
bool GeluLookup::valid(unsigned runs) const {
    const auto& s=*state_;uint64_t calls=0;
    for(const auto& p:s.layers){calls+=p.second.calls;if(p.second.over || (p.second.calls && p.second.calls!=runs))return false;}
    return !s.input && runs>0 && calls>0;
}
std::string GeluLookup::json() const {
    const auto& s=*state_;std::ostringstream out;out.precision(12);
    const auto mode=s.mode==Mode::Polynomial ? "polynomial" : s.mode==Mode::Reciprocal ? "reciprocal" : "lookup";
    out<<"{\"mode\":\""<<mode<<"\",\"verify\":"<<(s.verify?"true":"false")<<",\"layers\":[";bool first=true;
    for(const auto& p:s.layers){const auto& v=p.second;if(!first)out<<',';first=false;
        out<<"{\"name\":\""<<p.first<<"\",\"calls\":"<<v.calls<<",\"values\":"<<v.values<<",\"ms\":"<<v.ms<<",\"owner_threads\":"<<v.pool.configured_threads<<",\"pool_capacity\":"<<v.pool.available_workers<<",\"owner_task_index\":"<<v.pool.task_index<<",\"kernel_workers\":"<<v.workers<<",\"over_tolerance\":"<<v.over<<",\"maximum_error\":"<<v.maximum<<",\"rmse\":"<<std::sqrt(v.squared/std::max(1.0,double(v.values)))<<'}';}
    out<<"]}";return out.str();
}
