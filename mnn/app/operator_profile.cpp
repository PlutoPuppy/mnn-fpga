#include "operator_profile.hpp"
#include <MNN/Tensor.hpp>
#include "MNN_generated.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>

namespace operator_profile {
namespace {
void string(std::ostream& out,const std::string& s) {
    out<<'"';
    for(unsigned char c:s) {
        if(c=='"' || c=='\\') out<<'\\'<<char(c);
        else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<char(c);
    }
    out<<'"';
}
template<class Values> void array(std::ostream& out,const Values& values) {
    out<<'[';bool first=true;
    for(const auto& value:values) {if(!first) out<<',';first=false;out<<value;}
    out<<']';
}
std::vector<TensorDescription> describe(const std::vector<MNN::Tensor*>& tensors) {
    std::vector<TensorDescription> result;result.reserve(tensors.size());
    for(const auto* t:tensors) {
        TensorDescription d;
        for(int i=0;i<t->dimensions();++i) {d.shape.push_back(t->length(i));d.strides.push_back(t->stride(i));}
        const auto type=t->getType();d.code=type.code;d.bits=type.bits;d.lanes=type.lanes;d.layout=int(t->getDimensionType());
        result.push_back(std::move(d));
    }
    return result;
}
void tensors(std::ostream& out,const std::vector<TensorDescription>& values) {
    out<<'[';bool first=true;
    for(const auto& t:values) {
        if(!first) out<<',';first=false;
        out<<"{\"shape\":";array(out,t.shape);out<<",\"strides\":";array(out,t.strides);
        out<<",\"layout\":"<<t.layout<<",\"dtype_code\":"<<t.code<<",\"bits\":"<<t.bits<<",\"lanes\":"<<t.lanes<<'}';
    }
    out<<']';
}
void model_metadata(std::ostream& out,const std::string& path) {
    std::ifstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("Cannot read model metadata");
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(bytes.data()),bytes.size());
    if(!MNN::VerifyNetBuffer(verifier)) throw std::runtime_error("Invalid MNN metadata buffer");
    const auto* net=MNN::GetNet(bytes.data());
    out<<'[';bool first=true;
    auto scan=[&](const auto* nodes) {
        if(!nodes) return;
        for(const auto* op:*nodes) {
            if(!first) out<<',';first=false;
            out<<"{\"name\":";string(out,op->name() ? op->name()->str():"");
            out<<",\"type\":";string(out,MNN::EnumNameOpType(op->type()));
            out<<",\"inputs\":";if(op->inputIndexes()) array(out,*op->inputIndexes());else out<<"[]";
            out<<",\"outputs\":";if(op->outputIndexes()) array(out,*op->outputIndexes());else out<<"[]";
            if(const auto* p=op->main_as_Convolution2D()) {
                out<<",\"weight_elements\":"<<(p->weight() ? p->weight()->size():0)
                   <<",\"bias_elements\":"<<(p->bias() ? p->bias()->size():0)
                   <<",\"has_quantized_weights\":"<<((p->quanParameter() || p->symmetricQuan()) ? "true":"false");
                if(const auto* c=p->common()) {
                    out<<",\"conv\":{\"input_channels\":"<<c->inputCount()<<",\"output_channels\":"<<c->outputCount()
                       <<",\"kernel\":["<<c->kernelY()<<','<<c->kernelX()<<"],\"stride\":["<<c->strideY()<<','<<c->strideX()
                       <<"],\"dilation\":["<<c->dilateY()<<','<<c->dilateX()<<"],\"group\":"<<c->group()
                       <<",\"relu\":"<<(c->relu()?"true":"false")<<",\"relu6\":"<<(c->relu6()?"true":"false")<<'}';
                }
            }
            if(const auto* p=op->main_as_MatMul()) out<<",\"transpose_a\":"<<(p->transposeA()?"true":"false")<<",\"transpose_b\":"<<(p->transposeB()?"true":"false");
            if(const auto* p=op->main_as_Blob()) {
                out<<",\"constant_dtype\":";string(out,MNN::EnumNameDataType(p->dataType()));
                out<<",\"constant_shape\":";if(p->dims()) array(out,*p->dims());else out<<"[]";
                out<<",\"float_constant_elements\":"<<(p->float32s()?p->float32s()->size():0);
            }
            out<<'}';
        }
    };
    scan(net->oplists());
    if(net->subgraphs()) for(const auto* graph:*net->subgraphs()) scan(graph->nodes());
    out<<']';
}
}
void Profiler::reset() {records_.clear();records_.reserve(4096);active_=false;}
void Profiler::before(const std::vector<MNN::Tensor*>& values,const MNN::OperatorInfo* info) {
    if(active_) throw std::runtime_error("Overlapping operator callback intervals");
    Record r;r.name=info->name();r.type=info->type();r.estimated_mflops=info->flops();r.inputs=describe(values);
    records_.push_back(std::move(r));active_=true;start_=std::chrono::steady_clock::now();
}
void Profiler::after(const std::vector<MNN::Tensor*>& values,const MNN::OperatorInfo* info) {
    const auto end=std::chrono::steady_clock::now();
    if(!active_ || records_.back().name!=info->name()) throw std::runtime_error("Mismatched operator callbacks");
    auto& r=records_.back();r.ms=std::chrono::duration<double,std::milli>(end-start_).count();
    active_=false;r.outputs=describe(values);
}
void Profiler::save(const std::string& path,const std::string& model,double inference_ms,int runs) const {
    if(active_ || records_.empty()) throw std::runtime_error("Incomplete operator profile");
    const std::filesystem::path destination(path);
    if(!destination.parent_path().empty()) std::filesystem::create_directories(destination.parent_path());
    std::ofstream out(path);if(!out) throw std::runtime_error("Cannot create operator profile");
    double sum=0;for(const auto& r:records_) sum+=r.ms;
    out<<std::setprecision(12)<<"{\"schema_version\":1,\"runs\":"<<runs<<",\"inference_ms\":"<<inference_ms
       <<",\"sum_operator_ms\":"<<sum<<",\"outside_operator_ms\":"<<inference_ms-sum
       <<",\"records\":[";
    for(size_t i=0;i<records_.size();++i) {
        if(i) out<<',';const auto& r=records_[i];
        out<<"{\"index\":"<<i<<",\"name\":";string(out,r.name);out<<",\"type\":";string(out,r.type);
        out<<",\"ms\":"<<r.ms<<",\"mnn_estimated_mflops\":"<<r.estimated_mflops<<",\"inputs\":";tensors(out,r.inputs);
        out<<",\"outputs\":";tensors(out,r.outputs);out<<'}';
    }
    out<<"],\"model_nodes\":";model_metadata(out,model);out<<"}\n";
    out.close();if(!out) throw std::runtime_error("Cannot finish operator profile");
}
}
