#include "native_projection.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
MNN::Tensor::DimensionType storage_layout(const MNN::Tensor* tensor) {
    // getDimensionType() maps both NCHW and NC4HW4 to CAFFE in this MNN.
    switch(MNN::TensorUtils::getDescribe(tensor)->dimensionFormat) {
        case MNN::MNN_DATA_FORMAT_NCHW: return MNN::Tensor::CAFFE;
        case MNN::MNN_DATA_FORMAT_NC4HW4: return MNN::Tensor::CAFFE_C4;
        default: return MNN::Tensor::TENSORFLOW;
    }
}
size_t rows_of(const MNN::Tensor* tensor,size_t channels) {
    const auto format=storage_layout(tensor);
    if(tensor->dimensions()!=4 || (format!=MNN::Tensor::CAFFE_C4 && format!=MNN::Tensor::CAFFE) ||
       tensor->getType()!=halide_type_of<float>() || !tensor->host<float>() ||
       tensor->channel()!=int(channels) || tensor->batch()<1 || tensor->height()<1 || tensor->width()<1 ||
       (tensor->batch()>1 && (tensor->height()!=1 || tensor->width()!=1)))
        throw std::runtime_error("Native projection requires a contiguous FP32 NCHW or NC4 tensor");
    return size_t(tensor->batch())*tensor->height()*tensor->width();
}
}

NativeProjections::NativeProjections(const MNN::Interpreter& net,const std::string& ranges,int threads,bool verify)
    : threads_(threads),verify_(verify) {
    std::ifstream file(ranges);
    if(!file)throw std::runtime_error("Cannot open projection calibration ranges");
    const auto data=net.getModelBuffer();
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(data.first),data.second);
    if(!MNN::VerifyNetBuffer(verifier))throw std::runtime_error("Invalid projection model buffer");
    std::map<std::string,const MNN::Convolution2D*> available;
    for(const auto* op:*MNN::GetNet(data.first)->oplists())
        if(op->name() && op->type()==MNN::OpType_Convolution && op->inputIndexes() && op->inputIndexes()->size()==1)
            available.emplace(op->name()->str(),op->main_as_Convolution2D());
    std::string line;
    while(std::getline(file,line)) {
        std::istringstream values(line);std::string name;size_t n;
        if(!(values>>name) || name[0]=='#')continue;
        if(!(values>>n) || !n || n>4096 || n%4 || projections_.count(name))
            throw std::runtime_error("Invalid projection range header: "+name);
        const auto it=available.find(name);
        const auto* conv=it==available.end() ? nullptr : it->second;
        const auto* c=conv ? conv->common() : nullptr;
        if(!c || c->inputCount()<=0 || c->inputCount()>2048 || c->inputCount()%4 || c->outputCount()!=int(n) ||
           c->group()!=1 || c->kernelX()!=1 || c->kernelY()!=1 || c->strideX()!=1 || c->strideY()!=1 ||
           c->dilateX()!=1 || c->dilateY()!=1 || c->padX()!=0 || c->padY()!=0 || c->relu() || c->relu6() ||
           conv->quanParameter() || conv->symmetricQuan() || !conv->weight() ||
           conv->weight()->size()!=size_t(c->inputCount())*n || (conv->bias() && conv->bias()->size()!=n))
            throw std::runtime_error("FP32 pointwise convolution contract mismatch: "+name);
        std::vector<float> low(n),high(n);
        for(size_t j=0;j<n;++j)if(!(values>>low[j]>>high[j]))throw std::runtime_error("Missing calibration range: "+name);
        std::string extra;std::vector<size_t> fp32_columns;
        if(values>>extra) {
            if(extra!="fp32")throw std::runtime_error("Unexpected calibration data: "+name);
            size_t column;
            while(values>>column) {
                if(column>=n || std::find(fp32_columns.begin(),fp32_columns.end(),column)!=fp32_columns.end())
                    throw std::runtime_error("Invalid FP32 projection column: "+name);
                fp32_columns.push_back(column);
            }
            if(!values.eof() || fp32_columns.empty())throw std::runtime_error("Invalid FP32 projection column list: "+name);
        }
        auto& projection=projections_[name];projection.k=c->inputCount();projection.n=n;
        projection.execution.reset(new MnnFpgaProjection(conv->weight()->data(),
            conv->bias() ? conv->bias()->data() : nullptr,projection.k,n,low,high,fp32_columns));
    }
    if(projections_.empty())throw std::runtime_error("No projection calibration ranges");
}
bool NativeProjections::before(const std::string& name,const std::vector<MNN::Tensor*>& inputs) {
    const auto it=projections_.find(name);
    if(it==projections_.end())return true;
    if(pending_ || inputs.size()!=1)throw std::runtime_error("Unexpected native projection inputs");
    const auto rows=rows_of(inputs[0],it->second.k);
    input_=inputs[0];pending_=&it->second;
    if(verify_) {
        auto shape=input_->shape();shape[1]=int(pending_->n);
        verification_.reset(MNN::Tensor::create<float>(shape,nullptr,storage_layout(input_)));
        if(!verification_ || !verification_->host<float>())throw std::runtime_error("Cannot allocate projection oracle buffer");
        pending_->execution->run(input_,verification_.get(),rows,threads_);
        return true;
    }
    return false;
}
void NativeProjections::after(const std::string& name,const std::vector<MNN::Tensor*>& outputs) {
    if(!pending_)return;
    const auto it=projections_.find(name);
    if(it==projections_.end() || &it->second!=pending_ || outputs.size()!=1)
        throw std::runtime_error("Unexpected native projection outputs");
    const auto rows=rows_of(input_,pending_->k);
    if(rows_of(outputs[0],pending_->n)!=rows || outputs[0]->batch()!=input_->batch() ||
       outputs[0]->height()!=input_->height() || outputs[0]->width()!=input_->width())
        throw std::runtime_error("Native projection spatial extent mismatch");
    if(verify_) {
        if(storage_layout(verification_.get())!=storage_layout(outputs[0]))
            throw std::runtime_error("Cannot compare projection tensors with different layouts");
        const auto* candidate=verification_->host<float>();const auto* reference=outputs[0]->host<float>();
        for(size_t i=0;i<rows*pending_->n;++i) {
            const double e=double(candidate[i])-reference[i];
            if(!std::isfinite(e))throw std::runtime_error("Nonfinite projection verification result");
            pending_->error+=e*e;pending_->reference+=double(reference[i])*reference[i];
            pending_->maximum=std::max(pending_->maximum,std::abs(e));
        }
        verification_.reset();
    } else pending_->execution->run(input_,outputs[0],rows,threads_);
    ++pending_->calls;pending_=nullptr;input_=nullptr;
}
void NativeProjections::reset() {
    if(pending_)throw std::runtime_error("Incomplete native projection callback");
    for(auto& item:projections_) {
        auto& p=item.second;p.calls=0;p.error=p.reference=p.maximum=0;
    }
}
bool NativeProjections::valid(size_t runs) const {
    if(pending_)return false;
    for(const auto& projection:projections_)if(projection.second.calls!=runs)return false;
    return true;
}
std::string NativeProjections::json() const {
    std::ostringstream output;output<<"{\"verification\":"<<(verify_?"true":"false")<<",\"operations\":[";bool first=true;
    for(const auto& item:projections_) {
        if(!first)output<<',';
        const auto& p=item.second;
        first=false;output<<"{\"name\":\""<<item.first<<"\",\"calls\":"<<p.calls;
        if(verify_)output<<",\"relative_l2\":"<<std::sqrt(p.error/std::max(p.reference,1e-20))
                         <<",\"max_abs\":"<<p.maximum;
        output<<'}';
    }
    output<<"]}";return output.str();
}
