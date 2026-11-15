#include "linear_override.hpp"
#include "fpga_bridge.hpp"
#include "MNN_generated.h"
#include <chrono>
#include <cstdio>
#include <stdexcept>

double prepare_linear_override(const MNN::Interpreter& net,const std::string& name) {
    const auto start=std::chrono::steady_clock::now();
    if(name!="/layers.1/blocks.0/mlp/fc1/Add_output_0__matmul_converted")
        throw std::runtime_error("Linear override is restricted to the validated stage-1 block-0 fc1");
    const auto buffer=net.getModelBuffer();
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer.first),buffer.second);
    if(!MNN::VerifyNetBuffer(verifier)) throw std::runtime_error("Invalid model buffer for linear override");
    const auto* model=MNN::GetNet(buffer.first);
    const MNN::Convolution2D* conv=nullptr;
    if(model->oplists()) for(const auto* op:*model->oplists())
        if(op->name() && op->name()->str()==name) conv=op->main_as_Convolution2D();
    if(!conv || !conv->common()) throw std::runtime_error("fc1 convolution was not found in model");
    const auto* c=conv->common();
    if(c->inputCount()!=128 || c->outputCount()!=512 || c->group()!=1 || c->kernelX()!=1 || c->kernelY()!=1 ||
       c->strideX()!=1 || c->strideY()!=1 || c->dilateX()!=1 || c->dilateY()!=1 || c->relu() || c->relu6() ||
       conv->quanParameter() || conv->symmetricQuan() || !conv->weight() || conv->weight()->size()!=128*512 ||
       (conv->bias() && conv->bias()->size()!=512))
        throw std::runtime_error("Unsupported fc1 parameters or weight format");
    if(!mnn_fpga_prepare_linear(conv->weight()->data(),conv->bias()?conv->bias()->data():nullptr,784,128,512))
        throw std::runtime_error("Failed to prepare fc1 weights and tile plan");
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
namespace {
bool supported(const MNN::Tensor* tensor,int channels) {
    const auto t=tensor->getType();
    return t.code==halide_type_float && t.bits==32 && t.lanes==1 && tensor->dimensions()==4 &&
           tensor->length(0)==784 && tensor->length(1)==channels && tensor->length(2)==1 && tensor->length(3)==1 &&
           tensor->host<float>();
}
}
bool execute_linear_override(const std::vector<MNN::Tensor*>& inputs,const std::vector<MNN::Tensor*>& outputs) {
    if(inputs.size()!=1 || outputs.size()!=1 || !supported(inputs[0],128) || !supported(outputs[0],512)) {
        std::fprintf(stderr,"fc1 override: unsupported tensor shape/type; stopping inference\n");return false;
    }
    // NC4HW4 interleaves channels across batches even with H=W=1. Public
    // tensor copies provide the actual row-major [784,C] view and restore it.
    using Clock=std::chrono::steady_clock;
    const auto start=Clock::now();
    MNN::Tensor input(inputs[0],MNN::Tensor::CAFFE);
    MNN::Tensor output(outputs[0],MNN::Tensor::CAFFE);
    const auto allocated=Clock::now();
    if(!input.host<float>() || !output.host<float>() || !inputs[0]->copyToHostTensor(&input)) return false;
    const auto copied=Clock::now();
    if(!mnn_fpga_try_linear(input.host<float>(),output.host<float>(),784,128,512)) return false;
    const auto computed=Clock::now();
    if(!outputs[0]->copyFromHostTensor(&output)) return false;
    const auto restored=Clock::now();
    auto ns=[](Clock::time_point a,Clock::time_point b) {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count());
    };
    mnn_fpga_record_linear_layout(ns(start,allocated),ns(allocated,copied),ns(computed,restored));
    return true;
}
