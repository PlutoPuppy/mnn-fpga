#include "quantized_linear.hpp"
#include "fpga_bridge.hpp"
#include "MNN_generated.h"
#include "core/ConvolutionCommon.hpp"
#include <chrono>
#include <cmath>
#include <stdexcept>

double prepare_quantized_linear(const MNN::Interpreter& net,const std::string& name) {
    const auto start=std::chrono::steady_clock::now();
    if(name!="/layers.1/blocks.0/mlp/fc1/Add_output_0__matmul_converted")
        throw std::runtime_error("Native INT8 override is restricted to stage-1 block-0 fc1");
    const auto buffer=net.getModelBuffer();
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer.first),buffer.second);
    if(!MNN::VerifyNetBuffer(verifier)) throw std::runtime_error("Invalid W8A8 model");
    const auto* model=MNN::GetNet(buffer.first);const MNN::Op* target=nullptr;
    for(const auto* op:*model->oplists()) if(op->name() && op->name()->str()==name) target=op;
    const auto* conv=target?target->main_as_Convolution2D():nullptr;
    if(!conv || !conv->common() || !conv->quanParameter()) throw std::runtime_error("fc1 lacks quantized weights");
    const auto* c=conv->common();const auto* q=conv->quanParameter();
    if(c->inputCount()!=128 || c->outputCount()!=512 || c->group()!=1 || c->kernelX()!=1 || c->kernelY()!=1 ||
        c->strideX()!=1 || c->strideY()!=1 || c->dilateX()!=1 || c->dilateY()!=1 || c->relu() || c->relu6() ||
        !conv->bias() || conv->bias()->size()!=512 || !target->inputIndexes() || target->inputIndexes()->size()!=1 ||
        !target->outputIndexes() || target->outputIndexes()->size()!=1)
        throw std::runtime_error("Unsupported native INT8 fc1 parameters");
    const MNN::TensorQuantInfo* input=nullptr;const MNN::TensorQuantInfo* output=nullptr;
    if(model->extraTensorDescribe()) for(const auto* d:*model->extraTensorDescribe()) {
        if(d->index()==target->inputIndexes()->Get(0)) input=d->quantInfo();
        if(d->index()==target->outputIndexes()->Get(0)) output=d->quantInfo();
    }
    if(!input || !output || input->type()!=MNN::DataType_DT_INT8 || output->type()!=MNN::DataType_DT_INT8 ||
        input->zero()!=0 || output->zero()!=0 || input->scale()<=0 || output->scale()<=0)
        throw std::runtime_error("fc1 requires calibrated symmetric INT8 activation metadata");
    auto weights=MNN::ConvolutionCommon::load(target,nullptr,false,true);
    if(!weights || weights->asymmetric || weights->weight.size()!=128*512 || weights->alphaSize!=512)
        throw std::runtime_error("fc1 requires symmetric per-channel INT8 weights without blocks");
    const float* scales=weights->getAlphaFloat();
    if(!scales || q->scaleIn()!=input->scale() || q->scaleOut()!=output->scale())
        throw std::runtime_error("fc1 model activation scales disagree");
    if(!mnn_fpga_prepare_linear_int8(weights->weight.get(),scales,conv->bias()->data(),784,128,512,
        input->scale(),output->scale(),int(output->min()),int(output->max())))
        throw std::runtime_error("Cannot prepare native INT8 fc1");
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
