#include "MNN_generated.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
std::unique_ptr<MNN::NetT> load(const char* path) {
    std::ifstream in(path,std::ios::binary);if(!in) throw std::runtime_error("Cannot open model");
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),{});
    flatbuffers::Verifier verifier(data.data(),data.size());
    if(!MNN::VerifyNetBuffer(verifier)) throw std::runtime_error("Invalid model buffer");
    return std::unique_ptr<MNN::NetT>(MNN::GetNet(data.data())->UnPack());
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: create_fc1_quantized_model FP32.mnn CALIBRATED_W8A8.mnn OUTPUT.mnn");
        if(std::ifstream(argv[3],std::ios::binary)) throw std::runtime_error("Output already exists; choose a new model path");
        auto original=load(argv[1]),quantized=load(argv[2]);
        const std::string name="/layers.1/blocks.0/mlp/fc1/Add_output_0__matmul_converted";
        MNN::OpT* src=nullptr;MNN::OpT* dst=nullptr;
        for(auto& op:original->oplists) if(op->name==name) dst=op.get();
        for(auto& op:quantized->oplists) if(op->name==name) src=op.get();
        if(!src || !dst || src->inputIndexes!=dst->inputIndexes || src->outputIndexes!=dst->outputIndexes ||
            !src->main.AsConvolution2D() || !src->main.AsConvolution2D()->quanParameter || original->tensorName!=quantized->tensorName)
            throw std::runtime_error("Quantized/FP32 model graph or fc1 mismatch");
        dst->main=std::move(src->main);
        for(int index:{dst->inputIndexes.at(0),dst->outputIndexes.at(0)}) {
            const MNN::TensorQuantInfoT* q=nullptr;
            for(const auto& d:quantized->extraTensorDescribe) if(d->index==index) q=d->quantInfo.get();
            if(!q || q->type!=MNN::DataType_DT_INT8 || q->scale<=0 || q->zero!=0)
                throw std::runtime_error("No symmetric fc1 activation calibration");
            MNN::TensorDescribeT* desc=nullptr;
            for(auto& d:original->extraTensorDescribe) if(d->index==index) desc=d.get();
            if(!desc) {original->extraTensorDescribe.emplace_back(new MNN::TensorDescribeT);desc=original->extraTensorDescribe.back().get();desc->index=index;}
            desc->quantInfo.reset(new MNN::TensorQuantInfoT(*q));
        }
        flatbuffers::FlatBufferBuilder builder;builder.Finish(MNN::Net::Pack(builder,original.get()));
        std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),builder.GetSize());
        if(!out) throw std::runtime_error("Cannot write model");
        std::cout<<"PASS: only fc1 weights and its two activation quantization records replaced; "<<builder.GetSize()<<" bytes\n";
    } catch(const std::exception& e) {std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}
}
