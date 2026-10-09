#include "MNN_generated.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "fpga_bridge.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc,char** argv) {
    if (argc!=3 || (std::string(argv[1])!="--cpu" && std::string(argv[1])!="--require-fpga")) {
        std::cerr << "Usage: mnn_tile_test --cpu|--require-fpga output-model.mnn\n"; return 2;
    }
    try {
        MNN::NetT net;
        net.tensorName={"A","B","C"}; net.tensorNumber=3;
        net.outputName={"C"}; net.usage=MNN::Usage_INFERENCE;
        for(int i=0;i<2;++i) {
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);
            op->name=net.tensorName[i]; op->type=MNN::OpType_Input;
            op->outputIndexes={i}; op->main.type=MNN::OpParameter_Input;
            auto* p=new MNN::InputT;
            p->dims={16,16}; p->dtype=MNN::DataType_DT_FLOAT; p->dformat=MNN::MNN_DATA_FORMAT_NCHW;
            op->main.value=p; net.oplists.emplace_back(std::move(op));
        }
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        op->name="fpga_tile"; op->type=MNN::OpType_MatMul;
        op->inputIndexes={0,1}; op->outputIndexes={2}; op->main.type=MNN::OpParameter_MatMul;
        auto* p=new MNN::MatMulT; p->T=MNN::DataType_DT_FLOAT;
        op->main.value=p; net.oplists.emplace_back(std::move(op));
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Net::Pack(builder,&net));
        std::ofstream file(argv[2],std::ios::binary);
        file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),builder.GetSize());
        if(!file) throw std::runtime_error("Cannot save test model");
        file.close();
        std::unique_ptr<MNN::Interpreter> model(MNN::Interpreter::createFromBuffer(builder.GetBufferPointer(),builder.GetSize()));
        if(!model) throw std::runtime_error("Cannot create interpreter");
        MNN::ScheduleConfig config; config.type=MNN_FORWARD_CPU; config.numThread=1;
        MNN::BackendConfig backend; backend.precision=MNN::BackendConfig::Precision_High;
        config.backendConfig=&backend;
        auto* session=model->createSession(config);
        if(!session) throw std::runtime_error("Cannot create session");
        auto* a=model->getSessionInput(session,"A"); auto* b=model->getSessionInput(session,"B");
        auto* c=model->getSessionOutput(session,"C");
        if(!a || !b || !c) throw std::runtime_error("Missing tensors");
        MNN::Tensor ha(a,MNN::Tensor::CAFFE), hb(b,MNN::Tensor::CAFFE), hc(c,MNN::Tensor::CAFFE);
        for(int i=0;i<256;++i) { ha.host<float>()[i]=float(i%17-8); hb.host<float>()[i]=float(i%11-5); }
        a->copyFromHostTensor(&ha); b->copyFromHostTensor(&hb);
        const auto before=mnn_fpga_completed_jobs();
        if(model->runSession(session)!=MNN::NO_ERROR) throw std::runtime_error("runSession failed");
        c->copyToHostTensor(&hc);
        for(int i=0;i<16;++i) for(int j=0;j<16;++j) {
            float ref=0; for(int k=0;k<16;++k) ref+=ha.host<float>()[i*16+k]*hb.host<float>()[k*16+j];
            if(hc.host<float>()[i*16+j]!=ref) throw std::runtime_error("MNN output mismatch");
        }
        auto count=mnn_fpga_completed_jobs()-before;
        if(std::string(argv[1])=="--require-fpga" && count!=1)
            throw std::runtime_error("Expected one FPGA submission; CPU fallback is not a hardware pass");
        if(std::string(argv[1])=="--cpu" && count!=0)
            throw std::runtime_error("Unset MNN_FPGA_UIO for the CPU baseline");
        std::cout << "PASS MNN MatMul: 256 outputs; FPGA submissions=" << count << '\n';
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
