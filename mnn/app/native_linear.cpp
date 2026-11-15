#include "native_linear.hpp"
#include "fpga_bridge.hpp"
#include "MNN_generated.h"
#include "core/ConvolutionCommon.hpp"
#include <MNN/Tensor.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>

namespace {
struct Layer {
    const MNN::Op* op=nullptr;
    const MNN::TensorQuantInfo* input=nullptr,*output=nullptr;
    size_t k=0,n=0;
};
std::array<int8_t,256> activation_table(const MNN::Op& gelu,const MNN::Op& quantize,
                                      const MNN::TensorQuantInfo& before,const MNN::TensorQuantInfo& after) {
    // Execute the model's actual MNN GELU and INT8 cast on all 256 codes.
    // This avoids substituting a different GELU approximation/rounding rule.
    MNN::NetT graph;graph.tensorName={"input","gelu","output"};graph.tensorNumber=3;graph.outputName={"output"};
    std::unique_ptr<MNN::OpT> input(new MNN::OpT);input->name="input";input->type=MNN::OpType_Input;
    input->outputIndexes={0};MNN::InputT param;param.dims={256,4,1,1};param.dformat=MNN::MNN_DATA_FORMAT_NC4HW4;
    input->main.Set(std::move(param));graph.oplists.push_back(std::move(input));
    for(const auto* original:{&gelu,&quantize}) {
        std::unique_ptr<MNN::OpT> node(original->UnPack());
        const int i=int(graph.oplists.size());node->name=i==1?"gelu":"output";node->inputIndexes={i-1};node->outputIndexes={i};
        graph.oplists.push_back(std::move(node));
    }
    std::unique_ptr<MNN::TensorDescribeT> d(new MNN::TensorDescribeT);
    d->index=2;d->quantInfo.reset(after.UnPack());graph.extraTensorDescribe.push_back(std::move(d));
    flatbuffers::FlatBufferBuilder builder;builder.Finish(MNN::Net::Pack(builder,&graph));
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromBuffer(builder.GetBufferPointer(),builder.GetSize()));
    if(!net)throw std::runtime_error("Cannot construct native activation LUT graph");
    MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=1;
    MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_High;config.backendConfig=&backend;
    auto* session=net->createSession(config);if(!session)throw std::runtime_error("Cannot create activation LUT session");
    auto* in=net->getSessionInput(session,nullptr);MNN::Tensor host(in,MNN::Tensor::CAFFE);
    for(unsigned i=0;i<256;++i)for(unsigned j=0;j<4;++j)host.host<float>()[i*4+j]=float(int8_t(i))*before.scale();
    if(!in->copyFromHostTensor(&host) || net->runSession(session)!=MNN::NO_ERROR)throw std::runtime_error("Activation LUT evaluation failed");
    auto* out=net->getSessionOutput(session,nullptr);MNN::Tensor output(out,MNN::Tensor::CAFFE);
    if(out->getType().bits!=8 || !out->copyToHostTensor(&output))throw std::runtime_error("Activation LUT output is not INT8");
    std::array<int8_t,256> table{};
    for(unsigned i=0;i<256;++i) {
        table[i]=output.host<int8_t>()[i*4];
        for(unsigned j=1;j<4;++j)if(table[i]!=output.host<int8_t>()[i*4+j])throw std::runtime_error("Channel-dependent activation LUT");
    }
    return table;
}
}
size_t prepare_native_linears(const MNN::Interpreter& net,const std::string& allowlist,bool fuse,bool plan_only) {
    std::set<std::string> allowed;
    if(!allowlist.empty()) {
        std::ifstream file(allowlist);if(!file)throw std::runtime_error("Cannot read native allowlist");
        std::string name;while(std::getline(file,name)) {
            if(!name.empty() && name.back()=='\r')name.pop_back();
            if(!name.empty() && name[0]!='#')allowed.insert(name);
        }
        if(allowed.empty())throw std::runtime_error("Empty native allowlist");
    }
    const auto buffer=net.getModelBuffer();flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer.first),buffer.second);
    if(!MNN::VerifyNetBuffer(verifier))throw std::runtime_error("Invalid native model");
    const auto* model=MNN::GetNet(buffer.first);
    std::map<int,const MNN::TensorQuantInfo*> quant;
    if(model->extraTensorDescribe())for(const auto* d:*model->extraTensorDescribe())if(d->quantInfo())quant[d->index()]=d->quantInfo();
    std::map<int,std::vector<const MNN::Op*>> consumers;
    std::map<std::string,Layer> layers;std::vector<std::string> order;
    for(const auto* op:*model->oplists()) {
        if(op->inputIndexes())for(auto i:*op->inputIndexes())consumers[i].push_back(op);
        if(!op->name() || (!allowed.empty() && !allowed.count(op->name()->str())))continue;
        const auto* c=op->main_as_Convolution2D();if(!c || !c->common() || !c->quanParameter())continue;
        const auto* p=c->common();
        if(p->group()!=1 || p->kernelX()!=1 || p->kernelY()!=1 || p->strideX()!=1 || p->strideY()!=1 ||
           p->dilateX()!=1 || p->dilateY()!=1 || p->relu() || p->relu6() || p->inputCount()<=0 || p->inputCount()>1024 ||
           p->inputCount()%4 || p->outputCount()<=0 || p->outputCount()>4096 || p->outputCount()%4 ||
           !op->inputIndexes() || op->inputIndexes()->size()!=1 || !op->outputIndexes() || op->outputIndexes()->size()!=1)continue;
        const auto qi=quant.find(op->inputIndexes()->Get(0)),qo=quant.find(op->outputIndexes()->Get(0));
        if(qi==quant.end() || qo==quant.end())continue;
        auto valid=[](const MNN::TensorQuantInfo* q){return q->type()==MNN::DataType_DT_INT8 && q->zero()==0 &&
            std::isfinite(q->scale()) && q->scale()>0 && q->min()==-127 && q->max()==127;};
        if(!valid(qi->second) || !valid(qo->second))continue;
        if(c->quanParameter()->scaleIn()!=qi->second->scale() || c->quanParameter()->scaleOut()!=qo->second->scale() ||
           !c->bias() || c->bias()->size()!=unsigned(p->outputCount()))throw std::runtime_error("Native scale/bias mismatch");
        const auto name=op->name()->str();Layer l;l.op=op;l.input=qi->second;l.output=qo->second;l.k=p->inputCount();l.n=p->outputCount();
        layers.emplace(name,l);order.push_back(name);
    }
    if(layers.empty())throw std::runtime_error("No supported native W8A8 linear operations");
    if(!allowed.empty() && layers.size()!=allowed.size())throw std::runtime_error("Native allowlist contains unsupported/missing operations");
    for(const auto& name:order) {
        const auto& l=layers.at(name);auto weights=MNN::ConvolutionCommon::load(l.op,nullptr,false,true);
        if(!weights || weights->asymmetric || size_t(weights->weight.size())!=l.k*l.n || size_t(weights->alphaSize)!=l.n || !weights->getAlphaFloat())
            throw std::runtime_error("Unsupported native weight encoding: "+name);
        for(size_t j=0;j<l.n;++j) {
            const double multiplier=double(l.input->scale())*weights->getAlphaFloat()[j]/l.output->scale();
            const double bias=double(l.op->main_as_Convolution2D()->bias()->Get(unsigned(j)))/l.output->scale()*65536;
            if(!std::isfinite(multiplier) || multiplier<1.0/33554432 || multiplier>=1 || !std::isfinite(bias) || std::abs(bias)>2147483647)
                throw std::runtime_error("Native arithmetic range unsupported: "+name);
        }
        if(!plan_only)mnn_fpga_register_native(name,weights->weight.get(),weights->getAlphaFloat(),
            l.op->main_as_Convolution2D()->bias()->data(),l.k,l.n,l.input->scale(),l.output->scale());
        std::cout<<"Native candidate: "<<name<<" K="<<l.k<<" N="<<l.n<<'\n';
    }
    size_t pairs=0;
    if(fuse)for(const auto& first:order) {
        const auto marker=first.find("/mlp/fc1/");if(marker==std::string::npos)continue;
        auto second=first;second.replace(marker+5,3,"fc2");auto next=layers.find(second);if(next==layers.end())continue;
        const auto& f=layers.at(first);const auto& s=next->second;if(f.n!=s.k)throw std::runtime_error("Resident hidden dimensions mismatch");
        int tensor=f.op->outputIndexes()->Get(0);const MNN::Op* activation=nullptr,*cast=nullptr;
        std::vector<std::string> skipped;
        for(int step=0;step<12;++step) {
            std::vector<const MNN::Op*> data_consumers;
            for(const auto* consumer:consumers[tensor])
                if(consumer->type()!=MNN::OpType_Shape && consumer->type()!=MNN::OpType_Rank)data_consumers.push_back(consumer);
            if(data_consumers.size()!=1)throw std::runtime_error("Resident intermediate has multiple data consumers");
            const auto* op=data_consumers[0];if(op==s.op)break;
            const auto type=op->type();
            if(type==MNN::OpType_UnaryOp) {if(activation)throw std::runtime_error("Multiple resident activations");activation=op;}
            else if(type==MNN::OpType_FloatToInt8)cast=op;
            else if(type!=MNN::OpType_Int8ToFloat && type!=MNN::OpType_ConvertTensor && type!=MNN::OpType_Reshape)
                throw std::runtime_error("Unsupported resident intermediate operation");
            if(!op->outputIndexes() || op->outputIndexes()->size()!=1)throw std::runtime_error("Resident chain output count mismatch");
            skipped.push_back(op->name()->str());tensor=op->outputIndexes()->Get(0);
        }
        if(tensor!=s.op->inputIndexes()->Get(0) || !activation || !cast)throw std::runtime_error("Incomplete resident activation chain");
        const auto lut=activation_table(*activation,*cast,*f.output,*s.input);
        if(!plan_only)mnn_fpga_register_native_pair(first,second,lut,skipped);
        uint32_t hash=2166136261u;for(auto x:lut)hash=(hash^uint8_t(x))*16777619u;
        std::cout<<"Native resident pair: "<<first<<" -> "<<second<<"; activation LUT FNV1a="<<hash<<"; graph chain="<<skipped.size()<<'\n';++pairs;
    }
    std::cout<<"Native plan: "<<layers.size()<<" linear operations, "<<pairs<<" resident pairs\n";
    return layers.size();
}
