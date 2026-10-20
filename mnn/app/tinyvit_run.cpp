#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "fpga_bridge.hpp"
#include "image_input.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Options {
    std::string model,image,raw,output,dump;
    int width=0,height=0,threads=2,runs=1,warmup=0,resize_short=0,topk=5;
    bool zeros=false,mean_set=false,std_set=false,require_fpga=false;
    std::array<float,3> mean{0,0,0},stddev{1,1,1};
};
const char* usage=R"(Usage: tinyvit_run --model tiny-vit.mnn --height H --width W
    (--image image.jpg --mean R,G,B --std R,G,B | --input-f32 input.f32 | --zeros)
    [--resize-short N] [--threads N] [--runs N] [--warmup N]
    [--output-dir DIR] [--dump-input input.f32] [--topk N] [--require-fpga]

Loads the full model from disk. Requires one NCHW FP32 RGB input, batch 1.
Image: decode RGB, bilinear resize, normalize (pixel/255 - mean)/std.
Default resize stretches to WxH. --resize-short preserves aspect ratio then
center-crops WxH. Mean/std are required for images; choose your export's values.
Raw input: exactly 3*H*W little-endian FP32 values, already preprocessed in NCHW.
--zeros is a runtime smoke test, not an image accuracy test.
Outputs: raw FP32 files plus JSON shape/name metadata; top-k only for vectors.
CPU is used by default. MNN_FPGA_UIO enables the existing restricted tile path.
--require-fpga fails if no FPGA submissions complete during measured runs.
)";

int integer(const std::string& s,int minimum,int maximum) {
    size_t end=0; int v=std::stoi(s,&end);
    if(end!=s.size() || v<minimum || v>maximum) throw std::runtime_error("Invalid integer: "+s);
    return v;
}
std::array<float,3> triple(const std::string& s,bool positive) {
    std::array<float,3> out{}; std::istringstream in(s); std::string part;
    for(auto& x:out) {
        if(!std::getline(in,part,',')) throw std::runtime_error("Expected three comma-separated values");
        size_t end=0; x=std::stof(part,&end);
        if(end!=part.size() || !std::isfinite(x) || (positive && x<=0))
            throw std::runtime_error("Mean must be finite; std must be finite and positive");
    }
    if(std::getline(in,part,',') || (!s.empty() && s.back()==',')) throw std::runtime_error("Expected exactly three values");
    return out;
}
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(key=="--help" || key=="-h") {std::cout<<usage; std::exit(0);}
        if(key=="--zeros") {o.zeros=true; continue;}
        if(key=="--require-fpga") {o.require_fpga=true; continue;}
        if(i+1==argc) throw std::runtime_error("Missing value for "+key);
        const std::string value=argv[++i];
        if(key=="--model") o.model=value;
        else if(key=="--image") o.image=value;
        else if(key=="--input-f32") o.raw=value;
        else if(key=="--output-dir") o.output=value;
        else if(key=="--dump-input") o.dump=value;
        else if(key=="--height") o.height=integer(value,1,4096);
        else if(key=="--width") o.width=integer(value,1,4096);
        else if(key=="--threads") o.threads=integer(value,1,64);
        else if(key=="--runs") o.runs=integer(value,1,10000);
        else if(key=="--warmup") o.warmup=integer(value,0,10000);
        else if(key=="--resize-short") o.resize_short=integer(value,1,16384);
        else if(key=="--topk") o.topk=integer(value,0,1000);
        else if(key=="--mean") {o.mean=triple(value,false); o.mean_set=true;}
        else if(key=="--std") {o.stddev=triple(value,true); o.std_set=true;}
        else throw std::runtime_error("Unknown option: "+key);
    }
    if(o.model.empty() || !o.height || !o.width) throw std::runtime_error("--model, --height and --width are required");
    if(int(o.zeros)+int(!o.image.empty())+int(!o.raw.empty())!=1)
        throw std::runtime_error("Choose exactly one of --image, --input-f32, or --zeros");
    if(!o.image.empty() && (!o.mean_set || !o.std_set))
        throw std::runtime_error("Image input requires explicit --mean and --std");
    if(o.image.empty() && (o.mean_set || o.std_set || o.resize_short))
        throw std::runtime_error("Mean/std/resize options apply only to --image");
    return o;
}
std::string json(const std::string& s) {
    std::ostringstream out; out<<'"';
    for(unsigned char c:s) {
        if(c=='"' || c=='\\') out<<'\\'<<char(c);
        else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<char(c);
    }
    out<<'"'; return out.str();
}
std::string shape(const MNN::Tensor& t) {
    std::ostringstream out; out<<'[';
    for(int i=0;i<t.dimensions();++i) {if(i) out<<','; out<<t.length(i);}
    out<<']'; return out.str();
}
bool fp32(const MNN::Tensor& t) {
    auto type=t.getType(); return type.code==halide_type_float && type.bits==32 && type.lanes==1;
}
void save(const std::string& path,const float* data,size_t count) {
    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(data),std::streamsize(count*sizeof(float)));
    out.close();
    if(!out) throw std::runtime_error("Cannot write "+path);
}
}

int main(int argc,char** argv) {
    try {
        const auto o=parse(argc,argv);
        const uint16_t endian=1;
        if(*reinterpret_cast<const unsigned char*>(&endian)!=1 || sizeof(float)!=4 ||
           !std::numeric_limits<float>::is_iec559) throw std::runtime_error("Requires little-endian IEEE FP32 host");
        std::vector<float> data(size_t(3)*o.width*o.height,0.0f);
        if(!o.image.empty()) data=image_input(o.image,o.width,o.height,o.resize_short,o.mean,o.stddev);
        if(!o.raw.empty()) {
            std::ifstream in(o.raw,std::ios::binary|std::ios::ate);
            if(!in || in.tellg()!=std::streamoff(data.size()*sizeof(float)))
                throw std::runtime_error("Input FP32 file must contain exactly 3*H*W floats");
            in.seekg(0); in.read(reinterpret_cast<char*>(data.data()),std::streamsize(data.size()*sizeof(float)));
            if(!in) throw std::runtime_error("Cannot read FP32 input");
        }
        for(float v:data) if(!std::isfinite(v)) throw std::runtime_error("Input contains NaN/Inf");
        if(!o.dump.empty()) save(o.dump,data.data(),data.size());
        std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(o.model.c_str()));
        if(!net) throw std::runtime_error("Cannot load MNN model: "+o.model);
        net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
        MNN::ScheduleConfig config; config.type=MNN_FORWARD_CPU; config.numThread=o.threads;
        MNN::BackendConfig backend; backend.precision=MNN::BackendConfig::Precision_High;
        config.backendConfig=&backend;
        auto* session=net->createSession(config);
        if(!session) throw std::runtime_error("Cannot create MNN session");
        const auto& inputs=net->getSessionInputAll(session);
        if(inputs.size()!=1) throw std::runtime_error("Expected exactly one model input");
        auto* input=inputs.begin()->second;
        if(!fp32(*input) || input->dimensions()!=4 || input->getDimensionType()!=MNN::Tensor::CAFFE ||
           input->length(0)!=1 || input->length(1)!=3)
            throw std::runtime_error("Expected input shape [1,3,H,W] in FP32 NCHW format");
        for(int axis=2;axis<4;++axis) {
            const int wanted=axis==2 ? o.height : o.width;
            if(input->length(axis)>0 && input->length(axis)!=wanted)
                throw std::runtime_error("Requested size conflicts with a fixed model input dimension");
        }
        net->resizeTensor(input,{1,3,o.height,o.width}); net->resizeSession(session);
        int resize_status=-1;
        if(!net->getSessionInfo(session,MNN::Interpreter::RESIZE_STATUS,&resize_status) || resize_status!=0)
            throw std::runtime_error("Model cannot resize to requested input dimensions");
        MNN::Tensor host(input,MNN::Tensor::CAFFE);
        if(size_t(host.elementSize())!=data.size() || !host.host<float>()) throw std::runtime_error("Input allocation mismatch");
        std::copy(data.begin(),data.end(),host.host<float>());
        if(!input->copyFromHostTensor(&host)) throw std::runtime_error("Cannot upload input tensor");
        std::cout<<"Model: "<<o.model<<"\nInput: "<<inputs.begin()->first<<' '<<shape(host)<<" FP32 NCHW\n";
        if(o.zeros) std::cout<<"Input mode: synthetic zeros (smoke test only)\n";
        else if(!o.raw.empty()) std::cout<<"Input mode: already-preprocessed FP32 tensor\n";
        else {
            std::cout<<"Input mode: RGB, "<<(o.resize_short ? "resize short side then center crop" : "stretch resize")<<"; (pixel/255 - mean)/std\n";
            std::cout<<"Mean: "<<o.mean[0]<<','<<o.mean[1]<<','<<o.mean[2]<<"; std: "<<o.stddev[0]<<','<<o.stddev[1]<<','<<o.stddev[2]<<'\n';
        }
        auto run=[&] {auto rc=net->runSession(session); if(rc!=MNN::NO_ERROR) throw std::runtime_error("MNN runSession failed, code "+std::to_string(int(rc)));};
        for(int i=0;i<o.warmup;++i) run();
        const auto before=mnn_fpga_completed_jobs();
        double total=0;
        for(int i=0;i<o.runs;++i) {
            auto start=std::chrono::steady_clock::now(); run();
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            total+=ms; std::cout<<"Inference "<<i+1<<": "<<ms<<" ms\n";
        }
        const auto offloads=mnn_fpga_completed_jobs()-before;
        std::cout<<"Mean inference: "<<total/o.runs<<" ms (excludes loading/preprocessing/output copy)\nFPGA submissions: "<<offloads<<'\n';
        if(o.require_fpga && !offloads) throw std::runtime_error("No FPGA submissions; CPU inference is not FPGA offload");
        std::ofstream manifest;
        if(!o.output.empty()) {
            std::filesystem::create_directories(o.output);
            manifest.open(std::filesystem::path(o.output)/"outputs.json");
            if(!manifest) throw std::runtime_error("Cannot create output manifest");
            manifest<<"{\"model\":"<<json(o.model)<<",\"input_shape\":"<<shape(host)<<",\"outputs\":[\n";
        }
        unsigned index=0;
        const auto& outputs=net->getSessionOutputAll(session);
        if(outputs.empty()) throw std::runtime_error("Model has no outputs");
        for(const auto& entry:outputs) {
            if(!fp32(*entry.second)) throw std::runtime_error("Non-FP32 output: "+entry.first);
            MNN::Tensor out(entry.second,MNN::Tensor::CAFFE);
            if(out.elementSize()<=0 || !out.host<float>() || !entry.second->copyToHostTensor(&out))
                throw std::runtime_error("Cannot copy output: "+entry.first);
            const float* p=out.host<float>(); const int count=out.elementSize();
            for(int i=0;i<count;++i) if(!std::isfinite(p[i])) throw std::runtime_error("Nonfinite output: "+entry.first);
            const auto limits=std::minmax_element(p,p+count);
            std::cout<<"Output "<<entry.first<<": "<<shape(out)<<" FP32; "<<count<<" values; min="<<*limits.first<<", max="<<*limits.second<<"\nFirst values:";
            for(int i=0;i<std::min(count,8);++i) std::cout<<' '<<p[i];
            std::cout<<'\n';
            if(o.topk && (out.dimensions()==1 || (out.dimensions()==2 && out.length(0)==1))) {
                std::vector<int> order(count); std::iota(order.begin(),order.end(),0);
                int k=std::min(count,o.topk);
                std::partial_sort(order.begin(),order.begin()+k,order.end(),[&](int a,int b){return p[a]>p[b];});
                std::cout<<"Top indices/scores (raw model values, not probabilities):\n";
                for(int i=0;i<k;++i) std::cout<<"  "<<order[i]<<": "<<p[order[i]]<<'\n';
            }
            if(!o.output.empty()) {
                std::string filename="output_"+std::to_string(index)+".f32";
                save((std::filesystem::path(o.output)/filename).string(),p,size_t(count));
                if(index) manifest<<",\n";
                manifest<<"{\"name\":"<<json(entry.first)<<",\"file\":"<<json(filename)<<",\"shape\":"<<shape(out)
                        <<",\"dtype\":\"float32\",\"byte_order\":\"little\",\"dimension_type\":\"CAFFE\"}";
            }
            ++index;
        }
        if(manifest.is_open()) {manifest<<"\n]}\n"; manifest.close(); if(!manifest) throw std::runtime_error("Cannot finish output manifest");}
        return 0;
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<"\nUse --help for usage.\n"; return 1;}
}
