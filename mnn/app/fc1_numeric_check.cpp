#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "MNN_generated.h"
#include "image_input.hpp"
#include "matmul_soc_uio.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t M=784,K=128,N=512;
const char* OP="/layers.1/blocks.0/mlp/fc1/Add_output_0__matmul_converted";
struct Oracle : fpga::MatmulSocIo {
    std::array<uint32_t,9> regs{};
    std::array<int8_t,16384> a{},b{};std::array<int8_t,65536> c{};
    uint32_t read(unsigned off) override {return off==0x50 ? 2:regs.at(off/16);}
    void write(unsigned off,uint32_t v) override {
        regs.at(off/16)=v;if(off) return;
        const unsigned m=regs[1],k=regs[2],n=regs[4],q=regs[6];
        for(unsigned i=0;i<m;++i) for(unsigned j=0;j<n;++j) {
            int output=0;
            for(unsigned band=0;band<k;band+=16) {
                int64_t sum=0;for(unsigned p=band;p<band+16;++p) sum+=int(a[i*k+p])*int(b[p*n+j]);
                const int64_t product=sum*q;
                const int scaled=int(product>=0?product/65536:-((-product+65535)/65536));
                output=std::max(-128,std::min(127,output+std::max(-128,std::min(127,scaled))));
            }
            c[i*n+j]=int8_t(output);
        }
    }
    void write_beat(unsigned off,const Beat& v) override {
        auto* p=off<0x5000 ? a.data()+off-0x1000:b.data()+off-0x5000;std::copy(v.begin(),v.end(),p);
    }
    Beat read_beat(unsigned off) override {Beat v{};std::copy_n(c.data()+off-0x9000,16,v.begin());return v;}
    void write_buffer(unsigned off,const int8_t* p,size_t n) override {
        std::copy_n(p,n,off==0x1000?a.data():b.data());
    }
    void read_buffer(unsigned,int8_t* p,size_t n) override {std::copy_n(c.data(),n,p);}
};
void metrics(std::ostream& out,const std::vector<float>& reference,const std::vector<float>& actual) {
    double maxerr=0,ee=0,rr=0,bias=0;
    for(size_t i=0;i<reference.size();++i) {const double d=actual[i]-double(reference[i]);maxerr=std::max(maxerr,std::abs(d));ee+=d*d;rr+=double(reference[i])*reference[i];bias+=d;}
    out<<"{\"max_abs\":"<<maxerr<<",\"rmse\":"<<std::sqrt(ee/reference.size())<<",\"relative_l2\":"<<std::sqrt(ee/rr)<<",\"mean_signed_error\":"<<bias/reference.size()<<'}';
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: fc1_numeric_check MODEL IMAGE OUTPUT.json (requires root/DMA)");
        auto data=image_input(argv[2],224,224,0,{0.485f,0.456f,0.406f},{0.229f,0.224f,0.225f});
        std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(argv[1]));
        if(!net) throw std::runtime_error("No model");
        net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
        MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=2;
        MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_High;config.backendConfig=&backend;
        auto* session=net->createSession(config);auto* input=net->getSessionInput(session,nullptr);
        net->resizeTensor(input,{1,3,224,224});net->resizeSession(session);
        MNN::Tensor host(input,MNN::Tensor::CAFFE);std::copy(data.begin(),data.end(),host.host<float>());
        if(!input->copyFromHostTensor(&host)) throw std::runtime_error("Input copy failed");
        std::vector<float> a,native;
        auto before=[&](const std::vector<MNN::Tensor*>& ts,const MNN::OperatorInfo* info) {
            if(info->name()==OP) {
                MNN::Tensor dense(ts.at(0),MNN::Tensor::CAFFE);
                if(!ts.at(0)->copyToHostTensor(&dense)) throw std::runtime_error("fc1 input conversion failed");
                a.assign(dense.host<float>(),dense.host<float>()+M*K);
            }
            return true;
        };
        auto after=[&](const std::vector<MNN::Tensor*>& ts,const MNN::OperatorInfo* info) {
            if(info->name()==OP) {
                MNN::Tensor dense(ts.at(0),MNN::Tensor::CAFFE);
                if(!ts.at(0)->copyToHostTensor(&dense)) throw std::runtime_error("fc1 output conversion failed");
                native.assign(dense.host<float>(),dense.host<float>()+M*N);
            }
            return true;
        };
        if(net->runSessionWithCallBackInfo(session,before,after,true)!=MNN::NO_ERROR || a.size()!=M*K || native.size()!=M*N)
            throw std::runtime_error("Failed to capture native fc1");
        const auto buffer=net->getModelBuffer();const auto* model=MNN::GetNet(buffer.first);
        const MNN::Convolution2D* conv=nullptr;
        for(const auto* op:*model->oplists()) if(op->name() && op->name()->str()==OP) conv=op->main_as_Convolution2D();
        if(!conv || !conv->weight() || conv->weight()->size()!=K*N) throw std::runtime_error("No fc1 parameters");
        std::vector<float> bias(N,0);if(conv->bias()) std::copy_n(conv->bias()->data(),N,bias.begin());
        const auto qa=fpga::prepare_matmul_soc_operand(a.data(),M,K);
        const auto qb=fpga::prepare_matmul_soc_operand(conv->weight()->data(),K,N,true);
        std::vector<float> fp32(M*N),int8_full(M*N);
        double native_maxerr=0;
        for(size_t i=0;i<M;++i) for(size_t j=0;j<N;++j) {
            double sum=bias[j];int64_t integer=0;
            for(size_t p=0;p<K;++p) {sum+=double(a[i*K+p])*conv->weight()->Get(j*K+p);integer+=int(qa.values[i*K+p])*int(qb.values[p*N+j]);}
            fp32[i*N+j]=float(sum);int8_full[i*N+j]=float(integer*qa.scale*qb.scale+bias[j]);
            native_maxerr=std::max(native_maxerr,std::abs(sum-native[i*N+j]));
        }
        if(native_maxerr>1e-3) throw std::runtime_error("Native fc1 differs from independent FP32 OI GEMM: layout/weight interpretation wrong");
        std::cout<<"PASS native fc1 equals independent FP32 GEMM; max_error="<<native_maxerr<<'\n';
        std::ofstream out(argv[3]);out<<std::setprecision(12)<<"{\"M\":"<<M<<",\"K\":"<<K<<",\"N\":"<<N
            <<",\"operand_scales\":["<<qa.scale<<','<<qb.scale<<"],\"independent_fp32_vs_native\":";
        metrics(out,native,fp32);out<<",\"int8_operands_int32_full_accum_vs_native\":";metrics(out,native,int8_full);
        fpga::MatmulSocUio hardware("/dev/mem",true,true);
        for(const auto policy:{fpga::MatmulSocTilePolicy::Fixed64,fpga::MatmulSocTilePolicy::Capacity}) {
            const auto plan=fpga::plan_matmul_soc(M,N,K,policy);Oracle oracle;
            std::vector<float> expected(M*N),actual(M*N);
            fpga::matmul_soc_gemm_prepared(oracle,qa,qb,expected.data(),false,bias.data(),std::chrono::milliseconds(1000),&plan);
            fpga::matmul_soc_gemm_prepared(hardware,qa,qb,actual.data(),false,bias.data(),std::chrono::milliseconds(1000),&plan);
            if(expected!=actual) throw std::runtime_error("FPGA fc1 differs from independent banded arithmetic oracle");
            const char* name=policy==fpga::MatmulSocTilePolicy::Fixed64?"fixed64":"capacity";
            out<<",\""<<name<<"\":{\"hardware_matches_banded_oracle\":true,\"vs_native\":";metrics(out,native,actual);
            out<<",\"vs_full_int32_accum\":";metrics(out,int8_full,actual);out<<'}';
            std::cout<<"PASS "<<name<<" hardware equals banded arithmetic oracle, "<<plan.jobs<<" jobs\n";
        }
        out<<"}\n";out.close();if(!out) throw std::runtime_error("Cannot save numeric check");
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
