#pragma once
#include <MNN/Interpreter.hpp>
#include "fpga_bridge.hpp"
#include <map>
#include <memory>
#include <string>

class NativeProjections {
public:
    NativeProjections(const MNN::Interpreter& model,const std::string& ranges,int threads,bool verify);
    bool before(const std::string& name,const std::vector<MNN::Tensor*>& inputs);
    void after(const std::string& name,const std::vector<MNN::Tensor*>& outputs);
    void reset();
    bool valid(size_t runs) const;
    std::string json() const;
private:
    struct Projection {
        size_t k=0,n=0,calls=0;
        double error=0,reference=0,maximum=0;
        std::unique_ptr<MnnFpgaProjection> execution;
    };
    std::map<std::string,Projection> projections_;
    Projection* pending_=nullptr;
    const MNN::Tensor* input_=nullptr;
    std::unique_ptr<MNN::Tensor> verification_;
    int threads_;
    bool verify_;
};
