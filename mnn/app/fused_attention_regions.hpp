#pragma once
#include <MNN/Interpreter.hpp>
#include <memory>
#include <string>
#include <vector>

class FusedAttentionRegions {
public:
    FusedAttentionRegions(const MNN::Interpreter& model,int threads,bool verify);
    ~FusedAttentionRegions();
    bool before(const std::string& name,const std::vector<MNN::Tensor*>& inputs);
    void after(const std::string& name,const std::vector<MNN::Tensor*>& outputs);
    void reset();
    std::string json() const;
    bool valid(unsigned runs) const;
private:
    struct State;
    std::unique_ptr<State> state_;
};
