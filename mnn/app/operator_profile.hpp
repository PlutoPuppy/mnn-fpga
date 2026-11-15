#pragma once
#include <MNN/Interpreter.hpp>
#include <chrono>
#include <string>
#include <vector>

namespace operator_profile {
struct TensorDescription {
    std::vector<int> shape,strides;
    int layout=0,code=0,bits=0,lanes=0;
};
struct Record {
    std::string name,type;
    std::vector<TensorDescription> inputs,outputs;
    double ms=0,estimated_mflops=0;
};
// CPU callbacks are synchronous. Metadata capture is outside each op timer;
// its cost remains visible in full inference minus the sum of op intervals.
class Profiler {
public:
    void before(const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info);
    void after(const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info);
    void reset();
    void save(const std::string& path,const std::string& model,double inference_ms,int runs) const;
private:
    std::vector<Record> records_;
    std::chrono::steady_clock::time_point start_;
    bool active_=false;
};
}
