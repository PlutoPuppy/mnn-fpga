#pragma once
#include <MNN/Interpreter.hpp>
#include <memory>
#include <string>
#include <vector>
class GeluLookup {
public:
    enum class Mode { Lookup, Reciprocal, Polynomial };
    GeluLookup(const MNN::Interpreter& net,int threads,bool verify,Mode mode);
    ~GeluLookup();
    bool before(const std::string& name,const std::vector<MNN::Tensor*>& input);
    void after(const std::string& name,const std::vector<MNN::Tensor*>& output);
    void reset();
    bool valid(unsigned runs) const;
    std::string json() const;
private:
    struct State;std::unique_ptr<State> state_;
};
