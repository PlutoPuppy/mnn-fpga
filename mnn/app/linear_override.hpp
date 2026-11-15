#pragma once
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <string>
#include <vector>
// Narrow experimental callback replacement for the profiled stage-1 fc1.
// MNN calls the after callback even when before returns false, allowing output
// replacement before downstream operators run. CPU execution is skipped once.
// Explicit public tensor copies adapt NC4HW4 to/from dense CAFFE host tensors.
double prepare_linear_override(const MNN::Interpreter& net,const std::string& name);
bool execute_linear_override(const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs);
