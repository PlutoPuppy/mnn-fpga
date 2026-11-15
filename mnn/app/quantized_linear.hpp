#pragma once
#include <MNN/Interpreter.hpp>
#include <string>
// Prepare native per-channel INT8 weights and fixed activation/output scales.
double prepare_quantized_linear(const MNN::Interpreter& net,const std::string& name);
