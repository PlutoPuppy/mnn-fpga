#pragma once
#include <cstddef>
namespace tinyvit_gelu {
constexpr size_t intervals=4096;
// Reference samples are GELU(-8+i/256), i=0..4096. Two coefficients per bin.
void coefficients(float* table,const float* samples);
void apply(float* destination,const float* source,size_t count,const float* table);
// Same rational GELU polynomial as ARM MNN, using two NEON reciprocal refinements.
void reciprocal(float* destination,const float* source,size_t count);
void polynomial(float* destination,const float* source,size_t count);
}
