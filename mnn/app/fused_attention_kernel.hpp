#pragma once
#include <cstddef>

namespace tinyvit_attention {
using Softmax = void (*)(float*,const float*,size_t);
// QKV is NC4 [rows, heads*3*dim, 1, 1], with channels [head, Q/K/V, dim].
// Scratch q/k/v each have (tokens+8)*dim floats; scores has 4*(tokens+8).
// Output is NC4 [rows, heads*dim, 1, 1]. Different heads/windows are independent.
// Optional query ranges split independent rows, aligned to four except the final tail.
void head(const void* qkv,bool int8,float input_scale,const float* bias,float scale,
          size_t rows,size_t tokens,size_t heads,size_t dim,size_t window,size_t h,
          float* q,float* k,float* v,float* scores,float* output,Softmax softmax,
          size_t first_query=0,size_t last_query=0);
}
