#pragma once
#include <MNN/Interpreter.hpp>
#include <string>
// Empty allowlist selects every supported static W8A8 pointwise linear op.
// plan_only validates graph/quantization and builds the GELU table on CPU,
// without opening /dev/mem or touching the board.
size_t prepare_native_linears(const MNN::Interpreter& net,const std::string& allowlist,
                             bool fuse_pairs,bool plan_only);
