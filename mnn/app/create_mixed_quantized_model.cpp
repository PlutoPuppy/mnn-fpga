// Keep calibrated W8A8 convolutions inside explicit INT8 boundaries. Unselected
// convolutions and non-convolution arithmetic retain the original FP32 graph.
#include "MNN_generated.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::unique_ptr<MNN::NetT> load(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open input model");
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), {});
    flatbuffers::Verifier verifier(data.data(), data.size());
    if (!MNN::VerifyNetBuffer(verifier)) throw std::runtime_error("Invalid input model");
    return std::unique_ptr<MNN::NetT>(MNN::GetNet(data.data())->UnPack());
}

std::unique_ptr<MNN::OpT> cast(int input, int output, bool quantize, float scale, const std::string& name) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->name = name;
    op->type = quantize ? MNN::OpType_FloatToInt8 : MNN::OpType_Int8ToFloat;
    op->inputIndexes = {input};
    op->outputIndexes = {output};
    MNN::QuantizedFloatParamT param;
    param.tensorScale = {quantize ? 1.0f / scale : scale};
    param.clampMin = -127;
    param.clampMax = 127;
    op->main.Set(std::move(param));
    return op;
}

void describe(MNN::NetT& net, int index, const MNN::TensorQuantInfoT& quant) {
    std::unique_ptr<MNN::TensorDescribeT> desc(new MNN::TensorDescribeT);
    desc->index = index;
    desc->quantInfo.reset(new MNN::TensorQuantInfoT(quant));
    net.extraTensorDescribe.emplace_back(std::move(desc));
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) throw std::runtime_error(
            "Usage: create_mixed_quantized_model FP32.mnn CALIBRATED.mnn ALLOWLIST.txt NEW_OUTPUT.mnn [WEIGHT_ONLY_ALLOWLIST.txt]");
        if (std::ifstream(argv[4], std::ios::binary)) throw std::runtime_error("Output already exists");
        auto original = load(argv[1]);
        auto quantized = load(argv[2]);
        if (original->tensorName != quantized->tensorName || original->oplists.size() != quantized->oplists.size())
            throw std::runtime_error("Input graphs differ");
        if (!original->subgraphs.empty() || !quantized->subgraphs.empty())
            throw std::runtime_error("Subgraphs are unsupported by this model utility");
        for (const auto& d : original->extraTensorDescribe)
            if (d->quantInfo) throw std::runtime_error("Original model already contains activation quantization");

        std::ifstream policy(argv[3]);
        if (!policy) throw std::runtime_error("Cannot open allowlist");
        std::set<std::string> allowed;
        std::string line;
        while (std::getline(policy, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line.front() == '#') continue;
            if (!allowed.insert(line).second) throw std::runtime_error("Duplicate allowlist entry");
        }
        if (allowed.empty()) throw std::runtime_error("Empty allowlist");
        std::set<std::string> weightOnly, weightReplaced;
        if (argc == 6) {
            std::ifstream weights(argv[5]);
            if (!weights) throw std::runtime_error("Cannot open weight-only allowlist");
            while (std::getline(weights, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty() || line.front() == '#') continue;
                if (allowed.count(line) || !weightOnly.insert(line).second)
                    throw std::runtime_error("Overlapping or duplicate weight-only entry");
            }
            if (weightOnly.empty()) throw std::runtime_error("Empty weight-only allowlist");
        }
        std::map<int, MNN::TensorQuantInfoT> scales;
        for (const auto& d : quantized->extraTensorDescribe) {
            if (d->quantInfo && !scales.emplace(d->index, *d->quantInfo).second)
                throw std::runtime_error("Duplicate tensor quantization descriptor");
        }
        auto getScale = [&](int index) -> const MNN::TensorQuantInfoT& {
            const auto& q = scales.at(index);
            if (q.type != MNN::DataType_DT_INT8 || q.zero != 0 || !std::isfinite(q.scale) || q.scale <= 0 ||
                q.min != -127 || q.max != 127 || !std::isfinite(1.0f / q.scale))
                throw std::runtime_error("Unsupported symmetric activation calibration");
            return q;
        };
        std::vector<std::unique_ptr<MNN::OpT>> graph;
        std::set<std::string> replaced;
        for (size_t i = 0; i < original->oplists.size(); ++i) {
            auto& op = original->oplists[i];
            auto& src = quantized->oplists[i];
            if (op->name != src->name || op->type != src->type || op->inputIndexes != src->inputIndexes ||
                op->outputIndexes != src->outputIndexes) throw std::runtime_error("Operator graph mismatch");
            if (!allowed.count(op->name)) {
                if (weightOnly.count(op->name)) {
                    auto* dst = op->main.AsConvolution2D();
                    const auto* srcConv = src->main.AsConvolution2D();
                    if (op->type != MNN::OpType_Convolution || !dst || !dst->common || !srcConv ||
                        !srcConv->quanParameter || !srcConv->weight.empty() || !dst->external.empty() ||
                        !srcConv->external.empty() || !src->externalPath.empty() || !op->externalPath.empty() ||
                        dst->common->kernelX != 1 || dst->common->kernelY != 1 || dst->common->group != 1 ||
                        op->inputIndexes.size() != 1 || op->outputIndexes.size() != 1 ||
                        !weightReplaced.insert(op->name).second)
                        throw std::runtime_error("Weight-only policy requires static pointwise convolutions");
                    // Preserve original FP32 bias and graph boundaries. Memory_Low
                    // selects MNN's dynamic activation quantization at execution.
                    dst->quanParameter.reset(new MNN::IDSTQuanT(*srcConv->quanParameter));
                    dst->quanParameter->has_scaleInt = false;
                    dst->weight.clear();
                }
                graph.emplace_back(std::move(op));
                continue;
            }
            if (!replaced.insert(op->name).second) throw std::runtime_error("Ambiguous operator name");
            const auto* c = src->main.AsConvolution2D();
            if ((op->type != MNN::OpType_Convolution && op->type != MNN::OpType_ConvolutionDepthwise) ||
                !c || !c->quanParameter || !c->weight.empty() || !c->external.empty() || !src->externalPath.empty() ||
                op->inputIndexes.size() != 1 || op->outputIndexes.size() != 1)
                throw std::runtime_error("Allowlist must select static calibrated convolutions");
            const int input = op->inputIndexes[0], output = op->outputIndexes[0];
            const auto& qi = getScale(input);
            const auto& qo = getScale(output);
            if (c->quanParameter->scaleIn != qi.scale || c->quanParameter->scaleOut != qo.scale)
                throw std::runtime_error("Weight and activation scales disagree");
            const int quantInput = static_cast<int>(original->tensorName.size());
            original->tensorName.push_back(op->name + "/quant_input");
            const int quantOutput = static_cast<int>(original->tensorName.size());
            original->tensorName.push_back(op->name + "/quant_output");
            describe(*original, quantInput, qi);
            describe(*original, quantOutput, qo);
            graph.emplace_back(cast(input, quantInput, true, qi.scale, op->name + "/quantize"));
            const std::string name = op->name;
            op->main = std::move(src->main);
            op->inputIndexes[0] = quantInput;
            op->outputIndexes[0] = quantOutput;
            graph.emplace_back(std::move(op));
            graph.emplace_back(cast(quantOutput, output, false, qo.scale, name + "/dequantize"));
        }
        if (replaced != allowed) throw std::runtime_error("Allowlist contains missing operators");
        if (weightReplaced != weightOnly) throw std::runtime_error("Weight-only allowlist contains missing operators");
        original->oplists = std::move(graph);
        original->tensorNumber = static_cast<int>(original->tensorName.size());
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Net::Pack(builder, original.get()));
        std::ofstream out(argv[4], std::ios::binary);
        out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        if (!out) throw std::runtime_error("Cannot write output model");
        std::cout << "PASS: " << replaced.size() << " W8A8 convolutions, " << replaced.size() * 2
                  << " explicit conversion boundaries; " << weightReplaced.size() << " weight-only operators; "
                  << builder.GetSize() << " bytes\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << '\n';
        return 1;
    }
}
