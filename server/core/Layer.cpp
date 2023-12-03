#include <unordered_map>
#include <functional>
#include <string>
#include <vector>

#include "Layer.h"

static Blob dataInit(size_t h, size_t w, RandomObject* randomInit) {
    return Blob::constRandomBlob(Shape {{h, w}}, randomInit);
}

LinearLayer::LinearLayer(std::unordered_map<std::string, float> params,
                         const std::vector<TensorRef>& args, RandomObject* const randomInit) :
    W(dataInit((size_t)params["h"], (size_t)params["w"], randomInit)),
    b(dataInit(1, (size_t)params["w"], randomInit)) {

    layerOperationParams.push_back(W);
    layerOperationParams.push_back(b);
    
    Tensor multNode (mul, {args[0], W});
    pipeline.push_back(std::move(multNode));

    result = Tensor(sum, {pipeline[0], b});
}

ReLULayer::ReLULayer(std::unordered_map<std::string, float> params,
                     const std::vector<TensorRef>& args, RandomObject* randomInit) {
    result = Tensor(relu, {args[0]});
}

MSELoss::MSELoss(
    std::unordered_map<std::string, float> params,
    const std::vector<TensorRef>& args, RandomObject* randomInit
) : mean({0, 1, 2, 3}) {
    pipeline.reserve(2);

    Tensor diff(sub, {args[0], args[1]});
    pipeline.push_back(std::move(diff));

    Tensor square(sqr, {pipeline[0]});
    pipeline.push_back(std::move(square));

    result = Tensor(mean, {pipeline[1]});
}

MultLayer::MultLayer(std::unordered_map<std::string, float> params,
                     const std::vector<TensorRef>& args, RandomObject* randomInit) {
    result = Tensor(mult, {args[0].get(), args[1].get()});
}

