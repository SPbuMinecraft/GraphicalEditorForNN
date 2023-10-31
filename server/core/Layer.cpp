#include <unordered_map>
#include <functional>
#include <string>
#include <vector>

#include "Layer.h"

static Blob dataInit(size_t h, size_t w, RandomObject* randomInit = nullptr) {
    return Blob {h, w, randomInit};
}

LinearLayer::LinearLayer(const LinearLayerParameters& params,
                         const std::vector<TensorRef>& args,
                         RandomObject* randomInit)
    : W(dataInit(params.inFeatures, params.outFeatures, randomInit)),
    b(dataInit(1, params.outFeatures)) {
    layerOperationParams.push_back(W);

    if (params.bias) {
        b = Tensor(dataInit(1, params.outFeatures, randomInit));
        layerOperationParams.push_back(b);
    }
    
    Tensor multNode(mul, {args[0], W});
    pipeline.push_back(multNode);

    result = Tensor(sum, {pipeline[0], b});
}

ReLULayer::ReLULayer(const std::vector<TensorRef>& args) {
    result = Tensor(relu, {args[0]});
}

Data2dLayer::Data2dLayer(const Data2dLayerParameters& params, const std::vector<float>& values)
    : width(params.width) {
    result = Tensor({params.height, width, values.data()});
}

OutputLayer::OutputLayer(const std::vector<TensorRef>& args) {
    result = Tensor(id, {args[0]}, true);
}

MSELoss::MSELoss(const std::vector<TensorRef>& args, RandomObject* randomInit) {
    pipeline.reserve(2);

    Tensor diff(sub, {args[0], args[1]});
    pipeline.push_back(diff);

    Tensor square(sqr, {pipeline[0]});
    pipeline.push_back(square);

    result = Tensor(mean, {pipeline[1]});
}

MultLayer::MultLayer(const std::vector<TensorRef>& args, RandomObject* randomInit) {
    result = Tensor(mult, {args[0].get(), args[1].get()});
}

