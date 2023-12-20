#include <unordered_map>
#include <functional>
#include <string>

#include "Layer.h"

static Blob dataInit(size_t h, size_t w, RandomObject* randomInit) {
    return Blob::constRandomBlob(Shape {{h, w}}, randomInit);
}

LinearLayer::LinearLayer(const LinearLayerParameters& params,
                         const std::vector<TensorRef>& args,
                         RandomObject* randomInit)
    : W(dataInit(params.inFeatures, params.outFeatures, randomInit)) {
    layerOperationParams.push_back(W);

    if (params.bias) {
        b.emplace(dataInit(1, params.outFeatures, randomInit));
        layerOperationParams.push_back(b.value());
    }
    
    Tensor multNode(mul, {args[0], W});
    if (params.bias) {
        pipeline.push_back(std::move(multNode));
        result = Tensor(sum, {pipeline[0], b.value()});
    } else {
        result.emplace(std::move(multNode));
    }
}

ReLULayer::ReLULayer(const std::vector<TensorRef>& args) {
    result = Tensor(relu, {args[0]});
}

DataLayer::DataLayer(const Shape& shape, const std::vector<float>& values) {
    result = Tensor(Blob::constBlob(shape, values.data()));
}

DataLayer::DataLayer(const Shape& shape, size_t batch_size) {
    std::vector<size_t> dims = shape.getDims();
    dims[0] = batch_size;
    result = Tensor(Blob::constRandomBlob(Shape(std::move(dims)), nullptr));
}

MSELoss::MSELoss(const std::vector<TensorRef>& args) : mean({0, 1, 2, 3}) {
    pipeline.reserve(2);

    Tensor diff(sub, {args[0], args[1]});
    pipeline.push_back(std::move(diff));

    Tensor square(sqr, {pipeline[0]});
    pipeline.push_back(std::move(square));

    result = Tensor(mean, {pipeline[1]});
}

MultLayer::MultLayer(const std::vector<TensorRef>& args) {
    result = Tensor(mult, {args[0].get(), args[1].get()});
}
