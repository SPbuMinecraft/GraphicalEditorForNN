#include <unordered_map>
#include <functional>
#include <cassert>
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
    std::vector<size_t> new_dims(dims.end() - shape.dimsCount, dims.end());
    new_dims.insert(new_dims.begin(), batch_size);
    Shape batch_shape(new_dims);
    result = Tensor(Blob::constRandomBlob(batch_shape, nullptr));
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

Conv2DLayer::Conv2DLayer(const Conv2DLayerParameters& params,
                         const std::vector<TensorRef>& args,
                         RandomObject* randomInit)
    : kernel(Blob::constRandomBlob(
        Shape {{params.outChannels, params.inChannels, params.kernelSize, params.kernelSize}}, 
        randomInit)
    ) {

    layerOperationParams.push_back(kernel);

    result = Tensor(conv, {args[0], kernel});
}

VarLayer::VarLayer(const AxisParameters& params,
                   const std::vector<TensorRef>& args)
    : mean(params.axis), meanMinusOne(params.axis, true), sum(params.axis) {
    pipeline.reserve(5);
    TensorRef tensor = args[0];
    Tensor meanForVar(mean, {tensor});
    pipeline.push_back(std::move(meanForVar));

    Tensor fillForVar(fill, {tensor, pipeline[0]});
    pipeline.push_back(std::move(fillForVar));

    Tensor diff(sub, {tensor, pipeline[1]});
    pipeline.push_back(std::move(diff));

    Tensor square(sqr, {pipeline[2]});
    pipeline.push_back(std::move(square));

    result = Tensor(meanMinusOne, {pipeline[3]});     
}

MeanLayer::MeanLayer(const AxisParameters& params,
                     const std::vector<TensorRef>& args)
    // : mean(params.axis) {
    : mean({2}) {
    assert(args.size() == 1);
    result = Tensor(mean, {args[0]});     
}

LayerNorm::LayerNorm(const AxisParameters& params,
                     const std::vector<TensorRef>& args)
    : varLayer(params, args), mean(params.axis) {
    pipeline.reserve(6);
    TensorRef tensor = args[0];

    Tensor mean_(mean, {tensor});
    pipeline.push_back(std::move(mean_));

    Tensor fill_(fill, {tensor, pipeline[0]});
    pipeline.push_back(std::move(fill_));

    Tensor diff(sub, {tensor, pipeline[1]});
    pipeline.push_back(std::move(diff));

    Tensor eps_(eps, {varLayer.result.value()});
    pipeline.push_back(std::move(eps_));

    Tensor root_(root, {pipeline[3]});
    pipeline.push_back(std::move(root_));

    Tensor _fill(fill, {tensor, pipeline[4]});
    pipeline.push_back(std::move(_fill));

    result = Tensor(div, {pipeline[2], pipeline[5]});
}

SoftMax::SoftMax(const AxisParameters& params,
                   const std::vector<TensorRef>& args)
    : sum(params.axis) {
    assert(args.size() == 1);
    pipeline.reserve(3);
    TensorRef tensor = args[0];

    Tensor exp_(exp, {tensor});
    pipeline.push_back(std::move(exp_));

    Tensor sum_(sum, {pipeline[0]});
    pipeline.push_back(std::move(sum_));

    Tensor fill_(fill, {pipeline[0], pipeline[1]});
    pipeline.push_back(std::move(fill_));

    result = Tensor(div, {pipeline[0], pipeline[2]});
}

EntropyLoss::EntropyLoss(const CrossEntropyLossParameters& params,
                        const std::vector<TensorRef>& args)
    : softmax({{3}}, {args[0]}), mean({0, 1, 2, 3}), entropy(params.classCount) {
    assert(args.size() == 2);
    pipeline.reserve(1);

    Tensor entropy_(entropy, {softmax.result.value(), args[1]});
    pipeline.push_back(std::move(entropy_));

    result = Tensor(mean, {pipeline[0]});
}

MaxPool::MaxPool(const std::vector<TensorRef>& args) {
    assert(args.size() == 1);
    result = Tensor(maxPool, {args[0]});
}
