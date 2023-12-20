#pragma once

#include <optional>
#include <vector>

#include "RandomInit.h"
#include "Tensor.h"
#include "Parameters.h"


class Layer {
protected:
    std::vector<Tensor> pipeline;
public:
    std::optional<Tensor> result; 
    std::vector<TensorRef> layerOperationParams;
};

class Data2dLayer: public Layer {
public:
    size_t width;
    Data2dLayer(const Data2dLayerParameters& params, const std::vector<float>& values);
};

class LinearLayer: public Layer {
public:
    Multiply mul;
    BiasSum sum;
    Tensor W;
    std::optional<Tensor> b;
    
    LinearLayer(
        const LinearLayerParameters& params,
        const std::vector<TensorRef>& args, RandomObject* randomInit = nullptr
    );
};

class ReLULayer: public Layer {
public:
    ReLU relu;
    ReLULayer(const std::vector<TensorRef>& args);
};

class MSELoss: public Layer {
public:
    Substract sub;
    Square sqr;
    Mean mean;
    MSELoss(const std::vector<TensorRef>& args);
};

class MultLayer: public Layer {
public:
    Multiply mult;
    MultLayer(const std::vector<TensorRef>& args);
};

class Conv2DLayer: public Layer {
public:
    Tensor kernel;
    Conv2D conv;
    Conv2DLayer(const Conv2DLayerParameters& params,
        const std::vector<TensorRef>& args, RandomObject* randomInit = nullptr);
};

class VarLayer: public Layer {
public:
    Mean mean;
    Mean meanMinusOne;
    Fill fill;
    Substract sub;
    Square sqr;
    SumAxis sum;
    VarLayer(const AxisParameters& params,
        const std::vector<TensorRef>& args);
};

class LayerNorm: public Layer {
public:
    VarLayer varLayer;
    Mean mean;
    Fill fill;
    Substract sub;
    EPS eps;
    Root root;
    Divide div;
    LayerNorm(const AxisParameters& params,
        const std::vector<TensorRef>& args);
};

class SoftMax: public Layer {
public:
    Exp exp;
    SumAxis sum;
    Fill fill;
    Divide div;
    SoftMax(const AxisParameters& params,
        const std::vector<TensorRef>& args);
};

class EntropyLoss: public Layer {
public:
    SoftMax softmax;
    Mean mean;
    Entropy entropy;
    EntropyLoss(const CrossEntropyLossParameters& params, 
        const std::vector<TensorRef>& args);
};

class MaxPool: public Layer {
public:
    MaxPoolOp maxPool;
    MaxPool(const std::vector<TensorRef>& args);
};
