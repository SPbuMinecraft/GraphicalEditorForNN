#pragma once

#include <map>
#include <unordered_map>
#include <optional>
#include <string>
#include <vector>

#include "RandomInit.h"
#include "Tensor.h"
#include "Parser.h"


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

class OutputLayer: public Layer {
public:
    OpId id;
    OutputLayer(const std::vector<TensorRef>& args);
};

class LinearLayer: public Layer {
public:
    Multiply mul;
    BiasSum sum;
    Tensor W;
    Tensor b;
    
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
    MSELoss(const std::vector<TensorRef>& args, RandomObject* randomInit = nullptr);
};

class MultLayer: public Layer {
public:
    Multiply mult;
    MultLayer(
        const std::vector<TensorRef>& args, RandomObject* randomInit = nullptr);
};