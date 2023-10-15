#pragma once

#include <unordered_map>
#include <optional>
#include <string>
#include <vector>

#include "RandomInit.h"
#include "Tensor.h"

class Layer {
protected:
    std::vector<Tensor> pipeline;
public:
    std::optional<Tensor> result; 
    std::vector<TensorRef> layerOperationParams;
};

class LinearLayer: public Layer {
public:
    Multiply mul;
    BiasSum sum;
    Tensor W;
    Tensor b;
    
    LinearLayer(
        std::unordered_map<std::string, float> params, 
        const std::vector<const TensorRef>& args,
        RandomObject* const randomInit = nullptr
    );
};

class ReLULayer: public Layer {
public:
    ReLU relu;
    ReLULayer(
        std::unordered_map<std::string, float> params, 
        const std::vector<const TensorRef>& args,
        RandomObject* const randomInit = nullptr
    );
};

class MSELoss: public Layer {
public:
    Substract sub;
    Square sqr;
    Mean mean;
    MSELoss(
        std::unordered_map<std::string, float> params, 
        const std::vector<const TensorRef>& args,
        RandomObject* const randomInit = nullptr
    );
};

class MultLayer: public Layer {
public:
    Multiply mult;
    MultLayer(
        std::unordered_map<std::string, float> params, 
        const std::vector<const TensorRef>& args,
        RandomObject* const randomInit = nullptr
    );
};