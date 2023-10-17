#include <unordered_map>
#include <string>
#include <vector>

#include "Allocator.h"
#include "Tensor.h"
#include "RandomInit.h"
#include "Layer.h"
#include "Optimizer.h"

using namespace std;

float input[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};

float output[] = { 0, 1, 1, 0 };

void print(const Blob &a) {
    cout << a << endl;
}

static const Multiply multOperation;
static const BiasSum sumOperation;
static const ReLU reluOperation;

int main() {
    Allocator::start({
        {{4, 2}, 40},
        {{2, 1}, 10},
        {{4, 1}, 40},
        {{1, 1}, 30},
        {{1, 4}, 10},
        {{2, 4}, 10},
        {{2, 2}, 20},
        {{1, 2}, 20}
    });
    {

    std::unordered_map<std::string, float> layer1Params = {};
    layer1Params["h"] = 2;
    layer1Params["w"] = 2;
    // layer1Params["w"] = 1;

    std::unordered_map<std::string, float> layer2Params = {};
    layer2Params["h"] = 2;
    layer2Params["w"] = 1;

    auto inputNode = Tensor({4, 2, input});

    auto trueNode = Tensor({4, 1, output});

    RandomObject initObject(0, 1, 42);
    OptimizerBase SGD = OptimizerBase(0.1);
    // LinearLayer layer1 {layer1Params, {inputNode}};
    LinearLayer layer1 {layer1Params, {inputNode}, &initObject};
    SGD.append(layer1.layerOperationParams);

    TensorRef res = layer1.result.value();
    ReLULayer reluLayer1  {{}, {res}};

    res = reluLayer1.result.value();
    // LinearLayer layer2 {layer2Params, {*res}};
    LinearLayer layer2 {layer2Params, {res}, &initObject};
    res = layer2.result.value();
    SGD.append(layer2.layerOperationParams);

    MSELoss mseLoss {{}, {res, trueNode}};

    auto &lastNode = mseLoss.result.value();

    // Blob grad_1 {1, 1, (float) 1};
    // lastNode.gradient = grad_1;

    lastNode.clear();
    for (int j = 0; j < 1000; ++j) {
        auto &result = lastNode.forward();
        printf("%d: %f\n", j, result[0][0]);
        // lastNode.gradient = result;
        lastNode.gradient.value()[0][0] = 1;
        lastNode.backward();
        SGD.step();
        Allocator::endSession();
        lastNode.clear();
    }
    auto &result2 = res.get().forward();
    print(result2);

    }

    Allocator::end();
    return 0;
}
