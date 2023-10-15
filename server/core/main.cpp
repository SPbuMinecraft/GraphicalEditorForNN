#include <unordered_map>
#include <string>
#include <vector>

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

void print(Blob a) {
    cout << a << endl;
}

static const Multiply multOperation;
static const BiasSum sumOperation;
static const ReLU reluOperation;

int main() {
    LinearLayerParameters params1{2ull, 2ull, true};
    LinearLayerParameters params2{2ull, 1ull, true};

    Blob x {4, 2, input};
    auto inputNode = Tensor(x);
    Blob out {4, 1, output};

    auto trueNode = Tensor(out);

    RandomObject initObject(0, 1, 42);
    OptimizerBase SGD = OptimizerBase(0.1);
    // LinearLayer layer1 {layer1Params, {inputNode}};
    LinearLayer layer1 {params1, {inputNode}, &initObject};
    SGD.append(layer1.layerOperationParams);

    TensorRef res = layer1.result.value();
    ReLULayer reluLayer1  {{res}};

    res = reluLayer1.result.value();
    // LinearLayer layer2 {layer2Params, {*res}};
    LinearLayer layer2 {params2, {res}, &initObject};
    res = layer2.result.value();
    SGD.append(layer2.layerOperationParams);

    MSELoss mseLoss {{res, trueNode}};

    auto lastNode = mseLoss.result.value();
    
    // Blob grad_1 {1, 1, (float) 1};
    // lastNode.gradient = grad_1;

    Blob result {1, 1};


    for (int j = 0; j < 100; ++j) {
        result = lastNode.forward();
        printf("%d: %f\n", j, result[0][0]);
        // lastNode.gradient = result;
        lastNode.gradient.value()[0][0] = 1;
        lastNode.backward();
        SGD.step();
        lastNode.clear();
    }
    Blob result2 = res.get().forward();
    print(result2);
    return 0;
}
