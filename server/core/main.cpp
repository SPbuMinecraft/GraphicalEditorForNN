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
    LinearLayerParameters params1{2ull, 2ull, true};
    LinearLayerParameters params2{2ull, 1ull, true};
    Allocator::startVirtualMode();
    {

    auto inputNode = Tensor(Blob::constBlob(4, 2, input));

    auto trueNode = Tensor(Blob::constBlob(4, 1, output));

    RandomObject initObject(0, 1, 42);
    OptimizerBase SGD = OptimizerBase(0.1);

    LinearLayer layer1 {params1, {inputNode}, &initObject};
    SGD.append(layer1.layerOperationParams);

    TensorRef res = layer1.result.value();
    ReLULayer reluLayer1  {{res}};

    res = reluLayer1.result.value();
    LinearLayer layer2 {params2, {res}, &initObject};

    res = layer2.result.value();
    SGD.append(layer2.layerOperationParams);

    MSELoss mseLoss {{res, trueNode}};

    auto &lastNode = mseLoss.result.value();
    lastNode.forward();
    lastNode.backward(true);
    lastNode.clear();
    Allocator::endVirtualMode();

    for (int j = 0; j < 200; ++j) {
        auto &result = lastNode.forward();
        printf("%d: %f\n", j, result[0][0]);
        lastNode.backward(true);
        SGD.step();
        lastNode.clear();
    }
    auto &result2 = res.get().forward();
    print(result2);
    Allocator::endSession();
    }

    Allocator::end();
    return 0;
}
