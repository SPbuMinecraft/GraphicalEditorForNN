#include <unordered_map>
#include <string>
#include <vector>

#include "Allocator.h"
#include "Tensor.h"
#include "RandomInit.h"
#include "Layer.h"
#include "Optimizer.h"

using namespace std;

// float input[] = {
//     0, 0,
//     0, 1,
//     1, 0,
//     1, 1
// };

// float output[] = { 0, 1, 1, 0 };

float input[] = {
    0.f, 0.f, 1.f, 1.f, 1.f,
    0.f, 0.f, 0.f, 1.f, 1.f,
    1.f, 0.f, 0.f, 1.f, 0.f, 
    0.f, 1.f, 0.f, 0.f, 1.f, 
    0.f, 0.f, 1.f, 0.f, 1.f,

    0.f, 0.f, 1.f, 0.f, 1.f, 
    1.f, 0.f, 0.f, 1.f, 0.f, 
    1.f, 1.f, 1.f, 0.f, 0.f, 
    0.f, 0.f, 1.f, 1.f, 0.f, 
    0.f, 0.f, 0.f, 1.f, 1.f, 
    
    0.f, 0.f, 0.f, 0.f, 1.f, 
    1.f, 0.f, 0.f, 0.f, 0.f,
    1.f, 1.f, 0.f, 0.f, 0.f, 
    0.f, 1.f, 1.f, 0.f, 0.f, 
    0.f, 0.f, 1.f, 1.f, 1.f
};

float output[] = { 0, 1, 1};

void print(const Blob &a) {
    cout << a << endl;
}

void print(std::optional<Blob> &a) {
    if (a.has_value()) {
        cout << a.value() << endl;
    }
}

static const Multiply multOperation;
static const BiasSum sumOperation;
static const ReLU reluOperation;

thread_local int a = 0;

void testNN() {
    Allocator::startVirtualMode();
    {
    LinearLayerParameters params1{2ull, 2ull, true};
    LinearLayerParameters params2{2ull, 1ull, true};
    // Conv2DLayerParameters paramsConv{7, 3, 1};

    auto inputNode = Tensor(Blob::constBlob(Shape {{4, 2}}, input));

    auto trueNode = Tensor(Blob::constBlob(Shape {{4, 1}}, output));

    RandomObject initObject(0, 1, 17);
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
    lastNode.gradient = Blob::ones(Shape {{1}});
    lastNode.backward();
    Allocator::endSession();
    lastNode.clear();
    Allocator::endVirtualMode();

    for (int j = 0; j < 500; ++j) {
        lastNode.forward();
        auto &result = lastNode.output.value();
        lastNode.gradient = Blob::ones(Shape {{1}});
        printf("%d: %f\n", j, result(0, 0));
        lastNode.backward();
        SGD.step();
        Allocator::endSession();
        lastNode.clear();
    }
    res.get().forward();
    auto &result2 = res.get().output.value();
    print(result2);
    Allocator::endSession();
    }

    Allocator::end();
}

void testConv() {
    Allocator::startVirtualMode();
    {
    Conv2DLayerParameters paramsConv{3, 3, 1};

    auto inputNode = Tensor(Blob::constBlob(Shape {{3, 5, 5}}, input));

    auto trueNode = Tensor(Blob::constBlob(Shape {{3, 1}}, output));

    RandomObject initObject(0, 1, 17);
    OptimizerBase SGD = OptimizerBase(0.1);
    Conv2DLayer convL = {paramsConv, {inputNode}, &initObject};
    SGD.append(convL.layerOperationParams);

    TensorRef res = convL.result.value();
    res.get().forward();
    print(res.get().output);

    print(convL.kernel.output);

    res.get().gradient = Blob::ones(Shape {{1, 1, 5, 5}});
    res.get().backward();

    print(convL.kernel.gradient);

    print(inputNode.gradient);
    Allocator::endSession();
    }

    Allocator::end();
}

void testVar() {
    Allocator::startVirtualMode();
    {
    AxisParameters paramsConv{{1, 2, 3}};

    auto inputNode = Tensor(Blob::constBlob(Shape {{3, 5, 5}}, input));

    auto trueNode = Tensor(Blob::constBlob(Shape {{3, 1}}, output));
    OptimizerBase SGD = OptimizerBase(0.1);
    VarLayer var = {paramsConv, {inputNode}};

    TensorRef res = var.result.value();
    res.get().forward();
    print(res.get().output);

    res.get().gradient = Blob::ones(res.get().output.value().shape);
    res.get().backward();

    // print(convL.kernel.gradient);

    print(inputNode.gradient);
    Allocator::endSession();
    }

    Allocator::end();
}

void testNorm() {
    Allocator::startVirtualMode();
    {
    AxisParameters paramsConv{{2, 3}};

    auto inputNode = Tensor(Blob::constBlob(Shape {{3, 5, 5}}, input));

    auto trueNode = Tensor(Blob::constBlob(Shape {{3, 1}}, output));
    OptimizerBase SGD = OptimizerBase(0.1);
    LayerNorm norm = {paramsConv, {inputNode}};

    TensorRef res = norm.result.value();
    res.get().forward();
    print(res.get().output);

    // res.get().gradient = Blob::ones(res.get().output.value().shape);
    // res.get().backward();

    res.get().gradient = Blob::fill(res.get().output.value().shape, 100000000.f);
    res.get().backward();

    // print(convL.kernel.gradient);

    print(inputNode.gradient);
    Allocator::endSession();
    }

    Allocator::end();
}

float inputL[] = {
    10.f, 30.f,
    25.f, 25.f,
    1.f, 6.f
};

float outputL[] = { 0, 0, 1};

void testEntropyLoss() {
    Allocator::startVirtualMode();
    {
    CrossEntropyLossParameters params{2};

    auto inputNode = Tensor(Blob::constBlob(Shape {{3, 1, 1, 2}}, inputL));

    auto trueNode = Tensor(Blob::constBlob(Shape {{3, 1, 1, 1}}, outputL));
    OptimizerBase SGD = OptimizerBase(0.1);
    EntropyLoss loss = {params, {inputNode, trueNode}};

    TensorRef res = loss.result.value();
    res.get().forward();
    print(res.get().output);

    // res.get().gradient = Blob::ones(res.get().output.value().shape);
    // res.get().backward();

    res.get().gradient = Blob::fill(res.get().output.value().shape, 1.f);
    res.get().backward();

    // print(convL.kernel.gradient);

    print(inputNode.gradient);
    Allocator::endSession();
    }

    Allocator::end();
}

float inputM[] = {
    10.f, 30.f, 3.f, 4.f, 
    10.f, 30.f, 3.f, 4.f, 
    10.f, 30.f, 3.f, 4.f, 
    10.f, 30.f, 3.f, 4.f,

    25.f, 25.f, 5.f, 6.f, 
    10.f, 30.f, 3.f, 4.f, 
    10.f, 30.f, 3.f, 4.f, 
    40.f, 30.f, 3.f, 4.f,

    1.f, 6.f, 7.f, 8.f, 
    27.f, 25.f, 5.f, 6.f, 
    10.f, 30.f, 3.f, 4.f, 
    10.f, 30.f, 3.f, -1.f
};

float outputM[] = { 0, 0, 1};

void testMaxPoolLoss() {
    Allocator::startVirtualMode();
    {

    auto inputNode = Tensor(Blob::constBlob(Shape {{3, 1, 4, 4}}, inputM));

    auto trueNode = Tensor(Blob::constBlob(Shape {{3, 1, 1, 1}}, outputM));
    OptimizerBase SGD = OptimizerBase(0.1);
    MaxPool max = {{inputNode}};

    TensorRef res = max.result.value();
    res.get().forward();
    print(res.get().output);

    // res.get().gradient = Blob::ones(res.get().output.value().shape);
    // res.get().backward();

    res.get().gradient = Blob::fill(res.get().output.value().shape, 1.f);
    res.get().backward();

    // print(convL.kernel.gradient);

    print(inputNode.gradient);
    Allocator::endSession();
    }

    Allocator::end();
}

int main() {
    testMaxPoolLoss();
    return 0;
}