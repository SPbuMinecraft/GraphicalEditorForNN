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

// int main() {
//     Allocator::startVirtualMode();
//     {
//     LinearLayerParameters params1{2ull, 2ull, true};
//     LinearLayerParameters params2{2ull, 1ull, true};
//     Conv2DLayerParameters paramsConv{7, 3, 1};

//     auto inputNode = Tensor(Blob::constBlob(Shape {{4, 2}}, input));

//     auto trueNode = Tensor(Blob::constBlob(Shape {{4, 1}}, output));

//     RandomObject initObject(0, 1, 17);
//     OptimizerBase SGD = OptimizerBase(0.1);
//     LinearLayer layer1 {params1, {inputNode}, &initObject};
//     SGD.append(layer1.layerOperationParams);

//     TensorRef res = layer1.result.value();
//     ReLULayer reluLayer1  {{res}};

//     res = reluLayer1.result.value();
//     LinearLayer layer2 {params2, {res}, &initObject};
//     res = layer2.result.value();
//     SGD.append(layer2.layerOperationParams);

//     MSELoss mseLoss {{res, trueNode}};

//     auto &lastNode = mseLoss.result.value();
//     lastNode.forward();
//     lastNode.gradient = Blob::ones(Shape {{1}});
//     lastNode.backward();
//     Allocator::endSession();
//     lastNode.clear();
//     Allocator::endVirtualMode();

//     for (int j = 0; j < 500; ++j) {
//         auto &result = lastNode.forward();
//         lastNode.gradient = Blob::ones(Shape {{1}});
//         printf("%d: %f\n", j, result(0, 0));
//         lastNode.backward();
//         SGD.step();
//         Allocator::endSession();
//         lastNode.clear();
//     }
//     auto &result2 = res.get().forward();
//     print(result2);
//     Allocator::endSession();
//     }

//     Allocator::end();
//     return 0;
// }

// int main() {
//     Allocator::startVirtualMode();
//     {
//     Conv2DLayerParameters paramsConv{3, 3, 1};

//     auto inputNode = Tensor(Blob::constBlob(Shape {{3, 5, 5}}, input));

//     auto trueNode = Tensor(Blob::constBlob(Shape {{3, 1}}, output));

//     RandomObject initObject(0, 1, 17);
//     OptimizerBase SGD = OptimizerBase(0.1);
//     Conv2DLayer convL = {paramsConv, {inputNode}, &initObject};
//     SGD.append(convL.layerOperationParams);

//     TensorRef res = convL.result.value();
//     res.get().forward();
//     print(res.get().output);

//     print(convL.kernel.output);

//     res.get().gradient = Blob::ones(Shape {{1, 1, 5, 5}});
//     res.get().backward();

//     print(convL.kernel.gradient);

//     print(inputNode.gradient);
//     Allocator::endSession();
//     }

//     Allocator::end();
//     return 0;
// }

// int main() {
//     Allocator::startVirtualMode();
//     {
//     AxisParameters paramsConv{{1, 2, 3}};

//     auto inputNode = Tensor(Blob::constBlob(Shape {{3, 5, 5}}, input));

//     auto trueNode = Tensor(Blob::constBlob(Shape {{3, 1}}, output));
//     OptimizerBase SGD = OptimizerBase(0.1);
//     VarLayer var = {paramsConv, {inputNode}};

//     TensorRef res = var.result.value();
//     res.get().forward();
//     print(res.get().output);

//     res.get().gradient = Blob::ones(res.get().output.value().shape);
//     res.get().backward();

//     // print(convL.kernel.gradient);

//     print(inputNode.gradient);
//     Allocator::endSession();
//     }

//     Allocator::end();
//     return 0;
// }

int main() {
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
    return 0;
}