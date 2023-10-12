#include "Tensor.h"
#include <crow_all.h>

using namespace std;

float input[] = {
    0, 0,
    0, 1,
    1, 0,
    1, 1
};

float output[] = { 1, 1, 1, 0 };

static const Multiply multOperation;
static const Sum sumOperation;
static const Loss lossOperation;

int example() {
    Blob x {4, 2, input};
    Blob w {2, 1};
    Blob b {1, 1};
    Blob out {4, 1, output};

    auto inputNode = Tensor(x);
    auto wNode = Tensor(w);
    auto bNode = Tensor(b);

    auto multNode = Tensor(multOperation, {inputNode, wNode});
    auto sumNode = Tensor(sumOperation, {multNode, bNode});

    auto trueNode = Tensor(out);
    auto lossNode = Tensor(lossOperation, {sumNode, trueNode});

    Blob result = lossNode.forward();
    lossNode.clear();

    result = lossNode.forward();

    for (int j = 0; j < 100; ++j) {
        result = lossNode.forward();
        float sum = 0;
        for (int i = 0; i < 4; ++i) {
            sum += result[i][0];
        }
        float mean = sum / 4;
        printf("%d: %f\n", j, mean);

        lossNode.backward();
        *wNode.output -= 0.001 * *wNode.gradient;
        *bNode.output -= 0.001 * *bNode.gradient;

        lossNode.clear();
    }

//    Blob w {2, 4}, b {1, 4};
//    Blob loss;
//    Tensor x;
//    Tensor w, b;
//    Tensor loss;

    return 0;
}
