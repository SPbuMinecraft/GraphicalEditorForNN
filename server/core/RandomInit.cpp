#include "RandomInit.h"

void initGenerator(PRNG& generator, size_t seed) {
    generator.engine.seed(seed);
}

RandomObject::RandomObject(
    float mean, 
    float std,
    size_t seed
) {
    PRNG generator;
    this->generator = generator;
    initGenerator(this->generator, seed);
    std::normal_distribution<float> distribution(mean, std);
    this->distribution = distribution;
}

float RandomObject::randomNumber() {
    return distribution(generator.engine);
}

void RandomObject::simpleInit(float* data, size_t size) {
    distribution.reset();
    for (int i = 0; i < size; i++) data[i] = randomNumber();
}