#pragma once

#include <random>
#include <cmath>

struct PRNG {
    std::mt19937 engine;
};

void initGenerator(PRNG& generator, size_t seed);

class RandomObject {
public:
    int seed;
    PRNG generator;
    std::normal_distribution<float> distribution;
    RandomObject(float mean, float std, size_t seed);

    void simpleInit(float* data, size_t size);
    float randomNumber();
};
