#include <cassert>

#include "Iterations.h"

void map(
    std::vector<std::size_t> size, 
    const std::function <void (std::size_t, std::size_t, std::size_t, std::size_t)>& f, 
    std::vector<std::size_t> indices
) {
    if (indices.size() == 4) {
        f(indices[0], indices[1], indices[2], indices[3]);
        return;
    }
    int currentAxis = indices.size();
    for (int i = 0; i < size[currentAxis]; ++i) {
        indices.push_back(i);
        map(size, f, indices);
        indices.pop_back();
    }
}

std::vector<bool> fillAxis(std::vector<short> axisForStretch) {
    std::vector<bool> axis = {false, false, false, false};
    for (auto dim: axisForStretch) {
        assert(dim >= 0 && dim < 4);
        axis[dim] = true;
    }
    return axis;
}

std::vector<std::size_t> fillSize(std::vector<bool> axisForIter, Shape shape) {
    std::vector<std::size_t> size = {1, 1, 1, 1};
    for (int dim = 0; dim < 4; dim++) {
        size[dim] = axisForIter[dim] ? shape[dim] : 1;
    }
    return size;
}
