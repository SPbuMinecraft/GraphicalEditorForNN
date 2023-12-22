#include <cassert>

#include "Iterations.h"

void map(
    const size_t size[4],
    const std::function <void (std::size_t, std::size_t, std::size_t, std::size_t)>& f
) {
    for (int k = 0; k < size[0]; ++k)
        for (int l = 0; l < size[1]; ++l)
            for (int i = 0; i < size[2]; ++i)
                for (int j = 0; j < size[3]; ++j)
                    f(k, l, i, j);
//    if (indices.size() == 4) {
//        f(indices[0], indices[1], indices[2], indices[3]);
//        return;
//    }
//    int currentAxis = indices.size();
//    for (int i = 0; i < size[currentAxis]; ++i) {
//        indices.push_back(i);
//        map(size, f, indices);
//        indices.pop_back();
//    }
}

std::vector<bool> fillAxis(std::vector<short> axisForStretch) {
    std::vector<bool> axis = {false, false, false, false};
    for (auto dim: axisForStretch) {
        assert(dim >= 0 && dim < 4);
        axis[dim] = true;
    }
    return axis;
}

void fillSize(std::vector<bool> axisForIter, Shape shape, size_t result[4]) {
    for (int dim = 0; dim < 4; dim++)
        result[dim] = axisForIter[dim] ? shape[dim] : 1;
}
