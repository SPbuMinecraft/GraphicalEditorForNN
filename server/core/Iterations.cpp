#include "Iterations.h"

void opp(
    bool* axis, 
    std::size_t* size, 
    const std::function <void (std::size_t, std::size_t, std::size_t, std::size_t)>& f, 
    std::vector<std::size_t> indexes
) {
    if (indexes.size() == 4) {
        f(indexes[0], indexes[1], indexes[2], indexes[3]);
        return;
    }
    int currentAxis = indexes.size();
    if (axis[currentAxis]) {
        for (int i = 0; i < size[currentAxis]; ++i) {
            indexes.push_back(i);
            opp(axis, size, f, indexes);
            indexes.pop_back();
        }
    }
    else {
        indexes.push_back(0);
        opp(axis, size, f, indexes);
    }
}
