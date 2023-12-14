#include <vector>
#include <utility>
#include <unordered_set>

#include "Shape.h"

struct Stretch {

    std::vector<short> axisForStretch = {};

    Stretch() {};
    Stretch(std::vector<short> axisForStretch): axisForStretch(axisForStretch) {};
    Stretch(const Stretch& other): axisForStretch(other.axisForStretch) {};

    static std::pair<bool, Stretch> canStretch(Shape a, Shape b);

    Stretch& operator=(const Stretch& other);
};
