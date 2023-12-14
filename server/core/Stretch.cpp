#include <algorithm>

#include "Stretch.h"

std::pair<bool, Stretch> Stretch::canStretch(Shape a, Shape b) {
    std::vector<short> axisForStretch;
    Shape bigShape = a;
    Shape smallShape = b;
    bool thereIsStretch = false;
    bool canStretch = true;
    if (bigShape.dimsCount < smallShape.dimsCount) {
        std::swap(bigShape, smallShape);
    }
    for (int i = 3; i > - 1; i--) {
        if (bigShape[i] == smallShape[i] && !thereIsStretch) {
            continue;
        }
        else {
            if (bigShape[i] > smallShape[i] && smallShape[i] == 1) {
                axisForStretch.push_back(i);
                thereIsStretch = true;
                continue;
            }
            else if (smallShape[i] == 1 && bigShape[i] == 1) {
                axisForStretch.push_back(i);
                continue;
            }
            else {
                canStretch = false;
                break;
            }
        }
    }

    return {canStretch, Stretch(axisForStretch)};
}

Stretch& Stretch::operator=(const Stretch& other) {
    this->axisForStretch = other.axisForStretch;
    return *this;
}
