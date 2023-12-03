#pragma once

#include <functional>

#include "Shape.h"

/// To map a function to all indeces
/// - Parameter size: iterCount for every dimension
/// - Parameter f: a function that is called on all fours
/// - Warning: never pass indexes to a function
void map(
    std::vector<std::size_t> size,
    const std::function <void (std::size_t, std::size_t, std::size_t, std::size_t)>& f, 
    std::vector<std::size_t> indices = {}
);


/// The function tells us which dimensions we should iterate over
/// - Parameter axisForStretch: axis for stretching
/// - Returns: mask: which dimensions should be iterated on
std::vector<bool> fillAxis(std::vector<short> axisForStretch);

/// It tells us how many times we have to iterate over the dimensions from the mask
/// - Parameter axisForIter: mask of dimensions to iterate
/// - Parameter shape: shape of blob for sizes
/// - Returns: sizes for iterations from shape and mask
std::vector<std::size_t> fillSize(std::vector<bool> axisForIter, Shape shape);