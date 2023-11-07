#pragma once

#include <functional>

void opp(
    bool* axis, 
    std::size_t* size, 
    const std::function <void (std::size_t, std::size_t, std::size_t, std::size_t)>& f, 
    std::vector<std::size_t> indexes = {}
);