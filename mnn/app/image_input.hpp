#pragma once
#include <array>
#include <string>
#include <vector>
std::vector<float> image_input(const std::string& path, int width, int height,
    int resize_short, const std::array<float,3>& mean, const std::array<float,3>& stddev);
