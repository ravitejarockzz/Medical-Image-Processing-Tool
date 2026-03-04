#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <string>

class IFilter {
public:
    // Apply filter to input image with dynamic parameters
    virtual cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) = 0;

    // Unique operation name (must match frontend operation string)
    virtual std::string name() const = 0;

    virtual ~IFilter() = default;
};