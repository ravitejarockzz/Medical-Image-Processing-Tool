#pragma once

#include "core/IFilter.h"

class MedianBlurCPU : public IFilter {
public:
    MedianBlurCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "median_blur"; 
    }

    ~MedianBlurCPU() override = default;
};