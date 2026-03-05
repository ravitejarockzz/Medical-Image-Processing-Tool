#pragma once

#include "core/IFilter.h"

class MedianBlurCUDA : public IFilter {
public:
    MedianBlurCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "median_blur"; 
    }

    ~MedianBlurCUDA() override = default;
};