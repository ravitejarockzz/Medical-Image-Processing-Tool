#pragma once

#include "core/IFilter.h"

class GaussianBlurCPU : public IFilter {
public:
    GaussianBlurCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "gaussian_blur";
    }

    ~GaussianBlurCPU() override = default;
};