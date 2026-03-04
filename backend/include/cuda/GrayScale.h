#pragma once

#include "core/IFilter.h"

class GrayScaleCUDA : public IFilter {
public:
    GrayScaleCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "gray_scale";
    }

    ~GrayScaleCUDA() override = default;
};