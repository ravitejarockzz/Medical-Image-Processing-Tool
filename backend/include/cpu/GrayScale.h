#pragma once

#include "core/IFilter.h"

class GrayScaleCPU : public IFilter {
public:
    GrayScaleCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "gray_scale";
    }

    ~GrayScaleCPU() override = default;
};