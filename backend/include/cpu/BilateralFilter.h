#pragma once

#include "core/IFilter.h"

class BilateralFilterCPU : public IFilter {
public:
    BilateralFilterCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "bilateral_filter";
    }

    ~BilateralFilterCPU() override = default;
};