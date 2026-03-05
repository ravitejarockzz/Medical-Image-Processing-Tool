#pragma once

#include "core/IFilter.h"

class BilateralFilterCUDA : public IFilter {
public:
    BilateralFilterCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "bilateral_filter";
    }

    ~BilateralFilterCUDA() override = default;
};