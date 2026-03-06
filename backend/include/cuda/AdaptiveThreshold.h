#pragma once

#include "core/IFilter.h"

class AdaptiveThresholdCUDA : public IFilter {
public:
    AdaptiveThresholdCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "adaptive_threshold";
    }

    ~AdaptiveThresholdCUDA() override = default;
};