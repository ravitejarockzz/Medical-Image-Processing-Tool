#pragma once

#include "core/IFilter.h"

class AdaptiveThresholdCPU : public IFilter {
public:
    AdaptiveThresholdCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "adaptive_threshold";
    }

    ~AdaptiveThresholdCPU() override = default;
};