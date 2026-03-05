#pragma once

#include "core/IFilter.h"

class OtsuThresholdCUDA : public IFilter {
public:
    OtsuThresholdCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "otsu_threshold";
    }

    ~OtsuThresholdCUDA() override = default;
};