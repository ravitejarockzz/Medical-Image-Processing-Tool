#pragma once

#include "core/IFilter.h"

class OtsuThresholdCPU : public IFilter {
public:
    OtsuThresholdCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "otsu_threshold";
    }

    ~OtsuThresholdCPU() override = default;
};