#pragma once

#include "core/IFilter.h"

class FASTDetectorCUDA : public IFilter {
public:
    FASTDetectorCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "FAST";
    }

    ~FASTDetectorCUDA() override = default;
};