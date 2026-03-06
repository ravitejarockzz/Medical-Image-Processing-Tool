#pragma once

#include "core/IFilter.h"

class SIFTDetectorCUDA : public IFilter {
public:
    SIFTDetectorCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "sift";
    }

    ~SIFTDetectorCUDA() override = default;
};