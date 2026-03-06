#pragma once

#include "core/IFilter.h"

class SIFTDetectorCPU : public IFilter {
public:
    SIFTDetectorCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "sift";
    }

    ~SIFTDetectorCPU() override = default;
};