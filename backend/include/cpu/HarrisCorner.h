#pragma once

#include "core/IFilter.h"

class HarrisCornerCPU : public IFilter {
public:
    HarrisCornerCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "harris_corner"; // This is the string your frontend will send
    }

    ~HarrisCornerCPU() override = default;
};