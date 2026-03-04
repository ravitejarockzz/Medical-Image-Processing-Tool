#pragma once

#include "core/IFilter.h"

class HarrisCornerCUDA : public IFilter {
public:
    HarrisCornerCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "harris_corner"; // This is the string your frontend will send
    }

    ~HarrisCornerCUDA() override = default;
};