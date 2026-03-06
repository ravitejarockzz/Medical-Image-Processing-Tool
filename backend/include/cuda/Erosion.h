#pragma once

#include "core/IFilter.h"

class ErosionCUDA : public IFilter {
public:
    ErosionCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "erosion";
    }

    ~ErosionCUDA() override = default;
};