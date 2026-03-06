#pragma once

#include "core/IFilter.h"

class ErosionCPU : public IFilter {
public:
    ErosionCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "erosion";
    }

    ~ErosionCPU() override = default;
};