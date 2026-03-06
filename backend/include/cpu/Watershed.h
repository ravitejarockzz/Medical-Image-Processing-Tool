#pragma once

#include "core/IFilter.h"

class WatershedCPU : public IFilter {
public:
    WatershedCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "watershed";
    }

    ~WatershedCPU() override = default;
};