#pragma once

#include "core/IFilter.h"

class WatershedCUDA : public IFilter {
public:
    WatershedCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "watershed";
    }

    ~WatershedCUDA() override = default;
};