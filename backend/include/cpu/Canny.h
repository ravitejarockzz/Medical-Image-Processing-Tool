#pragma once

#include "core/IFilter.h"

class CannyCPU : public IFilter {
public:
    CannyCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "canny";
    }

    ~CannyCPU() override = default;
};