#pragma once

#include "core/IFilter.h"

class DilationCPU : public IFilter {
public:
    DilationCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "dilation";
    }

    ~DilationCPU() override = default;
};