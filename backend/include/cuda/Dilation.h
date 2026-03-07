#pragma once

#include "core/IFilter.h"

class DilationCUDA : public IFilter {
public:
    DilationCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "dilation";
    }

    ~DilationCUDA() override = default;
};