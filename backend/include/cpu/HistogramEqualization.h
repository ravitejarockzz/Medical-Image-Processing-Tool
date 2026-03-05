#pragma once

#include "core/IFilter.h"

class HistogramEqualizationCPU : public IFilter {
public:
    HistogramEqualizationCPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "histogram_equalization";
    }

    ~HistogramEqualizationCPU() override = default;
};