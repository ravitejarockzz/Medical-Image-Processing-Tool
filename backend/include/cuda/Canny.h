#pragma once

#include "core/IFilter.h"

class CannyCUDA : public IFilter {
public:
    CannyCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "canny";
    }

    ~CannyCUDA() override = default;
};