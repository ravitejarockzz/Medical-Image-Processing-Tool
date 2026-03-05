#pragma once

#include "core/IFilter.h"

class CLAHECPU : public IFilter {
public:
    CLAHECPU() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "clahe";
    }

    ~CLAHECPU() override = default;
};  