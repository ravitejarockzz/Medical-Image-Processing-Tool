#pragma once

#include "core/IFilter.h"

class KMeansSegmentationCUDA: public IFilter {
public:
    KMeansSegmentationCUDA() = default;

    cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    std::string name() const override {
        return "KMeansSegmentationCUDA";
    }

    ~KMeansSegmentationCUDA() override = default;
};