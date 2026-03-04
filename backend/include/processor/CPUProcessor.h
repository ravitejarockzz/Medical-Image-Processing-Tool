#pragma once

#include "processor/ImageProcessor.h"

class CPUProcessor : public ImageProcessor {
public:
    CPUProcessor() = default;

    cv::Mat process(
        const cv::Mat& input,
        const std::string& operation,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    ~CPUProcessor() override = default;
};