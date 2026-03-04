#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <string>

class ImageProcessor {
public:
    virtual cv::Mat process(
        const cv::Mat& input,
        const std::string& operation,
        const std::unordered_map<std::string, double>& parameters
    ) = 0;

    virtual ~ImageProcessor() = default;
};