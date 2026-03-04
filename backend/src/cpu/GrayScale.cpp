#include "cpu/GrayScale.h"
#include "core/IFilter.h"

#include "factory/FilterFactory.h"
#include <opencv2/opencv.hpp>


cv::Mat GrayScaleCPU::apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& params
)
    {
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

        cv::Mat output;
        cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);

        return output;
    }


REGISTER_CPU_FILTER(GrayScaleCPU, "grayscale");