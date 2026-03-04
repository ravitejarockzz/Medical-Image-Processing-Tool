#include "cuda/Canny.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat CannyCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itLow = parameters.find("low");
    auto itHigh = parameters.find("high");

    if (itLow == parameters.end() || itHigh == parameters.end()) {
        throw std::runtime_error("Missing parameters: low and high");
    }

    double low = itLow->second;
    double high = itHigh->second;

    if (low < 0 || high <= low) {
        throw std::runtime_error("Invalid Canny thresholds");
    }

    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    cv::Mat edges;
    cv::Canny(gray, edges, low, high);

    return edges;
}

REGISTER_CUDA_FILTER(CannyCUDA, "canny")