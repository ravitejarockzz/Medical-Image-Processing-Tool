#include "cpu/BilateralFilter.h" // Adjust path if needed
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat BilateralFilterCPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itD = parameters.find("d");
    auto itSigmaColor = parameters.find("sigma_color");
    auto itSigmaSpace = parameters.find("sigma_space");

    if (itD == parameters.end() || itSigmaColor == parameters.end() || itSigmaSpace == parameters.end()) {
        throw std::runtime_error("Missing parameters: d, sigma_color, or sigma_space");
    }

    int d = static_cast<int>(itD->second);
    double sigmaColor = itSigmaColor->second;
    double sigmaSpace = itSigmaSpace->second;

    if (d <= 0) {
        throw std::runtime_error("Diameter (d) must be greater than 0");
    }

    cv::Mat output;
    
    // Bilateral filter requires 8-bit or floating-point images.
    // If you pass it something unusual, it might crash, but standard images work perfectly.
    cv::bilateralFilter(input, output, d, sigmaColor, sigmaSpace);

    return output;
}

// Auto-register to the factory
REGISTER_CPU_FILTER(BilateralFilterCPU, "bilateral_filter")