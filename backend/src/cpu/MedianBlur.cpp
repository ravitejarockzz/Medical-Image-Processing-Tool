#include "cpu/MedianBlur.h" // Adjust path if needed
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat MedianBlurCPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itKSize = parameters.find("ksize");

    if (itKSize == parameters.end()) {
        throw std::runtime_error("Missing parameter: ksize");
    }

    int ksize = static_cast<int>(itKSize->second);

    // Validate OpenCV requirements for Median Blur
    if (ksize < 3 || ksize % 2 == 0) {
        throw std::runtime_error("ksize must be an odd number >= 3");
    }

    cv::Mat output;
    cv::medianBlur(input, output, ksize);

    return output;
}

// Auto-register to the factory!
REGISTER_CPU_FILTER(MedianBlurCPU, "median_blur")