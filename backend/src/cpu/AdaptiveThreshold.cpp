#include "cpu/AdaptiveThreshold.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat AdaptiveThresholdCPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itBlockSize = parameters.find("block_size");
    auto itC = parameters.find("c");

    if (itBlockSize == parameters.end() || itC == parameters.end()) {
        throw std::runtime_error("Missing parameters: block_size or c");
    }

    int blockSize = static_cast<int>(itBlockSize->second);
    double c = itC->second;

    // Validate OpenCV requirements
    if (blockSize < 3 || blockSize % 2 == 0) {
        throw std::runtime_error("block_size must be an odd number >= 3");
    }

    cv::Mat gray;
    // Adaptive thresholding strictly requires a grayscale image
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    cv::Mat output;
    
    // 255: Max value (white) for pixels passing the threshold
    // cv::ADAPTIVE_THRESH_GAUSSIAN_C: Uses a weighted Gaussian sum of the neighborhood (smoother than MEAN_C)
    // cv::THRESH_BINARY: Standard black/white output
    cv::adaptiveThreshold(
        gray, output, 255, 
        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 
        blockSize, c
    );

    return output;
}

// Auto-register to the factory
REGISTER_CPU_FILTER(AdaptiveThresholdCPU, "adaptive_threshold")