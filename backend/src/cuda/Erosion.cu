#include "cuda/Erosion.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat ErosionCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itKSize = parameters.find("kernel_size");
    auto itIterations = parameters.find("iterations");

    if (itKSize == parameters.end() || itIterations == parameters.end()) {
        throw std::runtime_error("Missing parameters: kernel_size or iterations");
    }

    int ksize = static_cast<int>(itKSize->second);
    int iterations = static_cast<int>(itIterations->second);

    if (ksize < 3 || ksize % 2 == 0) {
        throw std::runtime_error("kernel_size must be an odd number >= 3");
    }
    if (iterations < 1) {
        throw std::runtime_error("iterations must be at least 1");
    }

    // Create the structural element (the "brush" shape). We use a standard Rectangle here.
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT, 
        cv::Size(ksize, ksize)
    );

    cv::Mat output;
    cv::erode(input, output, element, cv::Point(-1, -1), iterations);

    return output;
}

REGISTER_CUDA_FILTER(ErosionCUDA, "erosion")