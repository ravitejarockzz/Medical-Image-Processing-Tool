#include "processor/CUDAProcessor.h"
#include "factory/FilterFactory.h"

#include <stdexcept>

cv::Mat CUDAProcessor::process(
    const cv::Mat& input,
    const std::string& operation,
    const std::unordered_map<std::string, double>& parameters
) {
    if (input.empty()) {
        throw std::runtime_error("Input image is empty.");
    }

    // Create filter from CUDA registry
    auto filter = FilterFactory::instance().create(operation, "cuda");

    if (!filter) {
        throw std::runtime_error("Failed to create CUDA filter: " + operation);
    }

    // For now this can internally still use CPU OpenCV.
    // Later you replace implementation with real CUDA kernels.
    return filter->apply(input, parameters);
}