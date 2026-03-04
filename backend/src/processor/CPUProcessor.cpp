#include "processor/CPUProcessor.h"
#include "factory/FilterFactory.h"

#include <stdexcept>

cv::Mat CPUProcessor::process(
    const cv::Mat& input,
    const std::string& operation,
    const std::unordered_map<std::string, double>& parameters
) {
    if (input.empty()) {
        throw std::runtime_error("Input image is empty.");
    }

    // Create filter from factory
    auto filter = FilterFactory::instance().create(operation, "cpu");

    if (!filter) {
        throw std::runtime_error("Failed to create CPU filter: " + operation);
    }

    // Apply filter
    return filter->apply(input, parameters);
}