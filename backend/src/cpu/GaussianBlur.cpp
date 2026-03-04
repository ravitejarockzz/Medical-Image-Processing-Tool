#include "cpu/GaussianBlur.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat GaussianBlurCPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto it = parameters.find("kernel");
    if (it == parameters.end()) {
        throw std::runtime_error("Missing parameter: kernel");
    }

    int kernel = static_cast<int>(it->second);

    if (kernel < 3 || kernel % 2 == 0) {
        throw std::runtime_error("Kernel must be odd and >= 3");
    }

    cv::Mat output;
    cv::GaussianBlur(input, output, cv::Size(kernel, kernel), 0);

    return output;
}

REGISTER_CPU_FILTER(GaussianBlurCPU, "gaussian_blur")