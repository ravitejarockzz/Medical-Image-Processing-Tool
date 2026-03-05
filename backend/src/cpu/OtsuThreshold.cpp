#include "cpu/OtsuThreshold.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>

cv::Mat OtsuThresholdCPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    // Otsu is automatic, so we don't need to extract any parameters!

    cv::Mat gray;
    // Otsu's method strictly requires a single-channel 8-bit image (Grayscale)
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    cv::Mat output;
    
    // The '0' is the threshold value, but because we pass cv::THRESH_OTSU, 
    // OpenCV completely ignores our '0' and calculates the optimal value itself.
    // The '255' means: "If a pixel passes the threshold, turn it pure white (255)."
    cv::threshold(gray, output, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    return output;
}

// Auto-register to the factory
REGISTER_CPU_FILTER(OtsuThresholdCPU, "otsu_threshold")