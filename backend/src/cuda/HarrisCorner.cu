#include "cuda/HarrisCorner.h" // Adjust path if your folder structure is different
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat HarrisCornerCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    // 1. Extract all required parameters from the frontend
    auto itBlockSize = parameters.find("block_size");
    auto itKSize = parameters.find("k_size");
    auto itK = parameters.find("k");
    auto itThreshold = parameters.find("threshold");

    // Ensure the frontend sent everything we need
    if (itBlockSize == parameters.end() || itKSize == parameters.end() ||
        itK == parameters.end() || itThreshold == parameters.end()) {
        throw std::runtime_error("Missing parameters: block_size, k_size, k, or threshold");
    }

    // Cast double parameters to appropriate types for OpenCV
    int blockSize = static_cast<int>(itBlockSize->second);
    int ksize = static_cast<int>(itKSize->second);
    double k = itK->second;
    int threshold = static_cast<int>(itThreshold->second);

    // 2. Validate parameters to prevent OpenCV crashes
    if (blockSize <= 0) throw std::runtime_error("block_size must be > 0");
    if (ksize <= 0 || ksize % 2 == 0) throw std::runtime_error("k_size must be a positive odd number (e.g., 3, 5, 7)");
    if (threshold < 0 || threshold > 255) throw std::runtime_error("threshold must be between 0 and 255");

    // 3. Convert image to Grayscale if necessary (Harris requires single-channel)
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    // 4. Apply Harris Corner Detection
    cv::Mat dst = cv::Mat::zeros(input.size(), CV_32FC1);
    cv::cornerHarris(gray, dst, blockSize, ksize, k);

    // 5. Normalize the result so the scores fit between 0 and 255
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // 6. Apply threshold to isolate the strongest corners
    // This turns it into a black image with white dots, matching your Canny output style!
    cv::Mat cornersMap;
    cv::threshold(dst_norm_scaled, cornersMap, threshold, 255, cv::THRESH_BINARY);

    return cornersMap;
}

// 🔥 The magic macro registers this to the factory automatically!
REGISTER_CUDA_FILTER(HarrisCornerCUDA, "harris_corner")