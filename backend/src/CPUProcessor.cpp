#include "CPUProcessor.h"

#include <opencv2/opencv.hpp>
#include <stdexcept>

// -------------------------------------
// Utility: Decode image from memory
// -------------------------------------
static cv::Mat decodeImage(const std::string& imageData)
{
    std::vector<unsigned char> buffer(imageData.begin(), imageData.end());
    cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (img.empty())
        throw std::runtime_error("Failed to decode image");

    return img;
}

// -------------------------------------
// Utility: Encode image to PNG memory
// -------------------------------------
static std::vector<unsigned char> encodePNG(const cv::Mat& image)
{
    std::vector<unsigned char> buffer;
    cv::imencode(".png", image, buffer);
    return buffer;
}

// -------------------------------------
// Gaussian Blur
// -------------------------------------
std::vector<unsigned char> CPUProcessor::applyGaussianBlur(
    const std::string& imageData,
    int kernel)
{
    cv::Mat img = decodeImage(imageData);

    cv::Mat result;
    cv::GaussianBlur(img, result, cv::Size(kernel, kernel), 0);

    return encodePNG(result);
}

// -------------------------------------
// Canny Edge Detection
// -------------------------------------
std::vector<unsigned char> CPUProcessor::applyCanny(
    const std::string& imageData,
    int low,
    int high)
{
    cv::Mat img = decodeImage(imageData);

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(gray, edges, low, high);

    return encodePNG(edges);
}