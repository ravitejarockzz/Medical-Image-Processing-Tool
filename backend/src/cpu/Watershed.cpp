#include "cpu/Watershed.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

cv::Mat WatershedCPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itKSize = parameters.find("kernel_size");
    auto itDistRatio = parameters.find("distance_ratio");

    if (itKSize == parameters.end() || itDistRatio == parameters.end()) {
        throw std::runtime_error("Missing parameters: kernel_size or distance_ratio");
    }

    int ksize = static_cast<int>(itKSize->second);
    double distRatio = itDistRatio->second;

    if (ksize < 3 || ksize % 2 == 0) throw std::runtime_error("kernel_size must be an odd number >= 3");

    // 1. Convert to grayscale and apply Otsu's Thresholding
    cv::Mat gray, thresh;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    // We use THRESH_BINARY_INV because watershed traditionally expects background to be black, objects to be white
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 2. Morphological Opening to remove noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
    cv::Mat opening;
    cv::morphologyEx(thresh, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

    // 3. Find sure background area (dilate the objects)
    cv::Mat sure_bg;
    cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3);

    // 4. Find sure foreground area using Distance Transform
    cv::Mat dist_transform, sure_fg;
    cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5);
    double maxVal;
    cv::minMaxLoc(dist_transform, nullptr, &maxVal);
    cv::threshold(dist_transform, sure_fg, distRatio * maxVal, 255, 0);
    sure_fg.convertTo(sure_fg, CV_8U);

    // 5. Find unknown region (borders between touching objects)
    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);

    // 6. Label the markers for the Watershed algorithm
    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);
    markers += 1; // Add 1 so the sure background is 1, not 0
    
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (unknown.at<uchar>(i, j) == 255) {
                markers.at<int>(i, j) = 0; // Mark the unknown border region as 0
            }
        }
    }

    // 7. Apply Watershed
    cv::Mat output = input.clone();
    if (output.channels() == 1) {
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR); // Watershed needs a 3-channel image
    }
    cv::watershed(output, markers);

    // 8. Draw the segmented boundaries in bright Red!
    output.setTo(cv::Scalar(0, 0, 255), markers == -1);

    return output;
}

REGISTER_CPU_FILTER(WatershedCPU, "watershed")