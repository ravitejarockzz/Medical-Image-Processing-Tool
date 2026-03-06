#include "cpu/SIFTDetector.h"
#include "factory/FilterFactory.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

cv::Mat SIFTDetectorCPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itFeatures = parameters.find("nfeatures");
    auto itContrast = parameters.find("contrast_threshold");
    auto itEdge = parameters.find("edge_threshold");
    auto itSigma = parameters.find("sigma");

    if (itFeatures == parameters.end() || itContrast == parameters.end() || 
        itEdge == parameters.end() || itSigma == parameters.end()) {
        throw std::runtime_error("Missing parameters for SIFT");
    }

    int nfeatures = static_cast<int>(itFeatures->second);
    double contrastThreshold = itContrast->second;
    double edgeThreshold = itEdge->second;
    double sigma = itSigma->second;

    // SIFT works best on grayscale images
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    // 1. Create the SIFT detector
    // nOctaveLayers is kept at default (3) as it rarely needs tweaking for general UI
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(nfeatures, 3, contrastThreshold, edgeThreshold, sigma);

    // 2. Detect the keypoints
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(gray, keypoints);

    // 3. Draw the keypoints onto the original color image
    cv::Mat output;
    // cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS draws circles with size and orientation lines!
    cv::drawKeypoints(input, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return output;
}

// Auto-register to the factory
REGISTER_CPU_FILTER(SIFTDetectorCPU, "sift")