#include "cpu/CLAHE.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <stdexcept>

cv::Mat CLAHECPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itClipLimit = parameters.find("clip_limit");
    auto itTileSize = parameters.find("tile_grid_size");

    if (itClipLimit == parameters.end() || itTileSize == parameters.end()) {
        throw std::runtime_error("Missing parameters: clip_limit or tile_grid_size");
    }

    double clipLimit = itClipLimit->second;
    int tileSize = static_cast<int>(itTileSize->second);

    if (tileSize <= 0) {
        throw std::runtime_error("tile_grid_size must be greater than 0");
    }

    // Create the OpenCV CLAHE object
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clipLimit);
    clahe->setTilesGridSize(cv::Size(tileSize, tileSize));

    cv::Mat output;

    if (input.channels() == 1) {
        // Grayscale image (e.g., standard X-ray)
        clahe->apply(input, output);
    } 
    else if (input.channels() == 3) {
        // Color image: Convert to LAB color space, apply to L (Lightness), convert back
        cv::Mat lab;
        cv::cvtColor(input, lab, cv::COLOR_BGR2Lab);
        
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);
        
        // Apply CLAHE only to the Lightness channel
        clahe->apply(channels[0], channels[0]);
        
        cv::merge(channels, lab);
        cv::cvtColor(lab, output, cv::COLOR_Lab2BGR);
    } 
    else {
        output = input.clone();
    }

    return output;
}

// Auto-register to the factory
REGISTER_CPU_FILTER(CLAHECPU, "clahe")