#pragma once

#include <string>
#include <vector>

// -----------------------------------------------------
// Abstract Base Class for Image Processing
// -----------------------------------------------------
class ImageProcessor
{
public:
    virtual ~ImageProcessor() = default;

    // Apply Gaussian Blur
    virtual std::vector<unsigned char> applyGaussianBlur(
        const std::string& imageData,
        int kernel) = 0;

    // Apply Canny Edge Detection
    virtual std::vector<unsigned char> applyCanny(
        const std::string& imageData,
        int low,
        int high) = 0;

};