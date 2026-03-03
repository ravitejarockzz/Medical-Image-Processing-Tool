#pragma once

#include "ImageProcessor.h"

// -----------------------------------------------------
// CPU Implementation (OpenCV based)
// -----------------------------------------------------
class CPUProcessor : public ImageProcessor
{
public:
    CPUProcessor() = default;
    ~CPUProcessor() override = default;

    std::vector<unsigned char> applyGaussianBlur(
        const std::string& imageData,
        int kernel) override;

    std::vector<unsigned char> applyCanny(
        const std::string& imageData,
        int low,
        int high) override;

    std::vector<unsigned char> applyGrayscale(
        const std::string& imageData) override;
};