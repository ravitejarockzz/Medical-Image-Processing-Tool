#pragma once

#include "ImageProcessor.h"

// -----------------------------------------------------
// CUDA Implementation (GPU Accelerated)
// -----------------------------------------------------
class CUDAProcessor : public ImageProcessor
{
public:
    CUDAProcessor() = default;
    ~CUDAProcessor() override = default;

    std::vector<unsigned char> applyGaussianBlur(
        const std::string& imageData,
        int kernel) override;

    std::vector<unsigned char> applyCanny(
        const std::string& imageData,
        int low,
        int high) override;
};