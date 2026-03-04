#pragma once

#include "processor/ImageProcessor.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class CUDAProcessor : public ImageProcessor {
public:
    CUDAProcessor() = default;

    cv::Mat process(
        const cv::Mat& input,
        const std::string& operation,
        const std::unordered_map<std::string, double>& parameters
    ) override;

    ~CUDAProcessor() override = default;
};