// #include "cuda/GaussianBlur.h"
// #include "factory/FilterFactory.h"

// #include <opencv2/imgproc.hpp>
// #include <stdexcept>

// cv::Mat GaussianBlurCUDA::apply(
//     const cv::Mat& input,
//     const std::unordered_map<std::string, double>& parameters
// ) {
//     auto it = parameters.find("kernel");
//     if (it == parameters.end()) {
//         throw std::runtime_error("Missing parameter: kernel");
//     }

//     int kernel = static_cast<int>(it->second);

//     if (kernel < 3 || kernel % 2 == 0) {
//         throw std::runtime_error("Kernel must be odd and >= 3");
//     }

//     cv::Mat output;
//     cv::GaussianBlur(input, output, cv::Size(kernel, kernel), 0);

//     return output;
// }

// REGISTER_CUDA_FILTER(GaussianBlurCUDA, "gaussian_blur")

#include "cuda/GaussianBlur.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <vector>
#include <cmath>
#include <iostream>

// ==========================================================
// 1. CUDA DEVICE KERNEL
// ==========================================================
__global__ void gaussianBlurKernel(
    const unsigned char* input, 
    unsigned char* output, 
    int width, int height, int step, 
    const float* kernel, int ksize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = ksize / 2;
    float b = 0.0f, g = 0.0f, r = 0.0f;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            // Boundary clamping
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);

            float kval = kernel[(ky + half) * ksize + (kx + half)];
            int idx = iy * step + ix * 3; 

            b += input[idx]     * kval;
            g += input[idx + 1] * kval;
            r += input[idx + 2] * kval;
        }
    }

    int out_idx = y * step + x * 3;
    output[out_idx]     = (unsigned char)b;
    output[out_idx + 1] = (unsigned char)g;
    output[out_idx + 2] = (unsigned char)r;
}

// ==========================================================
// 2. HOST C++ IMPLEMENTATION (OpenCV Wrapper)
// ==========================================================
cv::Mat GaussianBlurCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto it = parameters.find("kernel");
    if (it == parameters.end()) {
        throw std::runtime_error("Missing parameter: kernel");
    }

    int ksize = static_cast<int>(it->second);

    if (ksize < 3 || ksize % 2 == 0) {
        throw std::runtime_error("Kernel must be odd and >= 3");
    }

    if (input.type() != CV_8UC3) {
        throw std::runtime_error("CUDA kernel only supports CV_8UC3 images.");
    }

    cv::Mat output(input.size(), input.type());
    size_t memory_size = input.step * input.rows;

    // --- Generate Gaussian Weights ---
    float sigma = 0.3f * ((ksize - 1) * 0.5f - 1.0f) + 0.8f;
    std::vector<float> h_kernel(ksize * ksize);
    float sum = 0.0f;
    int half = ksize / 2;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float val = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            h_kernel[(y + half) * ksize + (x + half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < ksize * ksize; i++) h_kernel[i] /= sum;

    // --- Device Memory Allocation ---
    unsigned char *d_input = nullptr, *d_output = nullptr;
    float* d_kernel = nullptr;
    
    cudaMalloc(&d_input, memory_size);
    cudaMalloc(&d_output, memory_size);
    cudaMalloc(&d_kernel, ksize * ksize * sizeof(float));

    // --- Copy Data to Device ---
    cudaMemcpy(d_input, input.ptr(), memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), ksize * ksize * sizeof(float), cudaMemcpyHostToDevice);

    // --- Launch Kernel ---
    dim3 block(16, 16);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
    gaussianBlurKernel<<<grid, block>>>(d_input, d_output, input.cols, input.rows, input.step, d_kernel, ksize);
    
    // Synchronize to catch any execution errors
    cudaDeviceSynchronize();

    // --- Copy Data Back to Host ---
    cudaMemcpy(output.ptr(), d_output, memory_size, cudaMemcpyDeviceToHost);

    // --- Cleanup ---
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return output;
}

// Register with your factory
REGISTER_CUDA_FILTER(GaussianBlurCUDA, "gaussian_blur")