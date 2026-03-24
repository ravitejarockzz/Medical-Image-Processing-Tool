
#include "cuda/BilateralFilter.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <iostream>

// ==========================================================
// 1. CUDA KERNEL
// ==========================================================
__global__ void bilateralFilterKernel(
    const unsigned char* input, 
    unsigned char* output, 
    int width, int height, int step, 
    int radius, float coeff_space, float coeff_color) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int center_idx = y * step + x * 3;
    
    // Center pixel colors
    float b0 = input[center_idx];
    float g0 = input[center_idx + 1];
    float r0 = input[center_idx + 2];

    float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f;
    float sum_w = 0.0f;

    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            // Boundary clamping
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);

            int idx = iy * step + ix * 3;
            float b = input[idx];
            float g = input[idx + 1];
            float r = input[idx + 2];

            // 1. Spatial distance squared
            float dist_space = (kx * kx) + (ky * ky);
            
            // 2. Color distance squared (Euclidean distance in BGR space)
            float dist_color = (b - b0) * (b - b0) + 
                               (g - g0) * (g - g0) + 
                               (r - r0) * (r - r0);

            // 3. Combined weight (exp(A+B) is faster than exp(A)*exp(B))
            float w = expf(dist_space * coeff_space + dist_color * coeff_color);

            sum_b += b * w;
            sum_g += g * w;
            sum_r += r * w;
            sum_w += w;
        }
    }

    // Write normalized output
    output[center_idx]     = (unsigned char)(sum_b / sum_w);
    output[center_idx + 1] = (unsigned char)(sum_g / sum_w);
    output[center_idx + 2] = (unsigned char)(sum_r / sum_w);
}

// ==========================================================
// 2. METHOD IMPLEMENTATION
// ==========================================================
cv::Mat BilateralFilterCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itD = parameters.find("d");
    auto itSigmaColor = parameters.find("sigma_color");
    auto itSigmaSpace = parameters.find("sigma_space");

    if (itD == parameters.end() || itSigmaColor == parameters.end() || itSigmaSpace == parameters.end()) {
        throw std::runtime_error("Missing parameters: d, sigma_color, or sigma_space");
    }

    int d = static_cast<int>(itD->second);
    float sigmaColor = static_cast<float>(itSigmaColor->second);
    float sigmaSpace = static_cast<float>(itSigmaSpace->second);

    if (d <= 0) {
        throw std::runtime_error("Diameter (d) must be greater than 0");
    }

    if (input.type() != CV_8UC3) {
        throw std::runtime_error("CUDA Bilateral filter currently only supports CV_8UC3 (3-channel 8-bit) images.");
    }

    cv::Mat output(input.size(), input.type());
    size_t memory_size = input.step * input.rows;

    // Calculate radius and Gaussian coefficients for the kernel
    int radius = d / 2;
    float coeff_space = -0.5f / (sigmaSpace * sigmaSpace);
    float coeff_color = -0.5f / (sigmaColor * sigmaColor);

    // Device Memory Allocation
    unsigned char *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, memory_size);
    cudaMalloc(&d_output, memory_size);

    // Host to Device Copy
    cudaMemcpy(d_input, input.ptr(), memory_size, cudaMemcpyHostToDevice);

    // Kernel Execution
    dim3 block(16, 16);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
    
    bilateralFilterKernel<<<grid, block>>>(
        d_input, d_output, input.cols, input.rows, input.step, 
        radius, coeff_space, coeff_color
    );

    // Wait for GPU to finish & check for kernel launch errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(std::string("CUDA Kernel Error: ") + cudaGetErrorString(err));
    }

    // Device to Host Copy
    cudaMemcpy(output.ptr(), d_output, memory_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

// Auto-register to the factory
REGISTER_CUDA_FILTER(BilateralFilterCUDA, "bilateral_filter")

