
#include "cuda/AdaptiveThreshold.h" // Adjust path to match your headers
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <vector>
#include <cmath>

// ==========================================================
// 1. CUDA KERNEL (1-Channel Grayscale)
// ==========================================================
__global__ void adaptiveThresholdGaussianKernel(
    const unsigned char* input, 
    unsigned char* output, 
    int width, int height, int step, 
    const float* kernel, int ksize, float C) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = ksize / 2;
    float weighted_sum = 0.0f;

    // Calculate local Gaussian weighted sum
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            // Boundary clamping
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);

            float kval = kernel[(ky + half) * ksize + (kx + half)];
            int idx = iy * step + ix; // 1-channel stride

            weighted_sum += input[idx] * kval;
        }
    }

    // Apply adaptive threshold logic
    float threshold = weighted_sum - C;
    int center_idx = y * step + x;
    
    // Output 255 (White) if pixel > threshold, else 0 (Black)
    output[center_idx] = (input[center_idx] > threshold) ? 255 : 0;
}

// ==========================================================
// 2. CLASS IMPLEMENTATION
// ==========================================================
cv::Mat AdaptiveThresholdCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itBlockSize = parameters.find("block_size");
    auto itC = parameters.find("c");

    if (itBlockSize == parameters.end() || itC == parameters.end()) {
        throw std::runtime_error("Missing parameters: block_size or c");
    }

    int blockSize = static_cast<int>(itBlockSize->second);
    float C = static_cast<float>(itC->second);

    if (blockSize < 3 || blockSize % 2 == 0) {
        throw std::runtime_error("block_size must be an odd number >= 3");
    }

    // 1. Convert to Grayscale on CPU (Saves PCIe Bandwidth!)
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    cv::Mat output(gray.size(), CV_8UC1); // 1-Channel Output
    size_t memory_size = gray.step * gray.rows;

    // 2. Generate 2D Gaussian Weights (Same formula OpenCV uses)
    float sigma = 0.3f * ((blockSize - 1) * 0.5f - 1.0f) + 0.8f;
    std::vector<float> h_kernel(blockSize * blockSize);
    float sum = 0.0f;
    int half = blockSize / 2;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float val = std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            h_kernel[(y + half) * blockSize + (x + half)] = val;
            sum += val;
        }
    }
    // Normalize the weights
    for (int i = 0; i < blockSize * blockSize; i++) h_kernel[i] /= sum;

    // 3. Device Memory Allocation
    unsigned char *d_input = nullptr, *d_output = nullptr;
    float* d_kernel = nullptr;
    
    cudaMalloc(&d_input, memory_size);
    cudaMalloc(&d_output, memory_size);
    cudaMalloc(&d_kernel, blockSize * blockSize * sizeof(float));

    // 4. Host to Device Transfer
    cudaMemcpy(d_input, gray.ptr(), memory_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), blockSize * blockSize * sizeof(float), cudaMemcpyHostToDevice);

    // 5. Execute Kernel
    dim3 block(16, 16);
    dim3 grid((gray.cols + block.x - 1) / block.x, (gray.rows + block.y - 1) / block.y);
    
    adaptiveThresholdGaussianKernel<<<grid, block>>>(
        d_input, d_output, gray.cols, gray.rows, gray.step, 
        d_kernel, blockSize, C
    );

    // 6. Check for errors and Synchronize
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_kernel);
        throw std::runtime_error(std::string("CUDA Kernel Error: ") + cudaGetErrorString(err));
    }

    // 7. Device to Host Transfer & Cleanup
    cudaMemcpy(output.ptr(), d_output, memory_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    // NOTE: If your frontend expects 3 channels to display properly, 
    // uncomment the following line to convert back to BGR before returning:
    cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);

    return output;
}

// Auto-register to the factory
REGISTER_CUDA_FILTER(AdaptiveThresholdCUDA, "adaptive_threshold")