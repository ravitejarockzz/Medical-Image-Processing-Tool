
#include "cuda/HarrisCorner.h" // Adjust path to match your headers
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>

// ==========================================================
// KERNEL 1: Sobel Derivatives & Products
// ==========================================================
__global__ void sobelDerivativesKernel(
    const unsigned char* src, 
    float* Ix2, float* Iy2, float* Ixy, 
    int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip the 1-pixel border to avoid out-of-bounds memory access
    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) return;

    int idx = y * width + x;

    // Standard 3x3 Sobel Operator
    float dx = -src[(y-1)*width + (x-1)] + src[(y-1)*width + (x+1)]
               -2.0f*src[y*width + (x-1)]   + 2.0f*src[y*width + (x+1)]
               -src[(y+1)*width + (x-1)] + src[(y+1)*width + (x+1)];

    float dy = -src[(y-1)*width + (x-1)] - 2.0f*src[(y-1)*width + x] - src[(y-1)*width + (x+1)]
               +src[(y+1)*width + (x-1)] + 2.0f*src[(y+1)*width + x] + src[(y+1)*width + (x+1)];

    // Store the products
    Ix2[idx] = dx * dx;
    Iy2[idx] = dy * dy;
    Ixy[idx] = dx * dy;
}

// ==========================================================
// KERNEL 2: Harris Response Calculation
// ==========================================================
__global__ void harrisResponseKernel(
    const float* Ix2, const float* Iy2, const float* Ixy, 
    float* response, 
    int width, int height, int blockSize, float k) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int half = blockSize / 2;
    
    // Skip borders where the block window would go out of bounds
    if (x < half || x >= width - half || y < half || y >= height - half) return;

    float Sxx = 0.0f, Syy = 0.0f, Sxy = 0.0f;

    // Sum the derivative products over the block neighborhood
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int neighbor_idx = (y + ky) * width + (x + kx);
            Sxx += Ix2[neighbor_idx];
            Syy += Iy2[neighbor_idx];
            Sxy += Ixy[neighbor_idx];
        }
    }

    // Calculate Harris Response: R = det(M) - k(trace(M))^2
    float det = (Sxx * Syy) - (Sxy * Sxy);
    float trace = Sxx + Syy;
    
    response[y * width + x] = det - k * (trace * trace);
}

// ==========================================================
// CLASS IMPLEMENTATION
// ==========================================================
cv::Mat HarrisCornerCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itBlockSize = parameters.find("block_size");
    auto itKSize = parameters.find("k_size");
    auto itK = parameters.find("k");
    auto itThreshold = parameters.find("threshold");

    if (itBlockSize == parameters.end() || itKSize == parameters.end() ||
        itK == parameters.end() || itThreshold == parameters.end()) {
        throw std::runtime_error("Missing parameters: block_size, k_size, k, or threshold");
    }

    int blockSize = static_cast<int>(itBlockSize->second);
    int ksize = static_cast<int>(itKSize->second);
    float k = static_cast<float>(itK->second);
    int threshold = static_cast<int>(itThreshold->second);

    if (blockSize <= 0) throw std::runtime_error("block_size must be > 0");
    if (threshold < 0 || threshold > 255) throw std::runtime_error("threshold must be between 0 and 255");
    
    // Custom CUDA implementation utilizes a fixed 3x3 Sobel kernel for performance.
    if (ksize != 3) {
        throw std::runtime_error("This custom CUDA implementation requires k_size to be exactly 3.");
    }

    // 1. Convert to Grayscale on CPU (saves PCIe transfer time)
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    int width = gray.cols;
    int height = gray.rows;
    size_t size_8u = width * height * sizeof(unsigned char);
    size_t size_32f = width * height * sizeof(float);

    // 2. Allocate Device Buffers
    unsigned char* d_gray;
    float *d_Ix2, *d_Iy2, *d_Ixy, *d_response;

    cudaMalloc(&d_gray, size_8u);
    cudaMalloc(&d_Ix2, size_32f);
    cudaMalloc(&d_Iy2, size_32f);
    cudaMalloc(&d_Ixy, size_32f);
    cudaMalloc(&d_response, size_32f);

    // Initialize response map to 0 (since kernels skip the borders)
    cudaMemset(d_response, 0, size_32f);

    // 3. Transfer Data
    cudaMemcpy(d_gray, gray.ptr(), size_8u, cudaMemcpyHostToDevice);

    // 4. Execute Pipeline
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    sobelDerivativesKernel<<<grid, block>>>(d_gray, d_Ix2, d_Iy2, d_Ixy, width, height);
    
    harrisResponseKernel<<<grid, block>>>(d_Ix2, d_Iy2, d_Ixy, d_response, width, height, blockSize, k);

    // 5. Check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_gray); cudaFree(d_Ix2); cudaFree(d_Iy2); cudaFree(d_Ixy); cudaFree(d_response);
        throw std::runtime_error(std::string("CUDA Harris Error: ") + cudaGetErrorString(err));
    }

    // 6. Download Raw Float Response to CPU
    cv::Mat dst(height, width, CV_32FC1);
    cudaMemcpy(dst.ptr(), d_response, size_32f, cudaMemcpyDeviceToHost);

    // Cleanup GPU Memory immediately
    cudaFree(d_gray); cudaFree(d_Ix2); cudaFree(d_Iy2); cudaFree(d_Ixy); cudaFree(d_response);

    // 7. CPU Fallback for Reduction (Min-Max Normalization & Threshold)
    cv::Mat dst_norm, dst_norm_scaled, cornersMap;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    cv::threshold(dst_norm_scaled, cornersMap, threshold, 255, cv::THRESH_BINARY);

    return cornersMap;
}

REGISTER_CUDA_FILTER(HarrisCornerCUDA, "harris_corner")