#include "cuda/OtsuThreshold.h" // Adjust path to match your headers
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <iostream>

// ==========================================================
// KERNEL 1: Calculate Histogram
// ==========================================================
namespace otsu_kernels {
    __global__ void calcHistogramKernel(const unsigned char* input, unsigned int* hist, int width, int height);
    __global__ void calcOtsuThresholdKernel(const unsigned int* hist, int total_pixels, unsigned char* optimal_threshold);
    __global__ void applyOtsuThresholdKernel(const unsigned char* input, unsigned char* output, const unsigned char* optimal_threshold, int width, int height);
}
__global__ void otsu_kernels::calcHistogramKernel(const unsigned char* input, unsigned int* hist, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // Safely count pixel intensities without thread collisions
        atomicAdd(&hist[input[idx]], 1);
    }
}

// ==========================================================
// KERNEL 2: Calculate Optimal Threshold (Otsu's Method)
// ==========================================================
// Launched with 1 Block and 1 Thread
__global__ void otsu_kernels::calcOtsuThresholdKernel(const unsigned int* hist, int total_pixels, unsigned char* optimal_threshold) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 256; ++i) {
            sum += i * hist[i];
        }

        float sumB = 0.0f;
        int wB = 0;
        int wF = 0;

        float varMax = 0.0f;
        int threshold = 0;

        // Test all possible threshold values (0-255)
        for (int t = 0; t < 256; ++t) {
            wB += hist[t];                 // Weight Background
            if (wB == 0) continue;

            wF = total_pixels - wB;        // Weight Foreground
            if (wF == 0) break;

            sumB += (float)(t * hist[t]);

            float mB = sumB / wB;          // Mean Background
            float mF = (sum - sumB) / wF;  // Mean Foreground

            // Calculate Between-Class Variance
            float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

            // Check if we found a new maximum variance
            if (varBetween > varMax) {
                varMax = varBetween;
                threshold = t;
            }
        }

        // Save the winning threshold to device memory
        *optimal_threshold = (unsigned char)threshold;
    }
}

// ==========================================================
// KERNEL 3: Apply the Threshold
// ==========================================================
__global__ void otsu_kernels::applyOtsuThresholdKernel(const unsigned char* input, unsigned char* output, const unsigned char* optimal_threshold, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char t = *optimal_threshold; // Read the winning threshold
        
        // If pixel > threshold, turn white (255), else turn black (0)
        output[idx] = (input[idx] > t) ? 255 : 0;
    }
}

// ==========================================================
// CLASS IMPLEMENTATION
// ==========================================================
cv::Mat OtsuThresholdCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    // 1. Convert to Grayscale on CPU (saves PCIe bandwidth)
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    int width = gray.cols;
    int height = gray.rows;
    int total_pixels = width * height;
    size_t img_size = total_pixels * sizeof(unsigned char);

    cv::Mat output(height, width, CV_8UC1);

    // 2. Allocate Device Memory
    unsigned char *d_in, *d_out, *d_optimal_threshold;
    unsigned int *d_hist;

    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMalloc(&d_optimal_threshold, sizeof(unsigned char));

    // CRITICAL: Clear the histogram to 0 before counting!
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    // 3. Host to Device Transfer
    cudaMemcpy(d_in, gray.ptr(), img_size, cudaMemcpyHostToDevice);

    // 4. Setup Grid and Block
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 5. Fire the 3-Stage Pipeline
    otsu_kernels::calcHistogramKernel<<<grid, block>>>(d_in, d_hist, width, height);
    
    otsu_kernels::calcOtsuThresholdKernel<<<1, 1>>>(d_hist, total_pixels, d_optimal_threshold);

    otsu_kernels::applyOtsuThresholdKernel<<<grid, block>>>(d_in, d_out, d_optimal_threshold, width, height);

    // 6. Check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist); cudaFree(d_optimal_threshold);
        throw std::runtime_error(std::string("CUDA Otsu Error: ") + cudaGetErrorString(err));
    }

    // 7. Device to Host Transfer & Cleanup
    cudaMemcpy(output.ptr(), d_out, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in); 
    cudaFree(d_out); 
    cudaFree(d_hist); 
    cudaFree(d_optimal_threshold);

    return output;
}

// Auto-register to the factory
REGISTER_CUDA_FILTER(OtsuThresholdCUDA, "otsu_threshold")