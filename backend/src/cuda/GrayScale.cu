#include "cuda/GrayScale.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

// ==========================================================
// 1. CUDA KERNEL
// ==========================================================
__global__ void grayscaleKernel(
    const unsigned char* input, 
    unsigned char* output, 
    int width, int height, int step) 
{
    // Calculate 2D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) return;

    // Calculate the 1D index for the pixel
    int idx = y * step + x * 3;

    // Read BGR values
    float b = (float)input[idx];
    float g = (float)input[idx + 1];
    float r = (float)input[idx + 2];

    // Standard NTSC grayscale conversion formula
    // Gray = 0.299*R + 0.587*G + 0.114*B
    unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

    // Write the grayscale value back to all 3 channels to maintain CV_8UC3 format
    output[idx]     = gray; // B
    output[idx + 1] = gray; // G
    output[idx + 2] = gray; // R
}

// ==========================================================
// 2. CLASS IMPLEMENTATION
// ==========================================================
cv::Mat GrayScaleCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& params
) {
    // Grayscale doesn't strictly need parameters, but we accept the map to satisfy the interface.
    
    if (input.type() != CV_8UC3) {
        throw std::runtime_error("CUDA Grayscale requires a 3-channel 8-bit (CV_8UC3) input image.");
    }

    cv::Mat output(input.size(), input.type());
    size_t memory_size = input.step * input.rows;

    // Allocate Device Memory
    unsigned char *d_input = nullptr;
    unsigned char *d_output = nullptr;

    cudaMalloc(&d_input, memory_size);
    cudaMalloc(&d_output, memory_size);

    // Copy data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_input, input.ptr(), memory_size, cudaMemcpyHostToDevice);

    // Setup Grid and Block dimensions
    dim3 block(16, 16);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

    // Execute Kernel
    grayscaleKernel<<<grid, block>>>(d_input, d_output, input.cols, input.rows, input.step);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(std::string("CUDA Grayscale Error: ") + cudaGetErrorString(err));
    }

    // Copy result back to Host
    cudaMemcpy(output.ptr(), d_output, memory_size, cudaMemcpyDeviceToHost);

    // Cleanup Device Memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

// Auto-register to the factory
REGISTER_CUDA_FILTER(GrayScaleCUDA, "grayscale")