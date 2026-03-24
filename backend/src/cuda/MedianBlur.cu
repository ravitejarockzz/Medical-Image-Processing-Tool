#include "cuda/MedianBlur.h" // Adjust path to match your headers
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <iostream>

// The maximum supported window size is 15x15 (225 pixels). 
// Allocating massive arrays inside a CUDA thread causes "Register Spilling" and kills performance.
#define MAX_WINDOW_SIZE 225 

// ==========================================================
// 1. CUDA KERNEL (Neighborhood Gather & Insertion Sort)
// ==========================================================
__global__ void medianBlurKernel(
    const unsigned char* input, unsigned char* output, 
    int width, int height, int step, int channels, int radius, int window_size) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int out_idx = y * step + x * channels;

    // Process each channel independently (handles both Grayscale and BGR)
    for (int c = 0; c < channels; ++c) {
        
        // Local array to hold the neighborhood pixels
        unsigned char window[MAX_WINDOW_SIZE];
        int count = 0;

        // 1. Gather the neighborhood
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                // Boundary clamping
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);

                int in_idx = iy * step + ix * channels + c;
                window[count++] = input[in_idx];
            }
        }

        // 2. Insertion Sort (Highly efficient for small arrays on GPUs)
        for (int i = 1; i < window_size; ++i) {
            unsigned char key = window[i];
            int j = i - 1;
            
            // Shift elements that are greater than the key to the right
            while (j >= 0 && window[j] > key) {
                window[j + 1] = window[j];
                j = j - 1;
            }
            window[j + 1] = key;
        }

        // 3. Pick the median value
        output[out_idx + c] = window[window_size / 2];
    }
}

// ==========================================================
// 2. CLASS IMPLEMENTATION
// ==========================================================
cv::Mat MedianBlurCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itKSize = parameters.find("ksize");

    if (itKSize == parameters.end()) {
        throw std::runtime_error("Missing parameter: ksize");
    }

    int ksize = static_cast<int>(itKSize->second);

    if (ksize < 3 || ksize % 2 == 0) {
        throw std::runtime_error("ksize must be an odd number >= 3");
    }

    int window_size = ksize * ksize;
    if (window_size > MAX_WINDOW_SIZE) {
        throw std::runtime_error("CUDA Median Blur currently supports a maximum ksize of 15.");
    }

    int channels = input.channels();
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("CUDA Median Blur supports 1-channel or 3-channel 8-bit images only.");
    }

    cv::Mat output(input.size(), input.type());
    size_t memory_size = input.step * input.rows;
    int radius = ksize / 2;

    // 1. Allocate Device Memory
    unsigned char *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, memory_size);
    cudaMalloc(&d_output, memory_size);

    // 2. Host to Device Transfer
    cudaMemcpy(d_input, input.ptr(), memory_size, cudaMemcpyHostToDevice);

    // 3. Setup Execution Grid
    dim3 block(16, 16);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

    // 4. Fire the Kernel
    medianBlurKernel<<<grid, block>>>(
        d_input, d_output, 
        input.cols, input.rows, input.step, channels, radius, window_size
    );

    // 5. Check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input); cudaFree(d_output);
        throw std::runtime_error(std::string("CUDA Median Blur Error: ") + cudaGetErrorString(err));
    }

    // 6. Device to Host Transfer
    cudaMemcpy(output.ptr(), d_output, memory_size, cudaMemcpyDeviceToHost);

    // 7. Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

// Auto-register to the factory
REGISTER_CUDA_FILTER(MedianBlurCUDA, "median_blur")