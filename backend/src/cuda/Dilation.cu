
#include "cuda/Dilation.h" // Adjust path to match your headers
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <iostream>

// ==========================================================
// 1. CUDA KERNEL (Local Maximum Search)
// ==========================================================
__global__ void dilationRectKernel(
    const unsigned char* input, unsigned char* output, 
    int width, int height, int step, int channels, int radius) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Process each channel independently (handles both Grayscale and BGR)
    int out_idx = y * step + x * channels;

    for (int c = 0; c < channels; ++c) {
        unsigned char local_max = 0;

        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                // Boundary clamping
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);

                int in_idx = iy * step + ix * channels + c;
                unsigned char val = input[in_idx];
                
                // Dilation expands white/bright areas (find the maximum)
                if (val > local_max) {
                    local_max = val;
                }
            }
        }
        output[out_idx + c] = local_max;
    }
}

// ==========================================================
// 2. CLASS IMPLEMENTATION
// ==========================================================
cv::Mat DilationCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itKSize = parameters.find("kernel_size");
    auto itIterations = parameters.find("iterations");

    if (itKSize == parameters.end() || itIterations == parameters.end()) {
        throw std::runtime_error("Missing parameters: kernel_size or iterations");
    }

    int ksize = static_cast<int>(itKSize->second);
    int iterations = static_cast<int>(itIterations->second);

    if (ksize < 3 || ksize % 2 == 0) throw std::runtime_error("kernel_size must be an odd number >= 3");
    if (iterations < 1) throw std::runtime_error("iterations must be at least 1");

    int channels = input.channels();
    if (channels != 1 && channels != 3) {
        throw std::runtime_error("CUDA Dilation supports 1-channel or 3-channel 8-bit images only.");
    }

    cv::Mat output(input.size(), input.type());
    size_t memory_size = input.step * input.rows;
    int radius = ksize / 2;

    // 1. Allocate Two Device Buffers for "Ping-Pong" iterations
    unsigned char *d_bufferA = nullptr, *d_bufferB = nullptr;
    cudaMalloc(&d_bufferA, memory_size);
    cudaMalloc(&d_bufferB, memory_size);

    // 2. Host to Device Transfer
    cudaMemcpy(d_bufferA, input.ptr(), memory_size, cudaMemcpyHostToDevice);

    // 3. Setup Execution Grid
    dim3 block(16, 16);
    dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

    // 4. Ping-Pong Iteration Loop
    // We swap the read and write buffers on the GPU for every iteration
    unsigned char* d_read = d_bufferA;
    unsigned char* d_write = d_bufferB;

    for (int i = 0; i < iterations; ++i) {
        dilationRectKernel<<<grid, block>>>(
            d_read, d_write, 
            input.cols, input.rows, input.step, channels, radius
        );
        
        // Wait for this iteration to finish before starting the next
        cudaDeviceSynchronize();

        // Swap pointers for the next iteration
        unsigned char* temp = d_read;
        d_read = d_write;
        d_write = temp;
    }

    // 5. Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_bufferA); cudaFree(d_bufferB);
        throw std::runtime_error(std::string("CUDA Dilation Error: ") + cudaGetErrorString(err));
    }

    // 6. Device to Host Transfer
    // Because we swapped the pointers at the end of the loop, 
    // d_read now points to the buffer that was just written to!
    cudaMemcpy(output.ptr(), d_read, memory_size, cudaMemcpyDeviceToHost);

    // 7. Cleanup
    cudaFree(d_bufferA);
    cudaFree(d_bufferB);

    return output;
}

// Auto-register to the factory
REGISTER_CUDA_FILTER(DilationCUDA, "dilation")