#include "cuda/HistogramEqualization.h" // Adjust path to match your headers
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <vector>

// ==========================================================
// KERNEL 1: Calculate Histogram using Atomics
// ==========================================================
__global__ void calcHistogramKernel(const unsigned char* input, unsigned int* hist, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // atomicAdd safely counts pixels without threads overwriting each other
        atomicAdd(&hist[input[idx]], 1);
    }
}

// ==========================================================
// KERNEL 2: Calculate CDF and Build Look-Up Table (LUT)
// ==========================================================
// Launched with exactly 1 Block and 1 Thread. Iterating 256 times is instantaneous.
__global__ void buildLUTKernel(const unsigned int* hist, unsigned char* lut, int total_pixels) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int cdf = 0;
        int cdf_min = 0;
        bool min_found = false;

        for (int i = 0; i < 256; i++) {
            if (hist[i] > 0) {
                cdf += hist[i];
                if (!min_found) {
                    cdf_min = cdf; // The first non-zero CDF value
                    min_found = true;
                }
                
                // Standard Histogram Equalization formula
                float val = ((float)(cdf - cdf_min) / (float)(total_pixels - cdf_min)) * 255.0f;
                lut[i] = (unsigned char)min(max((int)(val + 0.5f), 0), 255);
            } else {
                lut[i] = 0;
            }
        }
    }
}

// ==========================================================
// KERNEL 3: Apply the LUT to remap the image
// ==========================================================
__global__ void applyLUTKernel(const unsigned char* input, unsigned char* output, const unsigned char* lut, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // Replace old pixel value with the new equalized value from the cheat sheet
        output[idx] = lut[input[idx]];
    }
}

// ==========================================================
// CLASS IMPLEMENTATION
// ==========================================================
cv::Mat HistogramEqualizationCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    // 1. Color Space Preparation (Done on CPU to save GPU PCIe bandwidth)
    cv::Mat process_channel;
    cv::Mat ycrcb;
    std::vector<cv::Mat> channels;

    if (input.channels() == 3) {
        cv::cvtColor(input, ycrcb, cv::COLOR_BGR2YCrCb);
        cv::split(ycrcb, channels);
        process_channel = channels[0]; // We only equalize the Y (Brightness) channel
    } else if (input.channels() == 1) {
        process_channel = input.clone();
    } else {
        // Fallback for 4-channel (RGBA) or unusual formats
        return input.clone(); 
    }

    int width = process_channel.cols;
    int height = process_channel.rows;
    int total_pixels = width * height;
    size_t img_size = total_pixels * sizeof(unsigned char);

    cv::Mat output_channel(height, width, CV_8UC1);

    // 2. Allocate Device Memory
    unsigned char *d_in, *d_out, *d_lut;
    unsigned int *d_hist;

    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMalloc(&d_lut, 256 * sizeof(unsigned char));

    // CRITICAL: Clear the histogram array to 0 before counting!
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    // 3. Host to Device Copy
    cudaMemcpy(d_in, process_channel.ptr(), img_size, cudaMemcpyHostToDevice);

    // 4. Setup Grid and Block
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 5. Fire the 3-Stage Pipeline
    calcHistogramKernel<<<grid, block>>>(d_in, d_hist, width, height);
    
    buildLUTKernel<<<1, 1>>>(d_hist, d_lut, total_pixels);

    applyLUTKernel<<<grid, block>>>(d_in, d_out, d_lut, width, height);

    // 6. Check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_hist); cudaFree(d_lut);
        throw std::runtime_error(std::string("CUDA Histogram Eq Error: ") + cudaGetErrorString(err));
    }

    // 7. Device to Host Copy & Cleanup
    cudaMemcpy(output_channel.ptr(), d_out, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in); 
    cudaFree(d_out); 
    cudaFree(d_hist); 
    cudaFree(d_lut);

    // 8. Reconstruct Image (If Color)
    if (input.channels() == 3) {
        cv::Mat final_output;
        channels[0] = output_channel; // Replace old Y with new equalized Y
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, final_output, cv::COLOR_YCrCb2BGR);
        return final_output;
    }

    return output_channel;
}

// Auto-register to the factory matching your provided code
REGISTER_CUDA_FILTER(HistogramEqualizationCUDA, "histogram_equalization")