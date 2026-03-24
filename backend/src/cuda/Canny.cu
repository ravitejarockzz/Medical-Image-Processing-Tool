
#include "cuda/Canny.h" // Adjust path to match your headers
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <cmath>

// ==========================================================
// KERNEL 1: Sobel Gradients (Magnitude & Direction)
// ==========================================================
__global__ void sobelKernel(
    const unsigned char* input, float* magnitude, unsigned char* direction, 
    int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ignore 1-pixel border to avoid out-of-bounds
    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) return;

    int idx = y * width + x;

    // Sobel X
    int gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
             -2*input[y*width + (x-1)]   + 2*input[y*width + (x+1)]
             -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];

    // Sobel Y
    int gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
             +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

    // Magnitude
    float mag = sqrtf((float)(gx * gx + gy * gy));
    magnitude[idx] = mag;

    // Angle (Converted to Degrees)
    float angle = atan2f((float)gy, (float)gx) * 180.0f / 3.14159265f;
    if (angle < 0.0f) angle += 180.0f;

    // Quantize angle to 4 discrete directions (0, 45, 90, 135)
    unsigned char d = 0;
    if ((angle < 22.5f) || (angle >= 157.5f)) d = 0;
    else if (angle >= 22.5f && angle < 67.5f) d = 45;
    else if (angle >= 67.5f && angle < 112.5f) d = 90;
    else if (angle >= 112.5f && angle < 157.5f) d = 135;
    
    direction[idx] = d;
}

// ==========================================================
// KERNEL 2: Non-Maximum Suppression (Thinning the edges)
// ==========================================================
__global__ void nmsKernel(
    const float* magnitude, const unsigned char* direction, unsigned char* nms, 
    int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) return;

    int idx = y * width + x;
    unsigned char d = direction[idx];
    float m = magnitude[idx];

    float mag1 = 0.0f, mag2 = 0.0f;

    // Check neighbors along the gradient direction
    if (d == 0) {
        mag1 = magnitude[idx - 1];
        mag2 = magnitude[idx + 1];
    } else if (d == 45) {
        mag1 = magnitude[(y-1)*width + (x+1)];
        mag2 = magnitude[(y+1)*width + (x-1)];
    } else if (d == 90) {
        mag1 = magnitude[(y-1)*width + x];
        mag2 = magnitude[(y+1)*width + x];
    } else if (d == 135) {
        mag1 = magnitude[(y-1)*width + (x-1)];
        mag2 = magnitude[(y+1)*width + (x+1)];
    }

    // Suppress non-maximums
    if (m >= mag1 && m >= mag2) {
        nms[idx] = (unsigned char)min(255.0f, m);
    } else {
        nms[idx] = 0;
    }
}

// ==========================================================
// KERNEL 3: Double Threshold & Simple Hysteresis
// ==========================================================
__global__ void hysteresisKernel(
    const unsigned char* nms, unsigned char* output, 
    int width, int height, float low_thresh, float high_thresh) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) return;

    int idx = y * width + x;
    unsigned char p = nms[idx];

    if (p >= high_thresh) {
        output[idx] = 255; // Strong edge
    } else if (p >= low_thresh) {
        // Weak edge: Check 8 neighbors to see if it touches a strong edge
        bool strong = false;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                if (nms[(y + ky) * width + (x + kx)] >= high_thresh) {
                    strong = true;
                }
            }
        }
        output[idx] = strong ? 255 : 0;
    } else {
        output[idx] = 0; // Not an edge
    }
}

// ==========================================================
// CLASS IMPLEMENTATION
// ==========================================================
cv::Mat CannyCUDA::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    auto itLow = parameters.find("low");
    auto itHigh = parameters.find("high");

    if (itLow == parameters.end() || itHigh == parameters.end()) {
        throw std::runtime_error("Missing parameters: low and high");
    }

    float low = static_cast<float>(itLow->second);
    float high = static_cast<float>(itHigh->second);

    if (low < 0 || high <= low) {
        throw std::runtime_error("Invalid Canny thresholds");
    }

    // 1. Convert to Grayscale on CPU
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    // Ensure memory is continuous for easy 1D array indexing on GPU
    if (!gray.isContinuous()) {
        gray = gray.clone();
    }

    int width = gray.cols;
    int height = gray.rows;
    size_t size_8u = width * height * sizeof(unsigned char);
    size_t size_32f = width * height * sizeof(float);

    cv::Mat output = cv::Mat::zeros(height, width, CV_8UC1);

    // 2. Allocate Pipeline Buffers on Device
    unsigned char *d_gray, *d_dir, *d_nms, *d_out;
    float *d_mag;

    cudaMalloc(&d_gray, size_8u);
    cudaMalloc(&d_mag,  size_32f); // Magnitude needs to be float!
    cudaMalloc(&d_dir,  size_8u);
    cudaMalloc(&d_nms,  size_8u);
    cudaMalloc(&d_out,  size_8u);

    // Initialize output to 0 (since we skip borders in kernels)
    cudaMemset(d_out, 0, size_8u);

    // 3. Host to Device Transfer (Only the initial image)
    cudaMemcpy(d_gray, gray.ptr(), size_8u, cudaMemcpyHostToDevice);

    // 4. Execution Config
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 5. Fire the Pipeline! (Data stays on GPU between steps)
    sobelKernel<<<grid, block>>>(d_gray, d_mag, d_dir, width, height);
    
    nmsKernel<<<grid, block>>>(d_mag, d_dir, d_nms, width, height);
    
    hysteresisKernel<<<grid, block>>>(d_nms, d_out, width, height, low, high);

    // 6. Check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_gray); cudaFree(d_mag); cudaFree(d_dir); cudaFree(d_nms); cudaFree(d_out);
        throw std::runtime_error(std::string("CUDA Canny Pipeline Error: ") + cudaGetErrorString(err));
    }

    // 7. Bring only the final result back to Host
    cudaMemcpy(output.ptr(), d_out, size_8u, cudaMemcpyDeviceToHost);

    // 8. Cleanup all intermediate buffers
    cudaFree(d_gray);
    cudaFree(d_mag);
    cudaFree(d_dir);
    cudaFree(d_nms);
    cudaFree(d_out);

    return output;
}

// Auto-register to the factory
REGISTER_CUDA_FILTER(CannyCUDA, "canny")