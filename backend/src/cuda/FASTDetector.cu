#include "cuda/FASTDetector.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <vector>

// ==========================================================
// KERNEL: FAST-9 Corner Detection
// ==========================================================
__global__ void fastCornerKernel(const unsigned char* input, unsigned char* output, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // We need a 3-pixel padding because the FAST circle has a radius of 3
    if (x < 3 || x >= width - 3 || y < 3 || y >= height - 3) return;

    int center_idx = y * width + x;
    int center_val = input[center_idx];

    // The 16 pixels surrounding the center at radius 3
    int offsets[16] = {
        -3 * width + 0, -3 * width + 1, -2 * width + 2, -1 * width + 3,
         0 * width + 3,  1 * width + 3,  2 * width + 2,  3 * width + 1,
         3 * width + 0,  3 * width - 1,  2 * width - 2,  1 * width - 3,
         0 * width - 3, -1 * width - 3, -2 * width - 2, -3 * width - 1
    };

    int pixels[16];
    for (int i = 0; i < 16; i++) {
        pixels[i] = input[center_idx + offsets[i]];
    }

    int t_lower = center_val - threshold;
    int t_upper = center_val + threshold;

    bool is_corner = false;

    // Check for 9 contiguous pixels that are strictly BRIGHTER or strictly DARKER
    for (int i = 0; i < 16; i++) {
        int count_bright = 0;
        int count_dark = 0;

        for (int j = 0; j < 9; j++) {
            int p = pixels[(i + j) % 16];
            if (p > t_upper) count_bright++;
            if (p < t_lower) count_dark++;
        }

        if (count_bright == 9 || count_dark == 9) {
            is_corner = true;
            break;
        }
    }

    output[center_idx] = is_corner ? 255 : 0;
}

// ==========================================================
// CLASS IMPLEMENTATION
// ==========================================================

cv::Mat FASTDetectorCUDA::apply(const cv::Mat& input, const std::unordered_map<std::string, double>& parameters)  {
        auto itThreshold = parameters.find("threshold");
        if (itThreshold == parameters.end()) {
            throw std::runtime_error("Missing parameter: threshold");
        }

        int threshold = static_cast<int>(itThreshold->second);

        // 1. Convert to Grayscale on CPU
        cv::Mat gray;
        if (input.channels() == 3) {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        int width = gray.cols;
        int height = gray.rows;
        size_t img_size = width * height * sizeof(unsigned char);

        cv::Mat cornersMap = cv::Mat::zeros(height, width, CV_8UC1);

        // 2. Allocate Device Memory
        unsigned char *d_in, *d_out;
        cudaMalloc(&d_in, img_size);
        cudaMalloc(&d_out, img_size);

        // Initialize output to black
        cudaMemset(d_out, 0, img_size);

        // 3. Host to Device
        cudaMemcpy(d_in, gray.ptr(), img_size, cudaMemcpyHostToDevice);

        // 4. Fire Kernel
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        fastCornerKernel<<<grid, block>>>(d_in, d_out, width, height, threshold);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_in); cudaFree(d_out);
            throw std::runtime_error(std::string("CUDA FAST Error: ") + cudaGetErrorString(err));
        }

        // 5. Device to Host
        cudaMemcpy(cornersMap.ptr(), d_out, img_size, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);

        // 6. Draw the keypoints onto the original image (Just like your SIFT code!)
        cv::Mat output = input.clone();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (cornersMap.at<unsigned char>(y, x) == 255) {
                    // Draw a green circle at the corner
                    cv::circle(output, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), 1);
                }
            }
        }

        return output;
    }

// Auto-register to the factory
REGISTER_CUDA_FILTER(FASTDetectorCUDA, "fast_detector")