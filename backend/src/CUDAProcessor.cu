#include "CUDAProcessor.h"
#include "CPUProcessor.h"

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// -------------------------------------
// CUDA Gaussian Blur Kernel
// -------------------------------------
__global__
void gaussianBlurKernel(unsigned char* input,
                        unsigned char* output,
                        int width,
                        int height,
                        int channels,
                        int kernelRadius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int kernelSize = 2 * kernelRadius + 1;
    int area = kernelSize * kernelSize;

    for (int c = 0; c < channels; ++c)
    {
        int sum = 0;

        for (int ky = -kernelRadius; ky <= kernelRadius; ++ky)
        {
            for (int kx = -kernelRadius; kx <= kernelRadius; ++kx)
            {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);

                int idx = (ny * width + nx) * channels + c;
                sum += input[idx];
            }
        }

        int outIdx = (y * width + x) * channels + c;
        output[outIdx] = sum / area;
    }
}

// -------------------------------------
// Utilities
// -------------------------------------
static cv::Mat decodeImage(const std::string& imageData)
{
    std::vector<unsigned char> buffer(imageData.begin(), imageData.end());
    cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (img.empty())
        throw std::runtime_error("Failed to decode image");

    return img;
}

static std::vector<unsigned char> encodePNG(const cv::Mat& image)
{
    std::vector<unsigned char> buffer;
    cv::imencode(".png", image, buffer);
    return buffer;
}

// -------------------------------------
// Gaussian Blur (CUDA)
// -------------------------------------
std::vector<unsigned char> CUDAProcessor::applyGaussianBlur(
    const std::string& imageData,
    int kernel)
{
    cv::Mat img = decodeImage(imageData);

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    size_t size = width * height * channels;

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, img.data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    int radius = kernel / 2;

    gaussianBlurKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height,
        channels,
        radius);

    cudaDeviceSynchronize();

    cv::Mat result(height, width, CV_8UC3);

    cudaMemcpy(result.data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return encodePNG(result);
}

// -------------------------------------
// Canny (Fallback to CPU)
// -------------------------------------
std::vector<unsigned char> CUDAProcessor::applyCanny(
    const std::string& imageData,
    int low,
    int high)
{
    CPUProcessor cpu;
    return cpu.applyCanny(imageData, low, high);
}

// -------------------------------------
// Grayscale Conversion (Fallback to CPU)   
// -------------------------------------
std::vector<unsigned char> CUDAProcessor::applyGrayscale(
    const std::string& imageData)
{
    CPUProcessor cpu;
    return cpu.applyGrayscale(imageData);
}