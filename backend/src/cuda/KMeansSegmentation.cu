#include "cuda/KMeansSegmentation.h" // Adjust to your interface path
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <vector>
#include <random>

// ==========================================================
// KERNEL 1: Assign each pixel to the nearest cluster
// ==========================================================
__global__ void assignClustersKernel(
    const unsigned char* input, int* labels, 
    const float* centroids, int width, int height, int step, int k) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * step + x * 3;
    float b = (float)input[idx];
    float g = (float)input[idx + 1];
    float r = (float)input[idx + 2];

    float min_dist = 1e9f;
    int best_cluster = 0;

    // Find the centroid with the shortest Euclidean distance in RGB space
    for (int i = 0; i < k; ++i) {
        float cb = centroids[i * 3];
        float cg = centroids[i * 3 + 1];
        float cr = centroids[i * 3 + 2];

        float dist = (b - cb)*(b - cb) + (g - cg)*(g - cg) + (r - cr)*(r - cr);

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = i;
        }
    }

    // Save the winning cluster ID
    labels[y * width + x] = best_cluster;
}

// ==========================================================
// KERNEL 2: Map the final image to the cluster colors
// ==========================================================
__global__ void applyClusterColorsKernel(
    unsigned char* output, const int* labels, 
    const float* centroids, int width, int height, int step) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int cluster_id = labels[y * width + x];
    int idx = y * step + x * 3;

    // Paint the pixel with the centroid's exact color
    output[idx]     = (unsigned char)centroids[cluster_id * 3];
    output[idx + 1] = (unsigned char)centroids[cluster_id * 3 + 1];
    output[idx + 2] = (unsigned char)centroids[cluster_id * 3 + 2];
}

// ==========================================================
// CLASS IMPLEMENTATION
// ==========================================================
cv::Mat KMeansSegmentationCUDA::apply(const cv::Mat& input, const std::unordered_map<std::string, double>& parameters){
        auto itK = parameters.find("k");
        auto itIter = parameters.find("iterations");

        if (itK == parameters.end() || itIter == parameters.end()) {
            throw std::runtime_error("Missing parameters: k or iterations");
        }

        int k = static_cast<int>(itK->second);
        int max_iter = static_cast<int>(itIter->second);

        if (k < 2 || k > 32) throw std::runtime_error("k must be between 2 and 32");
        if (max_iter < 1) throw std::runtime_error("iterations must be at least 1");
        if (input.type() != CV_8UC3) throw std::runtime_error("CUDA K-Means requires a 3-channel BGR image.");

        int width = input.cols;
        int height = input.rows;
        int total_pixels = width * height;
        
        size_t img_size = input.step * height;
        size_t labels_size = total_pixels * sizeof(int);
        size_t centroids_size = k * 3 * sizeof(float);

        cv::Mat output(height, width, CV_8UC3);

        // 1. Initialize Random Centroids on CPU
        std::vector<float> h_centroids(k * 3);
        std::mt19937 rng(42); // Fixed seed for reproducible results
        std::uniform_int_distribution<int> dist_w(0, width - 1);
        std::uniform_int_distribution<int> dist_h(0, height - 1);

        for (int i = 0; i < k; ++i) {
            int rx = dist_w(rng);
            int ry = dist_h(rng);
            int idx = ry * input.step + rx * 3;
            h_centroids[i * 3]     = input.data[idx];
            h_centroids[i * 3 + 1] = input.data[idx + 1];
            h_centroids[i * 3 + 2] = input.data[idx + 2];
        }

        // 2. Allocate Device Memory
        unsigned char *d_in, *d_out;
        int *d_labels;
        float *d_centroids;

        cudaMalloc(&d_in, img_size);
        cudaMalloc(&d_out, img_size);
        cudaMalloc(&d_labels, labels_size);
        cudaMalloc(&d_centroids, centroids_size);

        cudaMemcpy(d_in, input.ptr(), img_size, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        std::vector<int> h_labels(total_pixels);

        // 3. The Hybrid CPU-GPU Loop
        for (int iter = 0; iter < max_iter; ++iter) {
            
            // A. Send current centroids to GPU
            cudaMemcpy(d_centroids, h_centroids.data(), centroids_size, cudaMemcpyHostToDevice);

            // B. GPU calculates distances and assigns labels massively in parallel
            assignClustersKernel<<<grid, block>>>(d_in, d_labels, d_centroids, width, height, input.step, k);
            cudaDeviceSynchronize();

            // C. Download the labels back to the CPU
            cudaMemcpy(h_labels.data(), d_labels, labels_size, cudaMemcpyDeviceToHost);

            // D. CPU quickly calculates the new centroids (Averages the colors)
            std::vector<float> sum_b(k, 0.0f), sum_g(k, 0.0f), sum_r(k, 0.0f);
            std::vector<int> counts(k, 0);

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int cluster = h_labels[y * width + x];
                    int idx = y * input.step + x * 3;
                    
                    sum_b[cluster] += input.data[idx];
                    sum_g[cluster] += input.data[idx + 1];
                    sum_r[cluster] += input.data[idx + 2];
                    counts[cluster]++;
                }
            }

            // Update centroids
            for (int i = 0; i < k; ++i) {
                if (counts[i] > 0) {
                    h_centroids[i * 3]     = sum_b[i] / counts[i];
                    h_centroids[i * 3 + 1] = sum_g[i] / counts[i];
                    h_centroids[i * 3 + 2] = sum_r[i] / counts[i];
                }
            }
        }

        // 4. Final Color Mapping on GPU
        cudaMemcpy(d_centroids, h_centroids.data(), centroids_size, cudaMemcpyHostToDevice);
        applyClusterColorsKernel<<<grid, block>>>(d_out, d_labels, d_centroids, width, height, input.step);
        
        cudaDeviceSynchronize();
        cudaMemcpy(output.ptr(), d_out, img_size, cudaMemcpyDeviceToHost);

        // 5. Cleanup
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_labels); cudaFree(d_centroids);

        return output;
    }

// Auto-register to the factory
REGISTER_CUDA_FILTER(KMeansSegmentationCUDA, "kmeans_segmentation")