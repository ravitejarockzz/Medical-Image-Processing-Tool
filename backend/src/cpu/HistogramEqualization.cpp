#include "cpu/HistogramEqualization.h"
#include "factory/FilterFactory.h"

#include <opencv2/imgproc.hpp>
#include <vector>

cv::Mat HistogramEqualizationCPU::apply(
    const cv::Mat& input,
    const std::unordered_map<std::string, double>& parameters
) {
    // Notice: We don't extract any parameters here because this algorithm is fully automatic!

    cv::Mat output;

    if (input.channels() == 1) {
        // If it's already a grayscale image (like an X-ray), just apply directly
        cv::equalizeHist(input, output);
    } 
    else if (input.channels() == 3) {
        // If it's a color image, applying equalization directly to RGB ruins the colors.
        // Instead, we convert to YCrCb, equalize ONLY the 'Y' (brightness) channel, and convert back.
        cv::Mat ycrcb;
        cv::cvtColor(input, ycrcb, cv::COLOR_BGR2YCrCb);
        
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        
        // Equalize the brightness channel
        cv::equalizeHist(channels[0], channels[0]);
        
        // Merge back and convert to BGR
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, output, cv::COLOR_YCrCb2BGR);
    } 
    else {
        // Fallback for 4-channel (RGBA) or unusual formats
        output = input.clone(); 
    }

    return output;
}

// Auto-register to the factory
REGISTER_CPU_FILTER(HistogramEqualizationCPU, "histogram_equalization")    