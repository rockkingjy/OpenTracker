
#include <stdio.h>
#include <string>
#include <vector>
#include <numeric>
#include <iostream>

#ifdef USE_CUDA
using namespace std;
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

int main(int argc, char *argv[])
{
    try
    {
        cv::cuda::printCudaDeviceInfo(0);

        cv::Mat src_host = cv::imread("./cvgputest.png", CV_LOAD_IMAGE_GRAYSCALE);
        
        // upload to GPU: Mat -> GpuMat
        cv::cuda::GpuMat dst, src;
        src.upload(src_host);

        double timer = (double)cv::getTickCount();
        float timedft = 0;

        cv::cuda::threshold(src, dst, 100.0, 255.0, CV_THRESH_BINARY);

        timedft = ((double)cv::getTickCount() - timer) / cv::getTickFrequency();
        debug("GPU time: %f", timedft);

        // download to CPU: GpuMat -> Mat
        cv::Mat result_host;
        dst.download(result_host);

        cv::Mat dst_host;
        timer = (double)cv::getTickCount();

        cv::threshold(src_host, dst_host, 100.0, 255.0, CV_THRESH_BINARY);

        timedft = ((double)cv::getTickCount() - timer) / cv::getTickFrequency();
        debug("CPU time: %f", timedft);


        cv::cuda::multiply(dst, dst, dst);
    }
    catch (const cv::Exception &ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
#endif