#ifndef REG_FILTER
#define REG_FILTER

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "params.h"
#include "fftTool.h"
#include "debug.h"

using FFTTools::fftd;
using FFTTools::magnitude;

cv::Mat  get_reg_filter(cv::Size sz, cv::Size2f target_sz, const eco_params& params);

#endif 