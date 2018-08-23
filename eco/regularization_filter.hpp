#ifndef REGULARIZATION_FILTER_HPP
#define REGULARIZATION_FILTER_HPP

#include <opencv2/opencv.hpp>
#include <cmath>

#include "parameters.hpp"
#include "ffttools.hpp"
#include "debug.hpp"

namespace eco
{
cv::Mat get_regularization_filter(cv::Size sz,
                                  cv::Size2f target_sz,
                                  const EcoParameters &params);
}
#endif