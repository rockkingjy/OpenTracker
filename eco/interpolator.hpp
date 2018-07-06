#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "mat_element.hpp"


namespace eco{
class Interpolator
{
  public:
	Interpolator();

	virtual ~Interpolator();

	static void get_interp_fourier(cv::Size filter_sz, cv::Mat &interp1_fs,
								   cv::Mat &interp2_fs, float a);

	static cv::Mat cubic_spline_fourier(cv::Mat f, float a);
};
}

#endif