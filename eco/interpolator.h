#ifndef INTERPOLATOR
#define INTERPOLATOR

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "Mat_element.h"

#endif

class interpolator
{
public:

	interpolator();

	virtual ~interpolator();
	 
	static void  get_interp_fourier(cv::Size filter_sz, cv::Mat& interp1_fs, cv::Mat& interp2_fs, float a);

	static cv::Mat  cubic_spline_fourier(cv::Mat f, float a);
	  
};
