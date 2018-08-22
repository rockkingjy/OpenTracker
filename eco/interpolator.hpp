#ifndef INTERPOLATOR_HPP
#define INTERPOLATOR_HPP

#include <opencv2/opencv.hpp>
#include <math.h>
#include "debug.hpp"

namespace eco{
class Interpolator
{
  public:
	Interpolator();
	virtual ~Interpolator();
	static inline float mat_cos1(float x)
	{
		return (cos(x * M_PI));
	}
	static inline float mat_sin1(float x)
	{
		return (sin(x * M_PI));
	}
	static inline float mat_cos2(float x)
	{
		return (cos(2 * x * M_PI));
	}
	static inline float mat_sin2(float x)
	{
		return (sin(2 * x * M_PI));
	}
	static inline float mat_cos4(float x)
	{
		return (cos(4 * x * M_PI));
	}
	static inline float mat_sin4(float x)
	{
		return (sin(4 * x * M_PI));
	}

	static void get_interp_fourier(cv::Size filter_sz, 
								   cv::Mat &interp1_fs,
								   cv::Mat &interp2_fs, 
								   float a);

	static cv::Mat cubic_spline_fourier(cv::Mat f, float a);
};
}

#endif