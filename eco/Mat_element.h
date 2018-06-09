#ifndef MAT_ELEMENT_H
#define MAT_ELEMENT_H

#include <iostream>
#include <math.h>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#endif

inline float mat_cos1(float x)
{
	return (cos(x * 3.1415926));
}

inline float mat_sin1(float x)
{
	return (sin(x * 3.1415926));
}

inline float mat_cos2(float x)
{
	return (cos(2 * x * 3.1415926));
}

inline float mat_sin2(float x)
{
	return (sin(2 * x * 3.1415926));
}

inline float mat_cos4(float x)
{
	return (cos(4 * x * 3.1415926));
}

inline float mat_sin4(float x)
{
	return (sin(4 * x * 3.1415926));
}

cv::Mat precision(cv::Mat img);

inline cv::Mat precision(cv::Mat img)
{
	if (img.empty())
	{
		return img;
	}

	std::vector<cv::Mat> img_v;
	cv::split(img, img_v);

	for (size_t i = 0; i < img_v.size(); i++)
	{
		img_v[i].convertTo(img_v[i], CV_32FC1);
		for (size_t r = 0; r < (size_t)img_v[i].rows; r++)
		{
			for (size_t c = 0; c < (size_t)img_v[i].cols; c++)
			{
				if (std::abs(img_v[i].at<float>(r, c)) < 0.0000499999)
				{
					img_v[i].at<float>(r, c) = 0;
					continue;
				}
				if ((std::abs(img_v[i].at<float>(r, c)) > 0.0000499999) && (abs(img_v[i].at<float>(r, c)) < 0.0001))
				{
					img_v[i].at<float>(r, c) = 0.0001;
					continue;
				}
			}
		} //end for
	}

	cv::Mat result;
	cv::merge(img_v, result);
	return result;
}
