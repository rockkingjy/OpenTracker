#include "reg_filter.h"

cv::Mat  get_reg_filter(cv::Size sz, cv::Size2f target_sz, const eco_params& params)
{
	float reg_window_edge = params.reg_window_edge;
	//int reg_window_power = params.reg_window_power;
	cv::Size2f reg_scale = cv::Size2f(target_sz.width * 0.5, target_sz.height *0.5);

	// *** construct the regukarization window ***
	cv::Mat reg_window(sz, CV_32FC1);
	for (float x = -0.5*(sz.height - 1), counter1 = 0; counter1 < sz.height; x += 1, ++counter1)
		for (float y = -0.5*(sz.width - 1), counter2 = 0; counter2 < sz.width; y += 1, ++counter2)
			reg_window.at<float>(counter1, counter2) = (reg_window_edge - params.reg_window_min) *
			(pow((abs(x / reg_scale.height)), 2) + pow(abs(y / reg_scale.width), 2))
			+ params.reg_window_min;

	//**** find the max value and norm ****
	cv::Mat reg_window_dft = fftd(reg_window) / sz.area();
	cv::Mat reg_win_abs(sz, CV_32FC1);
	reg_win_abs = magnitude(reg_window_dft);
	double minv = 0.0, maxv = 0.0;;
	cv::minMaxLoc(reg_win_abs, &minv, &maxv);

	//*** set to zero while the element smaller than threshold ***
	for (size_t i = 0; i < (size_t)reg_window_dft.rows; i++)
		for (size_t j = 0; j < (size_t)reg_window_dft.cols; j++)
		{
			if (reg_win_abs.at<float>(i, j) < (params.reg_sparsity_threshold * maxv))
				reg_window_dft.at<cv::Vec<float, 2>>(i, j) = cv::Vec<float, 2>(0, 0);
		}
	cv::Mat reg_window_sparse = FFTTools::real(FFTTools::fftd(reg_window_dft, true));
	cv::minMaxLoc(magnitude(reg_window_sparse), &minv, &maxv);
	reg_window_dft.at<float>(0, 0) -= sz.area() * minv + params.reg_window_min;
	reg_window_dft = FFTTools::fftshift(reg_window_dft);

	cv::Mat tmp, result;
	for (size_t i = 0; i < (size_t)reg_window_dft.rows; i++)
	{
		for (size_t j = 0; j < (size_t)reg_window_dft.cols; j++)
		{
			if (((reg_window_dft.at<cv::Vec<float, 2>>(i, j) != cv::Vec<float, 2>(0, 0)) &&
				(reg_window_dft.at<cv::Vec<float, 2>>(i, j) != cv::Vec<float, 2>(2, 0))))
			{
				tmp.push_back(reg_window_dft.row(i));
				break;
			}
		} //end for
	}//end for

	tmp = tmp.t();
	for (size_t i = 0; i < (size_t)tmp.rows; i++)
	{
		for (size_t j = 0; j < (size_t)tmp.cols; j++)
		{
			if (((tmp.at<cv::Vec<float, 2>>(i, j) != cv::Vec<float, 2>(0, 0)) &&
				(tmp.at<cv::Vec<float, 2>>(i, j) != cv::Vec<float, 2>(1, 0))))
			{
				result.push_back(FFTTools::real(tmp.row(i)));
				break;
			}
		} //end for
	}//end for
	result = result.t();

	return result;
}