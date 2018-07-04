#include "regularization_filter.hpp"

namespace eco{
cv::Mat get_regularization_filter(cv::Size sz, cv::Size2f target_sz, const EcoParameters &params)
{
	cv::Mat result;

	if (params.use_reg_window)
	{
		cv::Size2d reg_scale = cv::Size2d(target_sz.width * 0.5, target_sz.height * 0.5);

		// *** construct the regukarization window ***
		cv::Mat reg_window(sz, CV_64FC1);
		for (double x = -0.5 * (sz.height - 1), counter1 = 0; counter1 < sz.height; x += 1, ++counter1)
			for (double y = -0.5 * (sz.width - 1), counter2 = 0; counter2 < sz.width; y += 1, ++counter2)
			{	// use abs() directly will cause error because it returns int!!! 
				reg_window.at<double>(counter1, counter2) = (params.reg_window_edge - params.reg_window_min) *
															   (std::pow(std::abs(x / reg_scale.height), params.reg_window_power) +
																std::pow(std::abs(y / reg_scale.width), params.reg_window_power)) +
														   params.reg_window_min;
			}
		//debug("%f %f", reg_scale.height, reg_scale.width);
		//debug("%d %d", sz.height, sz.width);
		//showmat(reg_window,3);
		//debug("Channels: %d",reg_window.channels());
		//assert(0);

		// compute the DFT and enforce sparsity
		cv::Mat reg_window_dft = fftd(reg_window) / sz.area();
		cv::Mat reg_win_abs(sz, CV_64FC1);
		reg_win_abs = magnitude(reg_window_dft);

		double minv = 0.0, maxv = 0.0;
		cv::minMaxLoc(reg_win_abs, &minv, &maxv);
		// set to zero while the element smaller than threshold
		for (size_t i = 0; i < (size_t)reg_window_dft.rows; i++)
			for (size_t j = 0; j < (size_t)reg_window_dft.cols; j++)
			{
				if (reg_win_abs.at<double>(i, j) < (params.reg_sparsity_threshold * maxv))
					reg_window_dft.at<cv::Vec<double, 2>>(i, j) = cv::Vec<double, 2>(0, 0);
			}

		// do the inverse transform, correct window minimum
		cv::Mat reg_window_sparse = real(fftd(reg_window_dft, true));
		cv::minMaxLoc(magnitude(reg_window_sparse), &minv, &maxv);
		reg_window_dft.at<double>(0, 0) -= sz.area() * minv + params.reg_window_min;
		reg_window_dft = fftshiftd(reg_window_dft);
		
		//showmat2ch(reg_window_dft,3);
		//debug("Channels: %d",reg_window_dft.channels());
		//debug("%lf", minv);
		//assert(0);

		// find the regularization filter by removing the zeros
		cv::Mat tmp;
		for (size_t i = 0; i < (size_t)reg_window_dft.rows; i++)
		{
			for (size_t j = 0; j < (size_t)reg_window_dft.cols; j++)
			{
				if (((reg_window_dft.at<cv::Vec<double, 2>>(i, j) != cv::Vec<double, 2>(0, 0)) &&
					 (reg_window_dft.at<cv::Vec<double, 2>>(i, j) != cv::Vec<double, 2>(2, 0))))
				{
					tmp.push_back(reg_window_dft.row(i));
					break;
				}
			} //end for
		}	 //end for

		tmp = tmp.t();
		for (size_t i = 0; i < (size_t)tmp.rows; i++)
		{
			for (size_t j = 0; j < (size_t)tmp.cols; j++)
			{
				if (((tmp.at<cv::Vec<double, 2>>(i, j) != cv::Vec<double, 2>(0, 0)) &&
					 (tmp.at<cv::Vec<double, 2>>(i, j) != cv::Vec<double, 2>(1, 0))))
				{
					result.push_back(real(tmp.row(i)));
					break;
				}
			} //end for
		}	 //end for
		result = result.t();
	}
	else
	{
		result.push_back(params.reg_window_min);
	}

	return result;
}
}