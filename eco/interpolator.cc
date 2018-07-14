#include "interpolator.hpp"

namespace eco{
Interpolator::Interpolator(){}

Interpolator::~Interpolator(){}

void Interpolator::get_interp_fourier(cv::Size filter_sz, 
									  cv::Mat &interp1_fs,
									  cv::Mat &interp2_fs, 
									  float a)
{
	cv::Mat temp1(filter_sz.height, 1, CV_32FC1);
	cv::Mat temp2(1, filter_sz.width, CV_32FC1);
	for (int j = 0; j < temp1.rows; j++)
	{
		temp1.at<float>(j, 0) = j - temp1.rows / 2;
	}
	for (int j = 0; j < temp2.cols; j++)
	{
		temp2.at<float>(0, j) = j - temp2.cols / 2;
	}

	interp1_fs = cubic_spline_fourier(temp1 / filter_sz.height, a) / filter_sz.height;
	interp2_fs = cubic_spline_fourier(temp2 / filter_sz.width, a) / filter_sz.width;

	// Multiply Fourier coeff with e ^ (-i*pi*k / N): [cos(pi*k/N), -sin(pi*k/N)]
	cv::Mat result1(temp1.size(), CV_32FC1), result2(temp1.size(), CV_32FC1);
	temp1 = temp1 / filter_sz.height;
	temp2 = temp2 / filter_sz.width;
	std::transform(temp1.begin<float>(), temp1.end<float>(), result1.begin<float>(), Interpolator::mat_cos1);
	std::transform(temp1.begin<float>(), temp1.end<float>(), result2.begin<float>(), Interpolator::mat_sin1);
	cv::Mat planes1[] = {interp1_fs.mul(result1), -interp1_fs.mul(result2)};
	cv::merge(planes1, 2, interp1_fs);

	interp2_fs = interp1_fs.t();
}

cv::Mat Interpolator::cubic_spline_fourier(cv::Mat f, float a)
{
	if (f.empty())
		return cv::Mat();

	cv::Mat bf(f.size(), CV_32FC1), 
			temp_cos2(f.size(), CV_32FC1), 
			temp_cos4(f.size(), CV_32FC1),
			temp_sin2(f.size(), CV_32FC1), 
			temp_sin4(f.size(), CV_32FC1);
	std::transform(f.begin<float>(), f.end<float>(), temp_cos2.begin<float>(), Interpolator::mat_cos2);
	std::transform(f.begin<float>(), f.end<float>(), temp_cos4.begin<float>(), Interpolator::mat_cos4);
	std::transform(f.begin<float>(), f.end<float>(), temp_sin2.begin<float>(), Interpolator::mat_sin2);
	std::transform(f.begin<float>(), f.end<float>(), temp_sin4.begin<float>(), Interpolator::mat_sin4);

	bf = 6 * (cv::Mat::ones(f.size(), CV_32FC1) - temp_cos2)
		 + 3 * a * (cv::Mat::ones(f.size(), CV_32FC1) - temp_cos4)
		 - (6 + a * 8) * CV_PI * f.mul(temp_sin2) 
		 - 2 * a * CV_PI * f.mul(temp_sin4) ;

	cv::Mat L(f.size(), CV_32FC1);
	cv::pow(f, 4, L);
	cv::divide(bf, 4 * L * cv::pow(CV_PI, 4), bf);
	bf.at<float>(bf.rows / 2, bf.cols / 2) = 1;

	return bf;
}
}