#include "interpolator.h"

interpolator::interpolator()
{
}

interpolator::~interpolator()
{
}


void  interpolator::get_interp_fourier(cv::Size filter_sz, cv::Mat& interp1_fs, cv::Mat& interp2_fs, float a)
{
	cv::Mat temp1(filter_sz.height, 1, CV_32FC1);
	cv::Mat temp2(1, filter_sz.width, CV_32FC1);
	for (int j = 0; j < temp1.rows; j++)
	{
		//wangsen
		//temp1.data[j] = j - temp1.rows / 2;
		//temp2.data[j] = j - temp1.rows / 2;
		temp1.at<float>(j, 0) = j - temp1.rows / 2;
		//wangsen why this is temp1.rows not temp1.cols
		temp2.at<float>(0, j) = j - temp1.rows / 2;
	}

	interp1_fs = cubic_spline_fourier(temp1 / filter_sz.height, a) / filter_sz.height;
	//wangsen interp2_fs定义的作用
	interp2_fs = cubic_spline_fourier(temp2 / filter_sz.width, a) / filter_sz.width;

	// ***Center the feature grids by shifting the interpolated features
	//*** Multiply Fourier coeff with e ^ (-i*pi*k / N)

	cv::Mat result1(temp1.size(), CV_32FC1), result2(temp1.size(), CV_32FC1);
	temp1 = temp1 / filter_sz.height; temp2 = temp2 / filter_sz.width;
	std::transform(temp1.begin<float>(), temp1.end<float>(), result1.begin<float>(), mat_cos1);
	std::transform(temp1.begin<float>(), temp1.end<float>(), result2.begin<float>(), mat_sin1);
	cv::Mat planes1[] = { interp1_fs.mul(result1), interp1_fs.mul(result2) };
	cv::merge(planes1, 2, interp1_fs);

	interp2_fs = interp1_fs.t();

}

cv::Mat interpolator::cubic_spline_fourier(cv::Mat f, float a)
{
	if (f.empty())
		return cv::Mat();

	cv::Mat bf(f.size(), CV_32FC1), temp1(f.size(), CV_32FC1), temp2(f.size(), CV_32FC1),
		temp3(f.size(), CV_32FC1), temp4(f.size(), CV_32FC1);
	std::transform(f.begin<float>(), f.end<float>(), temp1.begin<float>(), mat_cos2);
	std::transform(f.begin<float>(), f.end<float>(), temp2.begin<float>(), mat_cos4);

	std::transform(f.begin<float>(), f.end<float>(), temp3.begin<float>(), mat_sin2);
	std::transform(f.begin<float>(), f.end<float>(), temp4.begin<float>(), mat_sin4);

	bf = -1 * (-12 * a * cv::Mat::ones(f.size(), CV_32FC1) + 24 * temp1 +
		12 * a * temp2 + CV_PI * 24 * f.mul(temp3) +
		CV_PI * a * 32 * f.mul(temp3) + CV_PI * 8 * a * f.mul(temp4) -
		24 * cv::Mat::ones(f.size(), CV_32FC1));

	cv::Mat L(f.size(), CV_32FC1);
	cv::pow(f, 4, L);
	cv::divide(bf, 16 * L * cv::pow(CV_PI, 4), bf);
	bf.at<float>(bf.rows / 2, bf.cols / 2) = 1;

	return bf;
}