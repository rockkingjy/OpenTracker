#include "interpolator.hpp"

namespace eco
{
Interpolator::Interpolator() {}

Interpolator::~Interpolator() {}

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

	// Multiply Fourier coeff with e ^ (-i*pi*k / N):[cos(pi*k/N), -sin(pi*k/N)]
	cv::Mat result1(temp1.size(), CV_32FC1), result2(temp1.size(), CV_32FC1);
	temp1 = temp1 / filter_sz.height;
	temp2 = temp2 / filter_sz.width;
	std::transform(temp1.begin<float>(), temp1.end<float>(), result1.begin<float>(), Interpolator::mat_cos1);
	std::transform(temp1.begin<float>(), temp1.end<float>(), result2.begin<float>(), Interpolator::mat_sin1);

	//cv::Mat planes1[] = {interp1_fs.mul(result1), -interp1_fs.mul(result2)};
	//cv::merge(planes1, 2, interp1_fs);
	//interp2_fs = interp1_fs.t();
	cv::Mat temp = cv::Mat(interp1_fs.size(), CV_32FC2);
	cv::Mat tempT = cv::Mat(interp1_fs.cols, interp1_fs.rows, CV_32FC2);
	for(int r = 0; r < temp1.rows; r++)
	{
		for(int c = 0; c < temp1.cols; c++)
		{

			temp.at<cv::Vec2f>(r, c)[0] = interp1_fs.at<float>(r, c) * result1.at<float>(r, c);
			temp.at<cv::Vec2f>(r, c)[1] = -interp1_fs.at<float>(r, c) * result2.at<float>(r, c);
			tempT.at<cv::Vec2f>(c, r)[0] = temp.at<cv::Vec2f>(r, c)[0];
			tempT.at<cv::Vec2f>(c, r)[1] = temp.at<cv::Vec2f>(r, c)[1];
		}
	}
	interp1_fs = temp;
	interp2_fs = tempT;
}

cv::Mat Interpolator::cubic_spline_fourier(cv::Mat f, float a)
{
	if (f.empty())
	{
		assert(0 && "error: input mat is empty!");
	}
/*
	cv::Mat bf(f.size(), CV_32FC1),
		temp_cos2(f.size(), CV_32FC1),
		temp_cos4(f.size(), CV_32FC1),
		temp_sin2(f.size(), CV_32FC1),
		temp_sin4(f.size(), CV_32FC1);
	std::transform(f.begin<float>(), f.end<float>(), temp_cos2.begin<float>(), Interpolator::mat_cos2);
	std::transform(f.begin<float>(), f.end<float>(), temp_cos4.begin<float>(), Interpolator::mat_cos4);
	std::transform(f.begin<float>(), f.end<float>(), temp_sin2.begin<float>(), Interpolator::mat_sin2);
	std::transform(f.begin<float>(), f.end<float>(), temp_sin4.begin<float>(), Interpolator::mat_sin4);

	bf = 6 * (cv::Mat::ones(f.size(), CV_32FC1) - temp_cos2) + 3 * a * (cv::Mat::ones(f.size(), CV_32FC1) - temp_cos4) - (6 + a * 8) * CV_PI * f.mul(temp_sin2) - 2 * a * CV_PI * f.mul(temp_sin4);

	cv::Mat L(f.size(), CV_32FC1);
	cv::pow(f, 4, L);
	cv::divide(bf, 4 * L * cv::pow(CV_PI, 4), bf);
	bf.at<float>(bf.rows / 2, bf.cols / 2) = 1;
*/
	cv::Mat bf(f.size(), CV_32FC1);
	for(int r = 0; r < bf.rows; r++)
	{
		for(int c = 0; c < bf.cols; c++)
		{
			bf.at<float>(r, c) = 6.0f * (1 - cos(2.0f * f.at<float>(r, c) * M_PI)) 
			+ 3.0f * a * (1.0f - cos(4 * f.at<float>(r, c) * M_PI))
			- (6.0f + a * 8.0f) * M_PI * f.at<float>(r, c) * sin(2.0f * f.at<float>(r, c) * M_PI)
			- 2.0f * a * M_PI * f.at<float>(r, c) * sin(4.0f * f.at<float>(r, c) * M_PI);
			float L = 4.0f * pow(f.at<float>(r, c) * M_PI, 4);
			bf.at<float>(r, c) /= L;
		}
	}
	bf.at<float>(bf.rows / 2, bf.cols / 2) = 1;
	//printMat(bf);
	//showmat1channels(bf, 2);

	return bf;
}
} // namespace eco