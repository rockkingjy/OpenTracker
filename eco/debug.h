#pragma once

#include <stdio.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define getName(var) #var

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)
#define ddebug(a, args...) //printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

using namespace std;

// Show the type of an image
// Using like this:
//	string ty =  type2str( im.type() );
//	printf("im: %s %d x %d \n", ty.c_str(), im.cols, im.rows );
void imgInfo(cv::Mat mat);
inline void imgInfo(cv::Mat mat)
{
	int type = mat.type();
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth)
	{
	case CV_8U:
		r = "8U";
		break;
	case CV_8S:
		r = "8S";
		break;
	case CV_16U:
		r = "16U";
		break;
	case CV_16S:
		r = "16S";
		break;
	case CV_32S:
		r = "32S";
		break;
	case CV_32F:
		r = "32F";
		break;
	case CV_64F:
		r = "64F";
		break;
	default:
		r = "User";
		break;
	}

	r += "C";
	r += (chans + '0');

	debug("%s %d x %d", r.c_str(), mat.rows, mat.cols);
	//return r;
}

void showmat1chall(cv::Mat mat, int type);
inline void showmat1chall(cv::Mat mat, int type)
{
	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			if (type == 1)
			{ // int
				printf("%d ", mat.at<int>(i, j));
			}
			else if (type == 2)
			{ //float
				printf("%f ", mat.at<float>(i, j));
			}
			else if (type == 3)
			{ //double
				printf("%lf ", mat.at<double>(i, j));
			}
		}
		printf("\n");
	}
	//printf("End of mat\n");
}

void showmat2chall(cv::Mat mat, int type);

inline void showmat2chall(cv::Mat mat, int type)
{
	debug("channels: %d, just print the 1st channel:", mat.channels());
	for (int k = 0; k < mat.channels(); k++)
	{
		for (int i = 0; i < mat.rows; i++)
		{
			for (int j = 0; j < mat.cols; j++)
			{
				if (type == 0)
				{ // char
					printf("%d ", mat.at<cv::Vec3b>(i, j)[k]);
				}
				else if (type == 1)
				{ // int
					printf("%d ", mat.at<cv::Vec2i>(i, j)[k]);
				}
				else if (type == 2)
				{ //float
					printf("%f ", mat.at<cv::Vec2f>(i, j)[k]);
				}
				else if (type == 3)
				{ //float
					printf("%lf ", mat.at<cv::Vec2d>(i, j)[k]);
				}
			}
			printf("\n");
		}
		printf("\n\n");
		break;
	}
	printf("End of 2 channel mat\n");
}

void showmat3chall(cv::Mat mat, int type);

inline void showmat3chall(cv::Mat mat, int type)
{
	for (int k = 0; k < mat.channels(); k++)
	{
		for (int i = 0; i < mat.rows; i++)
		{
			for (int j = 0; j < mat.cols; j++)
			{
				if (type == 0)
				{ // char
					printf("%d ", mat.at<cv::Vec3b>(i, j)[k]);
				}
				else if (type == 1)
				{ // int
					printf("%d ", mat.at<cv::Vec3i>(i, j)[k]);
				}
				else if (type == 2)
				{ //float
					printf("%f ", mat.at<cv::Vec3f>(i, j)[k]);
				}
				else if (type == 3)
				{ //float
					printf("%lf ", mat.at<cv::Vec3d>(i, j)[k]);
				}
			}
			printf("\n");
		}
		printf("\n\n");
		break;
	}
	printf("End of 3 channel mat\n");
}

void showmat1ch(cv::Mat mat, int type);
inline void showmat1ch(cv::Mat mat, int type)
{
	printf("\nFirst row: \n");
	for (int j = 0; j < mat.cols; j++)
	{
		if (type == 1)
		{ // int
			printf("%d ", mat.at<int>(0, j));
		}
		else if (type == 2)
		{ //float
			printf("%f ", mat.at<float>(0, j));
		}
		else if (type == 3)
		{ // first row
			printf("%lf ", mat.at<double>(0, j));
		}
	}
	printf("\nrow %d: \n", mat.cols - 1);
	for (int j = 0; j < mat.cols; j++)
	{
		if (type == 1)
		{ // last row
			printf("%d ", mat.at<int>(mat.cols - 1, j));
		}
		else if (type == 2)
		{ // last row
			printf("%f ", mat.at<float>(mat.cols - 1, j));
		}
		else if (type == 3)
		{ // last row
			printf("%lf ", mat.at<double>(mat.cols - 1, j));
		}
	}
	printf("\nFirst col: \n");
	for (int i = 0; i < mat.rows; i++)
	{
		if (type == 1)
		{ // first col
			printf("%d ", mat.at<int>(i, 0));
		}
		else if (type == 2)
		{ // first col
			printf("%f ", mat.at<float>(i, 0));
		}
		else if (type == 3)
		{ // first col
			printf("%lf ", mat.at<double>(i, 0));
		}
	}
	printf("\ncol %d: \n", mat.rows - 1);
	for (int i = 0; i < mat.rows; i++)
	{
		if (type == 1)
		{ // last col
			printf("%d ", mat.at<int>(i, mat.rows - 1));
		}
		else if (type == 2)
		{ // last col
			printf("%f ", mat.at<float>(i, mat.rows - 1));
		}
		else if (type == 3)
		{ // last col
			printf("%lf ", mat.at<double>(i, mat.rows - 1));
		}
	}
	printf("\nEnd of 2 channel mat\n");
}

void showmat2ch(cv::Mat mat, int type);
inline void showmat2ch(cv::Mat mat, int type)
{
	printf("First row: \n");
	for (int k = 0; k < mat.channels(); k++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			if (type == 3)
			{ // first row
				printf("%lf ", mat.at<cv::Vec2d>(0, j)[k]);
			}
		}
		printf("\n\n");
	}
	printf("\n");
	printf("row %d: \n", mat.cols - 1);
	for (int k = 0; k < mat.channels(); k++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			if (type == 3)
			{ // last row
				printf("%lf ", mat.at<cv::Vec2d>(mat.cols - 1, j)[k]);
			}
		}
		printf("\n\n");
	}
	printf("\n");
	printf("First col: \n");
	for (int k = 0; k < mat.channels(); k++)
	{
		for (int i = 0; i < mat.rows; i++)
		{
			if (type == 3)
			{ // first col
				printf("%lf ", mat.at<cv::Vec2d>(i, 0)[k]);
			}
		}
		printf("\n\n");
	}
	printf("\n");
	printf("col %d: \n", mat.rows - 1);
	for (int k = 0; k < mat.channels(); k++)
	{
		for (int i = 0; i < mat.rows; i++)
		{
			if (type == 3)
			{ // last col
				printf("%lf ", mat.at<cv::Vec2d>(i, mat.rows - 1)[k]);
			}
		}
		printf("\n\n");
	}
	printf("End of 2 channel mat\n");
}

// Attention!!!! opencv/caffe: BGR, matlab: RGB, different order!!!
void showmat3ch(cv::Mat mat, int type);
inline void showmat3ch(cv::Mat mat, int type)
{
	std::vector<cv::Mat> splitmat;
	cv::split(mat, splitmat);

	debug("channels: %lu", splitmat.size());

	printf("First row of channel 0: \n");
	for (int j = 0; j < mat.cols; j += 1)
	{
		if (type == 0)
		{
			printf("%d ", splitmat[2].at<uchar>(0, j));
		}
		else if (type == 1)
		{ // first row
			printf("%d ", splitmat[2].at<int>(0, j));
		}
		else if (type == 2)
		{ // first row
			printf("%f ", splitmat[0].at<float>(0, j));
		}
		else if (type == 3)
		{ // first row
			printf("%lf ", splitmat[2].at<double>(0, j));
		}
	}
	printf("\n\n");

	printf("\n");
	printf("First col of  channel 0: \n");
	for (int i = 0; i < mat.rows; i += 1)
	{
		if (type == 0)
		{ // first row
			printf("%d ", splitmat[2].at<uchar>(i, 0));
		}
		else if (type == 1)
		{ // first row
			printf("%d ", splitmat[2].at<int>(i, 0));
		}
		else if (type == 2)
		{ // first row
			printf("%f ", splitmat[2].at<float>(i, 0));
		}
		else if (type == 3)
		{ // first col
			printf("%lf ", splitmat[2].at<double>(i, 0));
		}
	}
	printf("\n\n");

	printf("End of feature mat\n");
}

void showmatNch(cv::Mat mat, int type);
inline void showmatNch(cv::Mat mat, int type)
{
	std::vector<cv::Mat> splitmat;
	cv::split(mat, splitmat);

	debug("channels: %lu", splitmat.size());

	printf("First row of channel 0: \n");
	for (int j = 0; j < mat.cols; j += 1)
	{
		if (type == 0)
		{
			printf("%d ", splitmat[0].at<uchar>(0, j));
		}
		else if (type == 1)
		{ // first row
			printf("%d ", splitmat[0].at<int>(0, j));
		}
		else if (type == 2)
		{ // first row
			printf("%f ", splitmat[0].at<float>(0, j));
		}
		else if (type == 3)
		{ // first row
			printf("%lf ", splitmat[0].at<double>(0, j));
		}
	}
	printf("\n\n");

	printf("\n");
	printf("First col of  channel 0: \n");
	for (int i = 0; i < mat.rows; i += 1)
	{
		if (type == 0)
		{ // first row
			printf("%d ", splitmat[0].at<uchar>(i, 0));
		}
		else if (type == 1)
		{ // first row
			printf("%d ", splitmat[0].at<int>(i, 0));
		}
		else if (type == 2)
		{ // first row
			printf("%f ", splitmat[0].at<float>(i, 0));
		}
		else if (type == 3)
		{ // first col
			printf("%lf ", splitmat[0].at<double>(i, 0));
		}
	}
	printf("\n\n");

	printf("End of feature mat\n");
}

//TEST=============================================================================================
// Simple test of the structure of mat in opencv;
void opencvTest();
inline void opencvTest()
{
	printf("opencvTest begin=======================================\n");
	float *newdata = (float *)malloc(sizeof(float) * (2 * 3 * 4));

	for (int i = 0; i < 2 * 3 * 4; i++)
	{
		newdata[i] = i;
	}

	cv::Mat mat = cv::Mat(cv::Size(3, 4), CV_32FC(2), newdata);

	printf("\nInfo of original mat:");
	imgInfo(mat);
	for (int i = 0; i < 2 * 3 * 4; i++)
	{
		printf("%f ", mat.at<float>(0, i));
	}
	printf("\n");

	std::vector<cv::Mat> splitmat;
	cv::split(mat, splitmat);

	printf("\nInfo of splited mat:");
	imgInfo(splitmat[0]);

	printf("channel 0:\n");
	for (int j = 0; j < mat.rows; j++)
	{
		for (int i = 0; i < mat.cols; i++)
		{
			printf("%f ", splitmat[0].at<float>(j, i));
		}
		printf("\n");
	}
	printf("\n");
	printf("channel 1:\n");
	for (int j = 0; j < mat.rows; j++)
	{
		for (int i = 0; i < mat.cols; i++)
		{
			printf("%f ", splitmat[1].at<float>(j, i));
		}
		printf("\n");
	}
	printf("opencvTest end=======================================\n");
}

void absTest();
inline void absTest()
{
	printf("absTest begin=======================================\n");
	std::vector<float> v{0.1, 0.2};
	//sometimes this works:
	//```
	//float abs = abs(1.23f);
	//```
	//but it use a different liberay from eigen, cause error
	//so remember to add `std::` before!
	float abs = std::abs(1.23f);
	debug("False abs:%f", abs);

	abs = std::abs(1.23f);
	debug("True abs:%f", abs);
	printf("absTest end=======================================\n");
}

void accumulateTest();
inline void accumulateTest()
{
	printf("accumulateTest begin=======================================\n");
	std::vector<float> v{0.1, 0.2};
	float sum = std::accumulate(v.begin(), v.end(), 0);
	debug("False sum:%f", sum);

	sum = std::accumulate(v.begin(), v.end(), 0.0f);
	debug("True sum:%f", sum);
	printf("accumulateTest end=======================================\n");
}
