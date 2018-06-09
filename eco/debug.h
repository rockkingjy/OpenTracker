#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)
#define ddebug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

void showmatall(cv::Mat mat, int type);
inline void showmatall(cv::Mat mat, int type)
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
	printf("End of mat\n");
}

void showmat2chall(cv::Mat mat, int type);

inline void showmat2chall(cv::Mat mat, int type)
{
	for (int k = 0; k < mat.channels(); k++)
	{
		for (int i = 0; i < mat.rows; i++)
		{
			for (int j = 0; j < mat.cols; j++)
			{
				if (type == 1)
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

void showmat(cv::Mat mat, int type);

inline void showmat(cv::Mat mat, int type)
{
	printf("\nFirst row: \n");
	for (int j = 0; j < mat.cols; j++)
	{
		if (type == 3)
		{ // first row
			printf("%lf ", mat.at<double>(0, j));
		}
	}
	printf("\nrow %d: \n", mat.cols - 1);
	for (int j = 0; j < mat.cols; j++)
	{
		if (type == 3)
		{ // last row
			printf("%lf ", mat.at<double>(mat.cols - 1, j));
		}
	}
	printf("\nFirst col: \n");
	for (int i = 0; i < mat.rows; i++)
	{
		if (type == 3)
		{ // first col
			printf("%lf ", mat.at<double>(i, 0));
		}
	}
	printf("\ncol %d: \n", mat.rows - 1);
	for (int i = 0; i < mat.rows; i++)
	{
		if (type == 3)
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
