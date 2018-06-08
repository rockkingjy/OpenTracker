#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)
#define ddebug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

void showmat(cv::Mat mat, int type);
inline void showmat(cv::Mat mat, int type)
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

void showmat2ch(cv::Mat mat, int type);

inline void showmat2ch(cv::Mat mat, int type)
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
			}
			printf("\n");
		}
		printf("\n\n");
	}
	printf("End of 2 channel mat\n");
}