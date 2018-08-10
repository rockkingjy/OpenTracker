#include "readvideo.hpp"
bool ReadVideo::drawing_now_flag_;
bool ReadVideo::bbox_get_flag_;
cv::Rect2f ReadVideo::bbox_;
ReadVideo::ReadVideo(){};
ReadVideo::~ReadVideo(){};

void ReadVideo::IniRead(cv::Rect2f &bboxGroundtruth, cv::Mat &frame, std::string window_name, cv::VideoCapture &capture)
{
	ReadVideo::drawing_now_flag_ = false;
	ReadVideo::bbox_get_flag_ = false;
	//bool flag = false;
	// Register mouse callback
	cvNamedWindow(window_name.c_str(), CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(window_name.c_str(), ReadVideo::mouseHandler, NULL);
	
	cv::Mat temp;
	frame.copyTo(temp);
	while (!ReadVideo::bbox_get_flag_)
	{
		rectangle(frame, bbox_, cv::Scalar(0, 0, 255), 1);
		cv::imshow(window_name, frame);
		temp.copyTo(frame);
		int c = cvWaitKey(1);
		if (c == 27)
			break;
		if (c == 65)
		{
			printf("debug2\n");
			capture >> frame;
			frame.copyTo(temp);
			//cv::imshow(window_name, frame);
			//continue;
		}
	}
	// Remove callback
	cvSetMouseCallback(window_name.c_str(), NULL, NULL);
	printf("bbox:%d, %d, %d, %d\n", bbox_.x, bbox_.y, bbox_.width, bbox_.height);
	bboxGroundtruth.x = bbox_.x;
	bboxGroundtruth.y = bbox_.y;
	bboxGroundtruth.width = bbox_.width;
	bboxGroundtruth.height = bbox_.height;	
}
