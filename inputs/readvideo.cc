#include "readvideo.hpp"
bool ReadVideo::drawing_now_flag_;
bool ReadVideo::bbox_get_flag_;
cv::Rect2f ReadVideo::bbox_;
ReadVideo::ReadVideo(){};
ReadVideo::~ReadVideo(){};

void ReadVideo::IniRead(cv::Rect2f &bboxGroundtruth, cv::Mat &frame, std::string window_name)
{
	ReadVideo::drawing_now_flag_ = false;
	ReadVideo::bbox_get_flag_ = false;

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
		if (cvWaitKey(20) == 27)
			break;
	}
	// Remove callback
	cvSetMouseCallback(window_name.c_str(), NULL, NULL);
	printf("bbox:%d, %d, %d, %d\n", bbox_.x, bbox_.y, bbox_.width, bbox_.height);
	bboxGroundtruth.x = bbox_.x;
	bboxGroundtruth.y = bbox_.y;
	bboxGroundtruth.width = bbox_.width;
	bboxGroundtruth.height = bbox_.height;
}

/* read directory
while ((entry = readdir(dir)) != NULL)
{
	string img_name = entry->d_name;
	if (*(img_name.end() - 1) == 'g')
	{
		frame = cv::imread(path + "\\" + img_name);
		if (id++ == 0)
		{
			cvNamedWindow(WIN_NAME.c_str(), CV_WINDOW_AUTOSIZE);
			cvSetMouseCallback(WIN_NAME.c_str(), mouseHandler, NULL);

			//cv::resize(frame, frame, cv::Size(frame.cols / 2, frame.rows / 2));
			cv::Mat temp;
			frame.copyTo(temp);
			while (!gotBB)
			{
				drawBox(frame, box, cv::Scalar(0, 0, 255), 1);
				imshow(WIN_NAME, frame);
				temp.copyTo(frame);
				if (cvWaitKey(20) == 27)
					return 1;
			}
			//Remove callback
			cvSetMouseCallback(WIN_NAME.c_str(), NULL, NULL);

			//Convert im0 to grayscale
			cv::Mat im0_gray;
			if (frame.channels() > 1)
			{
				cvtColor(frame, im0_gray, CV_BGR2GRAY);
			}

			//box.x = 400 - 1; box.y = 48 - 1; box.width = 87; box.height = 319;
			//Initialize LS tracker
			//box.x =182; box.y = 154; box.width = 100; box.height = 78;
			// car 219 246 288 175
			//box.x = 218; box.y = 245; box.width = 288; box.height = 175;
			eco_tracker.init(frame, box); // police 192 154 100 78
		}
		else
		{
			if (img_name == "0013.jpg")
				std::cout << std::endl;

			//cv::resize(frame, frame, cv::Size(frame.cols /2, frame.rows / 2));
			if (frame.empty())
				return -1;
			cv::Mat im_gray;
			//cvtColor(frame, im_gray, CV_BGR2GRAY);
			eco_tracker.process_frame(frame);

			//cv::rectangle(frame, showRect, cv::Scalar(0, 255, 0));
			//cv::imshow(WIN_NAME, frame);
			//cv::waitKey(1);
		}
	}
}

closedir(dir);
*/