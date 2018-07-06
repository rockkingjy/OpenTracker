#ifndef READVIDEO_HPP
#define READVIDEO_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

using namespace cv;
using namespace std;

class ReadVideo
{
public:
  ReadVideo();
  virtual ~ReadVideo();
  void IniRead(cv::Rect2f &bboxGroundtruth, cv::Mat &frame, std::string window_name);
  static void mouseHandler(int event, int x, int y, int flags, void *param)
  {
    switch (event)
    {
    case CV_EVENT_LBUTTONDOWN:
      drawing_now_flag_ = true;
      bbox_ = cv::Rect(x, y, 0, 0);
      break;
    case CV_EVENT_MOUSEMOVE:
      if (drawing_now_flag_)
      {
        bbox_.width = x - bbox_.x;
        bbox_.height = y - bbox_.y;
      }
      break;
    case CV_EVENT_LBUTTONUP:
      drawing_now_flag_ = false;
      if (bbox_.width < 0)
      {
        bbox_.x += bbox_.width;
        bbox_.width *= -1;
      }
      if (bbox_.height < 0)
      {
        bbox_.y += bbox_.height;
        bbox_.height *= -1;
      }
      bbox_get_flag_ = true;
      break;
    }
  }

private:
  static bool drawing_now_flag_;
  static bool bbox_get_flag_;
  static cv::Rect2f bbox_;
};

#endif