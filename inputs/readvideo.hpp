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
    void IniRead(cv::Rect2f &bboxGroundtruth, cv::Mat &frame);
    void ReadNextFrame(cv::Rect2f &bboxGroundtruth, cv::Mat &frame);
    void DrawGroundTruth(cv::Rect2f &bboxGroundtruth, cv::Mat &frameDraw);

  private:
    float x, y, w, h;
    std::string path;
};

#endif