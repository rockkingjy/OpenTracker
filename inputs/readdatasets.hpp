#ifndef READDATASETS_HPP
#define READDATASETS_HPP

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

class ReadDatasets
{
  public:
    ReadDatasets();
    virtual ~ReadDatasets();
    void IniRead(cv::Rect2f &bboxGroundtruth, cv::Mat &frame);
    void ReadNextFrame(cv::Rect2f &bboxGroundtruth, cv::Mat &frame);
    void DrawGroundTruth(cv::Rect2f &bboxGroundtruth, cv::Mat &frameDraw);

  private:
    string databaseTypes[5] = {"Demo", "VOT-2017", "TB-2015", "TLP", "UAV123"};
    string databaseType = databaseTypes[0];
    int f;      // file index
    int isLost; // lost flag
    float x, y, w, h;
    float x1, y1, x2, y2, x3, y3, x4, y4; 
    std::string s;
    std::string path;
    ifstream *groundtruth;
    ostringstream osfile;
};
#endif