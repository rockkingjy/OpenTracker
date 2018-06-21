#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#define INF 0x7f800000 //0x7fffffff 

typedef   std::vector<std::vector<cv::Mat> > ECO_FEATS;// ECO feature[Num_features][Dimension_of_the_feature];
typedef   cv::Vec<float, 2>                  COMPLEX;  // represent a complex number;
