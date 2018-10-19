#ifndef SORT_HPP
#define SORT_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "kalmantracker.hpp"

using namespace std;

namespace sort
{
class Sort
{
  public:
    void init();
    void update(cv::Rect2f bbox);
    void associate_detections_to_trackers();

  private:
    int max_age_ = 1;
    int min_hits_ = 3;
    float iou_threhold = 0.3;
    //vector<sort::KalmanTracker> trackers;
};
} // namespace sort

#endif