#ifndef KALMANTRACKER_HPP
#define KALMANTRACKER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "debug.hpp"

using namespace std;

namespace sort
{
class KalmanTracker
{
  public:
    static int count_;
    void init(cv::Rect2f bbox);
    void update(cv::Rect2f bbox);
    void predict();
    cv::Rect get_state();

  private:
    unsigned int type_ = CV_32F;
    int stateSize_ = 7;
    int measSize_ = 4;
    int contrSize_ = 0;
    cv::KalmanFilter kf_;
    cv::Mat state_;
    cv::Mat measure_;

    int id_ = 0;
    int time_since_update_ = 0;
    int hits_ = 0;
    int hit_streak_ = 0;
    int age_ = 0;
};


} // namespace sort

#endif