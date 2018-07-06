#ifndef OPENPOSE_HPP
#define OPENPOSE_HPP

#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <thread> // std::this_thread
#include <gflags/gflags.h>
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <iostream>

struct UserDatum : public op::Datum
{
    bool boolThatUserNeedsForSomeReason;

    UserDatum(const bool boolThatUserNeedsForSomeReason_ = false) : boolThatUserNeedsForSomeReason{boolThatUserNeedsForSomeReason_}
    {
    }
};

class WUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<UserDatum>>>
{
  public:
    WUserOutput(cv::Rect2f* bboxGroundtruth);
    void initializationOnThread() {}
    void workConsumer(const std::shared_ptr<std::vector<UserDatum>> &datumsPtr);

  private:
    int xmin = 0;
    int ymin = 0;
    int xmax = 0;
    int ymax = 0;
    cv::Rect2f* gt;
    cv::Mat f;
};

int openPoseDemo(cv::Rect2f* bboxGroundtruth);

class OpenPose
{
  public: 
    OpenPose();
    virtual ~OpenPose();
    void IniRead(cv::Rect2f &bboxGroundtruth);
};
#endif