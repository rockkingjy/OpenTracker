#ifndef TRACKER_H
#define TRACKER_H

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videostab/inpainting.hpp>

#include "../helper/helper.h"
#include "../helper/bounding_box.h"
#include "../helper/high_res_timer.h"
#include "../helper/image_proc.h"
#include "../network/regressor.h"

namespace goturn{


class Tracker
{
public:
  Tracker(const bool show_tracking);

  // Estimate the location of the target object in the current image.
  virtual void Track(const cv::Mat& image_curr, RegressorBase* regressor,
             BoundingBox* bbox_estimate_uncentered);



  // Initialize the tracker with the ground-truth bounding box of the first frame.
  void Init(const cv::Mat& image_curr, const BoundingBox& bbox_gt,
            RegressorBase* regressor);

  // Initialize the tracker with the ground-truth bounding box of the first frame.
  // VOTRegion is an object for initializing the tracker when using the VOT Tracking dataset.


private:
  // Show the tracking output, for debugging.
  void ShowTracking(const cv::Mat& target_pad, const cv::Mat& curr_search_region, const BoundingBox& bbox_estimate) const;

  // Predicted prior location of the target object in the current image.
  // This should be a tight (high-confidence) prior prediction area.  We will
  // add padding to this region.
  BoundingBox bbox_curr_prior_tight_;

  // Estimated previous location of the target object.
  BoundingBox bbox_prev_tight_;

  // Full previous image.
  cv::Mat image_prev_;

  // Whether to visualize the tracking results
  bool show_tracking_;
};

}
#endif // TRACKER_H
