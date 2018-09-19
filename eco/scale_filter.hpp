#ifndef SCALE_FILTER_HPP
#define SCALE_FILTER_HPP

#include "parameters.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "feature_extractor.hpp"
#include "debug.hpp"
#include <opencv2/core.hpp>

namespace eco
{

class ScaleFilter
{
  public:
    ScaleFilter(){};
    virtual ~ScaleFilter(){};
    void init(int &nScales, float &scale_step, const EcoParameters &params);
    float scale_filter_track(const cv::Mat &im, const cv::Point2f &pos, const cv::Size2f &base_target_sz, const float &currentScaleFactor, const EcoParameters &params);
    cv::Mat extract_scale_sample(const cv::Mat &im, const cv::Point2f &posf, const cv::Size2f &base_target_sz, vector<float> &scaleFactors, const cv::Size &scale_model_sz);
    
  private:
    vector<float> scaleSizeFactors_;
    vector<float> interpScaleFactors_;
    cv::Mat yf_;
    vector<float> window_;
    bool max_scale_dim_;
};
} // namespace eco
#endif