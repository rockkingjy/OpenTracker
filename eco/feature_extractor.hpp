#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>

#ifdef USE_CAFFE
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/caffe.hpp>
#endif

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "parameters.hpp"
#include "fftTool.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "debug.hpp"

namespace eco
{
class FeatureExtractor
{
  public:
	FeatureExtractor() {}

	virtual ~FeatureExtractor(){};

	ECO_FEATS extractor(cv::Mat image,
						cv::Point2f pos,
						vector<float> scales,
						const EcoParameters &params);

	cv::Mat sample_patch(const cv::Mat &im,
						 const cv::Point2f &pos,
						 cv::Size2f sample_sz,
						 cv::Size2f input_sz,
						 const EcoParameters &gparams);
#ifdef USE_CAFFE
	ECO_FEATS get_cnn_layers(vector<cv::Mat> im, const cv::Mat &deep_mean_mat);
	void cnn_feature_normalization(ECO_FEATS &feature);
	inline ECO_FEATS get_cnn_feats() const { return cnn_feat_maps_; }
#endif

	vector<cv::Mat> get_hog_features(vector<cv::Mat> im);
	vector<cv::Mat> hog_feature_normalization(vector<cv::Mat> &feature);
	inline vector<cv::Mat> get_hog_feats() const { return hog_feat_maps_; }

	cv::Mat sample_pool(const cv::Mat &im, int smaple_factor, int stride);

  private:
#ifdef USE_CAFFE
	boost::shared_ptr<caffe::Net<float>> net_;
	CnnFeatures cnn_features_;
	int cnn_feat_ind_ = -1;
	ECO_FEATS cnn_feat_maps_;
#endif

	HogFeatures hog_features_;
	int hog_feat_ind_ = -1;
	vector<cv::Mat> hog_feat_maps_;

};
} // namespace eco
#endif
