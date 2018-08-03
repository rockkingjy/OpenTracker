#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include <opencv2/core/core.hpp>

#include "parameters.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "debug.hpp"

#ifdef USE_SIMD
#include <stdio.h>
#include <stdlib.h>
#include "hog/rgbConvertMex.hpp"
#include "hog/imPadMex.hpp"
#include "hog/convConst.hpp"
#include "hog/imResampleMex.hpp"
#include "hog/gradientMex.hpp"
#endif

#ifdef USE_CAFFE
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/caffe.hpp>
#endif

namespace eco
{
class FeatureExtractor
{
  public:
	FeatureExtractor() {}
	virtual ~FeatureExtractor(){};

	ECO_FEATS extractor(const cv::Mat image,
						const cv::Point2f pos,
						const vector<float> scales,
						const EcoParameters &params);

	cv::Mat sample_patch(const cv::Mat im,
						 const cv::Point2f pos,
						 cv::Size2f sample_sz,
						 cv::Size2f input_sz);

	vector<cv::Mat> get_hog_features(const vector<cv::Mat> ims);
	vector<cv::Mat> hog_feature_normalization(vector<cv::Mat> &feature);

#ifdef USE_SIMD
	vector<cv::Mat> get_hog_features_simd(const vector<cv::Mat> ims);
	vector<cv::Mat> hog_feature_normalization_simd(vector<cv::Mat> &feature);
#endif	

	inline vector<cv::Mat> get_hog_feats() const { return hog_feat_maps_; }

#ifdef USE_CAFFE
	ECO_FEATS get_cnn_layers(vector<cv::Mat> im, const cv::Mat &deep_mean_mat);
	cv::Mat sample_pool(const cv::Mat &im, int smaple_factor, int stride);
	void cnn_feature_normalization(ECO_FEATS &feature);
	inline ECO_FEATS get_cnn_feats() const { return cnn_feat_maps_; }
#endif

  private:
	HogFeatures hog_features_;
	int hog_feat_ind_ = -1;
	vector<cv::Mat> hog_feat_maps_;

#ifdef USE_CAFFE
	boost::shared_ptr<caffe::Net<float>> net_;
	CnnFeatures cnn_features_;
	int cnn_feat_ind_ = -1;
	ECO_FEATS cnn_feat_maps_;
#endif
};
} // namespace eco
#endif
