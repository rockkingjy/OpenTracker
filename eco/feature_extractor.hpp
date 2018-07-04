#ifndef FEATURE_EXTRACTOR
#define FEATURE_EXTRACTOR

#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/caffe.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <numeric>

#include "parameters.hpp"
#include "fftTool.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "debug.hpp"

using namespace caffe;

namespace eco
{

class FeatureExtractor
{
  public:
	FeatureExtractor() {}

	virtual ~FeatureExtractor(){};

	ECO_FEATS extractor(cv::Mat 			image, 
						cv::Point2f 		pos,
						vector<float> 		scales,
						const EcoParameters 	&params,
						const cv::Mat 		&deep_mean_mat,
						const boost::shared_ptr<Net<float>> &net = boost::shared_ptr<Net<float>>());

	cv::Mat sample_patch(const cv::Mat &im, 
						 const cv::Point2f &pos, 
						 cv::Size2f sample_sz,
						 cv::Size2f input_sz, 
						 const EcoParameters &gparams);

	vector<cv::Mat> get_hog(vector<cv::Mat> im);

	vector<cv::Mat> hog_feature_normalization(vector<cv::Mat> &feature);

	ECO_FEATS get_cnn_layers(vector<cv::Mat> im, const cv::Mat &deep_mean_mat);

	void cnn_feature_normalization(ECO_FEATS &feature);

	cv::Mat sample_pool(const cv::Mat &im, int smaple_factor, int stride);

	inline ECO_FEATS get_cnn_feats() const { return cnn_feat_maps; }

	inline vector<cv::Mat> get_hog_feats() const { return hog_feat_maps; }

  private:
	CnnFeatures cnn_features;
	HogFeatures hog_features;

	int cnn_feat_ind = -1;
	int hog_feat_ind = -1;
	int cn_feat_ind  = -1;

	ECO_FEATS cnn_feat_maps;
	vector<cv::Mat> hog_feat_maps;

	boost::shared_ptr<Net<float>> net;

};
}
#endif
