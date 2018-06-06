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

#include "params.h"
#include "feature_type.h"
#include "fftTool.h"
#include "recttools.hpp"
#include "FHOG.hpp"
#include "fhog_f.hpp"

using namespace FFTTools;
//using namespace std;
using namespace caffe;

class feature_extractor
{
  public:
	feature_extractor() {}

	virtual ~feature_extractor(){};

	ECO_FEATS extractor(cv::Mat 			image, 
						cv::Point2f 		pos,
						vector<float> 		scales,
						const eco_params 	&params,
						const cv::Mat 		&yml_mean,
						const boost::shared_ptr<Net<float>> &net = boost::shared_ptr<Net<float>>());

	cv::Mat sample_patch(const cv::Mat &im, 
						 const cv::Point2f &pos, 
						 cv::Size2f sample_sz,
						 cv::Size2f output_sz, 
						 const eco_params &gparams);

	vector<cv::Mat> get_hog(vector<cv::Mat> im);

	vector<cv::Mat> hog_feature_normalization(vector<cv::Mat> &feature);

	ECO_FEATS get_cnn_layers(vector<cv::Mat> im, const cv::Mat &yml_mean);

	void cnn_feature_normalization(ECO_FEATS &feature);

	void WrapInputLayer(std::vector<cv::Mat> *input_channels);

	cv::Mat sample_pool(const cv::Mat &im, int smaple_factor, int stride);

	inline ECO_FEATS get_cnn_feats() const { return cnn_feat_maps; }

	inline vector<cv::Mat> get_hog_feats() const { return hog_feat_maps; }

  private:
	cnn_feature cnn_features;
	hog_feature hog_features;

	ECO_FEATS cnn_feat_maps;
	vector<cv::Mat> hog_feat_maps;

	boost::shared_ptr<Net<float>> net;
};

#endif
