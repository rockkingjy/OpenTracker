#ifndef ECO_H
#define ECO_H

#include <iostream>
#include <string>
#include <math.h>

#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "params.h"
#include "feature_type.h"
#include "interpolator.h"
#include "reg_filter.h"
#include "feature_extractor.h"
#include "feature_operator.h"
#include "eco_sample_update.h"
#include "optimize_scores.h"
#include "training.h"

#endif

//using namespace std;
using namespace caffe;
//using namespace cv;
using namespace FFTTools;
using namespace eco_sample_update;

namespace eco
{

class ECO
{
  public:
	virtual ~ECO() {}

	ECO() {};

	void init(cv::Mat &im, const cv::Rect &rect); //****** tracker intialization****

	bool update(const cv::Mat &frame, cv::Rect2d& roi);

	void init_features(); // *** init the ECO features include deep feature or non-deep feature

	cv::Mat meanMatFromYML(string path);

	cv::Mat deep_mean(const string &mean_file);

	void yf_gaussian(); //***** the desired outputs of features, real part of (9) in paper C-COT

	void cos_wind(); 	//***** construct cosine window of features;

	ECO_FEATS do_windows_x(const ECO_FEATS &xl, vector<cv::Mat> &cos_win);

	ECO_FEATS interpolate_dft(const ECO_FEATS &xlf, vector<cv::Mat> &interp1_fs,
							  vector<cv::Mat> &interp2_fs);

	ECO_FEATS compact_fourier_coeff(const ECO_FEATS &xf);

	vector<cv::Mat> init_projection_matrix(const ECO_FEATS &init_sample,
										   const vector<int> &compressed_dim,
										   const vector<int> &feature_dim);

	vector<cv::Mat> project_mat_energy(vector<cv::Mat> proj, vector<cv::Mat> yf);

	ECO_FEATS full_fourier_coeff(const ECO_FEATS &xf);

	ECO_FEATS shift_sample(ECO_FEATS &xf, cv::Point2f shift,
						   std::vector<cv::Mat> kx, std::vector<cv::Mat> ky);

  private:
	eco_params 			params;
	boost::shared_ptr<Net<float>> 	net;
	cv::Mat 			deep_mean_mat, yml_mean;

	//*** the max size of feature and its index
	size_t 				output_sz, max_output_index; //output_sz is T in (9) of C-COT paper
	size_t 				frameID, frames_since_last_train;
	cv::Point2f 		pos; // final result.

	//***  current target size,  initial target size,
	cv::Size 			target_sz, img_sample_sz, img_support_sz;
	cv::Size2f 			base_target_sz; // *** target size without scale
	float 				currentScaleFactor; //*** current img scale 


	vector<cv::Size> 	feature_sz, filter_sz;
	vector<int> 		feature_dim, compressed_dim;

	// *** Compute the Fourier series indices and their transposes
	//kx, ky is the k in (9) of C-COT paper, yf is the left part of (9);
	vector<cv::Mat> 	ky, kx, yf, cos_window; 
	vector<cv::Mat> 	interp1_fs, interp2_fs; // *** interpl fourier series

	vector<cv::Mat> 	reg_filter, projection_matrix; //**** spatial filter *****
	vector<float> 		reg_energy, scaleFactors;

	feature_extractor 	feat_extrator;

	sample_update 		SampleUpdate;

	ECO_FEATS 			sample_energy;
	ECO_FEATS 			hf_full;
	eco_train 			eco_trainer;
};

} // namespace eco