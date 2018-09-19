#ifndef ECO_HPP
#define ECO_HPP

#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

#ifdef USE_CAFFE
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#endif
/*
#ifdef USE_CUDA
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#endif 
*/
#ifdef USE_MULTI_THREAD
#include <pthread.h>
#include <unistd.h>
#endif

#include "parameters.hpp"
#include "interpolator.hpp"
#include "regularization_filter.hpp"
#include "feature_extractor.hpp"
#include "feature_operator.hpp"
#include "sample_update.hpp"
#include "optimize_scores.hpp"
#include "training.hpp"
#include "ffttools.hpp"
#include "scale_filter.hpp"
#include "debug.hpp"

namespace eco
{
class ECO
{
  public:
	ECO() {};
	virtual ~ECO() {}

	void init(cv::Mat &im, const cv::Rect2f &rect, const eco::EcoParameters &paramters); 

	bool update(const cv::Mat &frame, cv::Rect2f &roi);
	
	void init_parameters(const eco::EcoParameters &parameters);

	void init_features(); 
#ifdef USE_CAFFE
	void read_deep_mean(const string &mean_file);
#endif
	void yf_gaussian(); // the desired outputs of features, real part of (9) in paper C-COT

	void cos_window(); 	// construct cosine window of features;

	ECO_FEATS interpolate_dft(const ECO_FEATS &xlf, 
							  vector<cv::Mat> &interp1_fs,
							  vector<cv::Mat> &interp2_fs);

	ECO_FEATS compact_fourier_coeff(const ECO_FEATS &xf);

	ECO_FEATS full_fourier_coeff(const ECO_FEATS &xf);

	vector<cv::Mat> project_mat_energy(vector<cv::Mat> proj, 
									   vector<cv::Mat> yf);
	
	ECO_FEATS shift_sample(ECO_FEATS &xf, 
						   cv::Point2f shift,
						   std::vector<cv::Mat> kx, 
						   std::vector<cv::Mat> ky);
#ifdef USE_MULTI_THREAD
	static void *thread_train(void *params);
#endif

  private:
	bool				is_color_image_;
	EcoParameters 		params_;
	cv::Point2f 		pos_; 							// final result
	size_t 				frames_since_last_train_; 	 	// used for update;

	// The max size of feature and its index, output_sz is T in (9) of C-COT paper
	size_t 				output_size_, output_index_; 	

	cv::Size2f 			base_target_size_; 	// target size without scale
	cv::Size2i			img_sample_size_;  	// base_target_sz * sarch_area_scale
	cv::Size2i			img_support_size_;	// the corresponding size in the image

	vector<cv::Size> 	feature_size_, filter_size_;
	vector<int> 		feature_dim_, compressed_dim_;

	ScaleFilter 		scale_filter_;
	int 				nScales_;				// number of scales;
	float 				scale_step_;
	vector<float>		scale_factors_;
	float 				currentScaleFactor_; 	// current img scale 

	// Compute the Fourier series indices 
	// kx_, ky_ is the k in (9) of C-COT paper, yf_ is the left part of (9);
	vector<cv::Mat> 	ky_, kx_, yf_; 
	vector<cv::Mat> 	interp1_fs_, interp2_fs_; 
	vector<cv::Mat> 	cos_window_;
	vector<cv::Mat> 	projection_matrix_;

	vector<cv::Mat> 	reg_filter_;
	vector<float> 		reg_energy_;

	FeatureExtractor 	feature_extractor_;

	SampleUpdate 		sample_update_;
	ECO_FEATS 			sample_energy_;

	EcoTrain 			eco_trainer_;

	ECO_FEATS 			hf_full_;

#ifdef USE_MULTI_THREAD
	bool 				thread_flag_train_;
  public:
	pthread_t			thread_train_;
#endif
	
};

} // namespace eco

#endif
