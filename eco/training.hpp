#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <iostream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "ffttools.hpp"
#include "recttools.hpp"
#include "parameters.hpp"
#include "feature_operator.hpp"
#include "debug.hpp"

namespace eco
{
class EcoTrain
{
  public:
	EcoTrain();
	virtual ~EcoTrain();

	struct STATE
	{
		ECO_FEATS p, r_prev;
		float rho;
	};
	// the right and left side of the equation (18) of suppl. paper ECO
	struct ECO_EQ
	{
		ECO_EQ() {}
		ECO_EQ(ECO_FEATS up_part, std::vector<cv::Mat> low_part) : up_part_(up_part), low_part_(low_part) {}

		ECO_FEATS up_part_;			    // this is f + delta(f)
		std::vector<cv::Mat> low_part_; // this is delta(P)

		ECO_EQ operator+(const ECO_EQ data);
		ECO_EQ operator-(const ECO_EQ data);
		ECO_EQ operator*(const float scale);
	};

	void train_init(const ECO_FEATS &hf,
					const ECO_FEATS &hf_inc,
					const vector<cv::Mat> &proj_matrix,
					const ECO_FEATS &xlf,
					const vector<cv::Mat> &yf,
					const vector<cv::Mat> &reg_filter,
					const ECO_FEATS &sample_energy,
					const vector<float> &reg_energy,
					const vector<cv::Mat> &proj_energy,
					const EcoParameters &params);

	// Filter training and Projection updating(for the 1st Frame)==============
	void train_joint();

	ECO_EQ pcg_eco_joint(const ECO_FEATS &init_samplef_proj,
						   const vector<cv::Mat> &reg_filter,
						   const ECO_FEATS &init_samplef,
						   const vector<cv::Mat> &init_samplesf_H,
						   const ECO_FEATS &init_hf,
						   const ECO_EQ &rhs_samplef,
						   const ECO_EQ &diag_M, // preconditionor
						   const ECO_EQ &hf);

	ECO_EQ lhs_operation_joint(const ECO_EQ &hf,
								  const ECO_FEATS &samplesf,
								  const vector<cv::Mat> &reg_filter,
								  const ECO_FEATS &init_samplef,
								  const vector<cv::Mat> &XH,
								  const ECO_FEATS &init_hf);
	// Only filter training(for tracker update)===============================
	void train_filter(const vector<ECO_FEATS> &samplesf,
					  const vector<float> &sample_weights,
					  const ECO_FEATS &sample_energy);

	ECO_FEATS pcg_eco_filter(const vector<ECO_FEATS> &samplesf,
							 const vector<cv::Mat> &reg_filter,
							 const vector<float> &sample_weights,
							 const ECO_FEATS &rhs_samplef,
							 const ECO_FEATS &diag_M,
							 const ECO_FEATS &hf);

	ECO_FEATS lhs_operation_filter(const ECO_FEATS &hf,
								   const vector<ECO_FEATS> &samplesf,
								   const vector<cv::Mat> &reg_filter,
								   const vector<float> &sample_weights);
	// joint structure basic operation================================
	ECO_EQ jointDotDivision(const ECO_EQ &a, const ECO_EQ &b);
	float inner_product_joint(const ECO_EQ &a, const ECO_EQ &b);
	float inner_product_filter(const ECO_FEATS &a, const ECO_FEATS &b);
	vector<cv::Mat> get_proj() const { return projection_matrix_; }
	ECO_FEATS get_hf() const { return hf_; }

  private:
	ECO_FEATS hf_, hf_inc_; // filter parameters and its increament

	ECO_FEATS xlf_, sample_energy_;

	vector<cv::Mat> yf_; // the label of sample

	vector<cv::Mat> reg_filter_;
	vector<float> reg_energy_;

	vector<cv::Mat> projection_matrix_, proj_energy_;

	EcoParameters params_;
	STATE state_;
}; // end of class
} // namespace eco
#endif