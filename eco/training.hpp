#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <iostream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>

#include "fftTool.hpp"
#include "recttools.hpp"
#include "parameters.hpp"
#include "feature_operator.hpp"

namespace eco{
class EcoTrain
{
public:
	EcoTrain();
	virtual ~EcoTrain();

	struct STATE
	{
		ECO_FEATS p, r_prev;
		float  rho;
	};

	typedef struct rl_out    // the right and left side of the equation 
	{
		rl_out(){}
		rl_out(ECO_FEATS pup_part, std::vector<cv::Mat> plow_part):
			   up_part(pup_part), low_part(plow_part){}

		ECO_FEATS			       up_part;   // this is f + delta(f)
		std::vector<cv::Mat>       low_part;  // this is delta(P)
		
		rl_out operator+(rl_out data2);
		rl_out operator-(rl_out data2); 
		rl_out operator*(float  scale);    

	} joint_out, joint_fp;

	void               train_init(ECO_FEATS phf, 
								  ECO_FEATS phf_inc, 
								  vector<cv::Mat> pproj_matrix, 
								  ECO_FEATS pxlf, 
								  vector<cv::Mat> pyf,
								  vector<cv::Mat> preg_filter, 
								  ECO_FEATS psample_energy, 
								  vector<float> preg_energy, 
								  vector<cv::Mat> pproj_energy,
								  EcoParameters& params);

	void                train_joint();

	ECO_FEATS           mtimesx(ECO_FEATS& x, vector<cv::Mat> y, bool _conj = 0); // feature * yf 

	vector<cv::Mat>     compute_rhs2(const vector<cv::Mat>& proj_mat, const vector<cv::Mat>& init_samplef_H, 
									 const ECO_FEATS& fyf, const vector<int>& lf_ind); 

	vector<cv::Mat>     feat_vec(const ECO_FEATS& x); // conver feature into a  vector-matrix

	joint_out           lhs_operation_joint(joint_fp& hf, 
											const ECO_FEATS& samplesf, 
											const vector<cv::Mat>& reg_filter, 
											const ECO_FEATS& init_samplef, vector<cv::Mat>XH,
		                					const ECO_FEATS&  init_hf, 
											float proj_reg);	// the left side of equation in paper  A(x) to compute residual 

	joint_fp            pcg_eco(const ECO_FEATS& init_samplef_proj, 
								const vector<cv::Mat>& reg_filter, 
								const ECO_FEATS& init_samplef,
								const vector<cv::Mat>& init_samplesf_H, 
								const ECO_FEATS& init_hf, 
								float proj_reg, 				// right side of equation A(x)
								const joint_out& rhs_samplef,  	// the left side of the equation
								const joint_out& diag_M,       	// preconditionor 
								joint_fp& hf);                  // the union of filter [f+delta(f) delta(p)]
								
	//**** access to private membership 
	ECO_FEATS          	get_hf()  const{ return hf; }
	vector<cv::Mat>    	get_proj()const{ return projection_matrix; }

	//**** joint structure basic operation 
	joint_out          	joint_minus(const joint_out&a, const joint_out& b);    	// minus
	joint_out          	diag_precond(const joint_out&a, const joint_out& b);    // devide
	float              	inner_product_joint(const joint_out&a, const joint_out& b);
	float              	inner_product(const ECO_FEATS& a, const ECO_FEATS& b);

	//****  this part is for filter training ***
	void	           	train_filter(const vector<ECO_FEATS>& samplesf, 
									 const vector<float>& sample_weights, 
									 const ECO_FEATS& sample_energy);

	ECO_FEATS          	pcg_eco_filter( const vector<ECO_FEATS>& samplesf, 
										const vector<cv::Mat>& reg_filter, 
										const vector<float> &sample_weights,  	// right side of equation A(x)
										const ECO_FEATS& rhs_samplef,  			// the left side of the equation
										const ECO_FEATS& diag_M,       			// preconditionor 
										ECO_FEATS& hf);                			// the union of filter [f+delta(f) delta(p)]
						
	ECO_FEATS          	lhs_operation(ECO_FEATS& hf, 
									  const vector<ECO_FEATS>& samplesf, 
									  const vector<cv::Mat>& reg_filter, 
									  const vector<float> &sample_weights);

	ECO_FEATS          	conv2std(const vector<ECO_FEATS>& samplesf) const;

private:

	ECO_FEATS			hf, hf_inc;      	// filter parameters and its increament 

	ECO_FEATS           xlf, sample_energy; // the features fronier transform and its energy 
	
	vector<cv::Mat>     yf;              	// the label of sample 
	
	vector<cv::Mat>     reg_filter;      
	vector<float>       reg_energy;     

	vector<cv::Mat>     projection_matrix, proj_energy; // projection matrix and its energy 
	
	EcoParameters       params;

	float               resvec, resle;   	// Prellocate vector for norm of residuals  norm(b - A(x))s
	STATE               state;
}; // end of class
} // end of namespace
#endif