#include "training.hpp"

namespace eco
{
EcoTrain::EcoTrain() {}
EcoTrain::~EcoTrain() {}
void EcoTrain::train_init(ECO_FEATS hf,
						  ECO_FEATS hf_inc,
						  vector<cv::Mat> proj_matrix,
						  ECO_FEATS xlf,
						  vector<cv::Mat> yf,
						  vector<cv::Mat> reg_filter,
						  ECO_FEATS sample_energy,
						  vector<float> reg_energy,
						  vector<cv::Mat> proj_energy,
						  EcoParameters &params)
{
	hf_ = hf;
	hf_inc_ = hf_inc;
	projection_matrix_ = proj_matrix;
	xlf_ = xlf;
	yf_ = yf;
	reg_filter_ = reg_filter;
	sample_energy_ = sample_energy;
	reg_energy_ = reg_energy;
	proj_energy_ = proj_energy;
	params_ = params;
}

void EcoTrain::train_joint()
{
	// Initial Gauss - Newton optimization of the filter and
	// projection matrix.

	// Index for the start of the last column of frequencies
	std::vector<int> lf_ind;
	for (size_t i = 0; i < hf_.size(); i++)
	{
		lf_ind.push_back(hf_[i][0].rows * (hf_[i][0].cols - 1));
	}

	// Construct stuff for the proj matrix part
	ECO_FEATS init_samplesf = xlf_;
	vector<cv::Mat> init_samplesf_H;

	for (size_t i = 0; i < xlf_.size(); i++)
	{
		cv::Mat temp;
		for (size_t j = 0; j < xlf_[i].size(); j++)
		{
			cv::Mat temp2 = xlf_[i][j].t();
			temp.push_back(cv::Mat(1,
								   xlf_[i][j].size().area(),
								   CV_32FC2,
								   temp2.data));
		}
		init_samplesf_H.push_back(mat_conj(temp));
	}

	// Construct preconditioner
	float precond_reg_param = params_.precond_reg_param,
		  precond_data_param = params_.precond_data_param;

	ECO_FEATS diag_M1;
	for (size_t i = 0; i < sample_energy_.size(); i++)
	{
		cv::Mat mean(cv::Mat::zeros(sample_energy_[i][0].size(), CV_32FC2));
		for (size_t j = 0; j < sample_energy_[i].size(); j++)
			mean += sample_energy_[i][j];
		mean = mean / sample_energy_[i].size(); // mean equal to matlab

		vector<cv::Mat> temp_vec;
		for (size_t j = 0; j < sample_energy_[i].size(); j++)
		{
			cv::Mat m;
			m = (1 - precond_data_param) * mean +
				precond_data_param * sample_energy_[i][j]; // totally equal to matlab
			m = m * (1 - precond_reg_param) +
				precond_reg_param * reg_energy_[i] *
					cv::Mat::ones(sample_energy_[i][0].size(), CV_32FC2);
			temp_vec.push_back(m);
		}
		diag_M1.push_back(temp_vec);
	}

	vector<cv::Mat> diag_M2;
	for (size_t i = 0; i < proj_energy_.size(); i++)
	{
		diag_M2.push_back(real2complex(params_.precond_proj_param *
									   (proj_energy_[i] + params_.projection_reg)));
	}

	joint_out diag_M(diag_M1, diag_M2);

	// training
	for (size_t i = 0; i < (size_t)params_.init_GN_iter; i++)
	{
		//  Project sample with new matrix
		ECO_FEATS init_samplef_proj =
			FeatureProjection(init_samplesf, projection_matrix_);
		ECO_FEATS init_hf = hf_;

		// Construct the right hand side vector for the filter part
		// A^H_P x conj{\hat{y}}
		ECO_FEATS rhs_samplef1 =
			FeatureVectorMultiply(init_samplef_proj, yf_, 1);

		// Construct the right hand side vector for the projection matrix part
		ECO_FEATS fyf = FeatureVectorMultiply(hf_, yf_, 1);
		vector<cv::Mat> rhs_samplef2 =
			compute_rhs2(projection_matrix_, init_samplesf_H, fyf, lf_ind);

		joint_out rhs_samplef(rhs_samplef1, rhs_samplef2);
		// Initialize the projection matrix increment to zero
		vector<cv::Mat> deltaP;
		for (size_t i = 0; i < projection_matrix_.size(); i++)
		{
			deltaP.push_back(cv::Mat::zeros(projection_matrix_[i].size(),
											projection_matrix_[i].type()));
		}

		joint_fp jointFP(hf_, deltaP);
		// Do conjugate gradient
		joint_out outFP = pcg_eco(init_samplef_proj,
								  reg_filter_,
								  init_samplesf,
								  init_samplesf_H,
								  init_hf,
								  params_.projection_reg,
								  rhs_samplef,
								  diag_M,
								  jointFP);

		// Make the filter symmetric(avoid roundoff errors)
		FilterSymmetrize(outFP.up_part);

		hf_ = outFP.up_part; // update the filter f
		// Add to the projection matrix
		projection_matrix_ = projection_matrix_ + outFP.low_part; // update P
	}
}

vector<cv::Mat> EcoTrain::compute_rhs2(const vector<cv::Mat> &proj_mat,
									   const vector<cv::Mat> &X_H,
									   const ECO_FEATS &fyf,
									   const vector<int> &lf_ind)
{
	vector<cv::Mat> res;
	vector<cv::Mat> fyf_vec = FeatureVectorization(fyf);
	for (size_t i = 0; i < X_H.size(); i++)
	{
		cv::Mat fyf_vect = fyf_vec[i].t();
		cv::Mat l1 = cmat_multi(X_H[i], fyf_vect);
		cv::Mat l2 = cmat_multi(X_H[i].colRange(lf_ind[i], X_H[i].cols),
								fyf_vect.rowRange(lf_ind[i], fyf_vect.rows));
		cv::Mat temp;
		temp = real2complex(2 * real(l1 - l2)) +
			   params_.projection_reg * proj_mat[i];
		res.push_back(temp);
	}
	return res;
}

EcoTrain::joint_fp EcoTrain::pcg_eco(const ECO_FEATS &init_samplef_proj,
									 const vector<cv::Mat> &reg_filter,
									 const ECO_FEATS &init_samplef,
									 const vector<cv::Mat> &init_samplesf_H,
									 const ECO_FEATS &init_hf,
									 float proj_reg,
									 const joint_out &rhs_samplef,
									 const joint_out &diag_M,
									 joint_fp &hf)
{
	joint_fp fpOut;
	int maxit = 15;		 // max iteration of conjugate gradient
	bool existM1 = true; // exist preconditoner
	if (diag_M.low_part.empty())
		existM1 = false; // no preconditioner
	joint_fp x = hf;	 // initialization of CG

	// Load the CG state
	joint_out p, r_perv;
	float rho = 1, rho1, alpha, beta;

	// calculate A(x)
	joint_out Ax = lhs_operation_joint(x, init_samplef_proj, reg_filter,
									   init_samplef, init_samplesf_H,
									   init_hf, params_.projection_reg);
	joint_out r = joint_minus(rhs_samplef, Ax);

	for (size_t ii = 0; ii < (size_t)maxit; ii++)
	{
		joint_out y, z;
		if (existM1) // exist preconditioner
			y = diag_precond(r, diag_M);
		else
			y = r;

		if (1) // exist M2
			z = y;
		else
			z = y;

		rho1 = rho;
		rho = inner_product_joint(r, z);

		if (ii == 0 && p.low_part.empty())
			p = z;
		else
		{
			beta = rho / rho1;
			beta = cv::max(0.0f, beta);
			p = z + p * beta;
		}
		joint_out q = lhs_operation_joint(p, init_samplef_proj, reg_filter,
										  init_samplef, init_samplesf_H,
										  init_hf, params_.projection_reg);

		float pq = inner_product_joint(p, q);

		if (pq <= 0 || pq > INT_MAX)
		{
			assert("GC condition is not matched");
			break;
		}
		else
			alpha = rho / pq; // standard alpha

		if (alpha <= 0 || alpha > INT_MAX)
		{
			assert("GC condition alpha is not matched");
			break;
		}

		x = x + p * alpha;

		if (ii < (size_t)maxit)
			r = r - q * alpha;
	}

	return x;
}

EcoTrain::joint_out EcoTrain::lhs_operation_joint(joint_fp &hf,
												  const ECO_FEATS &samplesf,
												  const vector<cv::Mat> &reg_filter,
												  const ECO_FEATS &init_samplef,
												  vector<cv::Mat> XH,
												  const ECO_FEATS &init_hf,
												  float proj_reg)
{
	joint_out AX;

	// Extract projection matrix and filter separately
	ECO_FEATS fAndDel = hf.up_part;		  // f + delta(f)
	vector<cv::Mat> deltaP = hf.low_part; // delta(P)

	// Get sizes
	int num_features = fAndDel.size();
	vector<cv::Size> filter_sz;
	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		filter_sz.push_back(fAndDel[i][0].size());
	}

	// find the maximum of size and its index
	vector<cv::Size>::iterator pos =
		max_element(filter_sz.begin(), filter_sz.end(), SizeCompare);
	size_t k1 = pos - filter_sz.begin();
	cv::Size output_sz = cv::Size(2 * pos->width - 1, pos->height);

	// Compute the operation corresponding to the data term in the optimization
	// (blockwise matrix multiplications)

	// 1 :sum over all features and feature blocks: A * f
	vector<cv::Mat> scores = FeatureComputeScores(samplesf, fAndDel);
	cv::Mat sh(cv::Mat::zeros(scores[k1].size(), scores[k1].type()));
	for (size_t i = 0; i < scores.size(); i++)
	{
		int pad = (output_sz.height - scores[i].rows) / 2;
		cv::Rect roi = cv::Rect(pad, pad, scores[i].cols, scores[i].rows);
		cv::Mat temp = scores[i] + sh(roi);
		temp.copyTo(sh(roi));
	}

	// 2: multiply with the transpose : A^H * A * f
	ECO_FEATS hf_out1;
	for (size_t i = 0; i < (size_t)num_features; i++) // for each feature
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < samplesf[i].size(); j++) // for each dimension
		{
			int pad = (output_sz.height - scores[i].rows) / 2;
			cv::Mat roi =
				sh(cv::Rect(pad, pad, scores[i].cols, scores[i].rows));
			cv::Mat res = complexMultiplication(mat_conj(roi), samplesf[i][j]);
			tmp.push_back(mat_conj(res));
		}
		hf_out1.push_back(tmp);
	}

	// compute the operation corresponding to the regularization term(convolve
	// each feature dimension with the DFT of w, and the tramsposed operation)
	// add the regularization part hf_conv = cell(1, 1, num_features);

	for (size_t i = 0; i < (size_t)num_features; i++) // for each feature
	{
		int reg_pad = cv::min(reg_filter[i].cols - 1, fAndDel[i][0].cols - 1);
		//vector<cv::Mat> hf_conv;
		for (size_t j = 0; j < fAndDel[i].size(); j++) // for each dimension
		{
			int c = fAndDel[i][j].cols;
			cv::Mat temp =
				fAndDel[i][j].colRange(c - reg_pad - 1, c - 1).clone();
			rot90(temp, 3); // flip

			// add part needed for convolution
			cv::hconcat(fAndDel[i][j], mat_conj(temp), temp);

			// do first convolution: W * f
			cv::Mat res1 = conv_complex(temp, reg_filter[i]);
			temp = res1;

			// do final convolution and put together result
			temp = conv_complex(temp.colRange(0, temp.cols - reg_pad),
								reg_filter[i], 1);

			// A^H * A * f + W^H * W * f
			hf_out1[i][j] += temp; // -0.2779
		}
	}

	// Stuff related to the projection matrix

	// 3: B * deltaP = X * inti(f)(before GC . previous  NG) * delta(P)
	vector<cv::Mat> BP_cell =
		FeatureComputeScores(FeatureProjection(init_samplef, deltaP), init_hf);

	cv::Mat BP(cv::Mat::zeros(BP_cell[k1].size(), BP_cell[k1].type()));
	for (size_t i = 0; i < scores.size(); i++)
	{
		int pad = (output_sz.height - BP_cell[i].rows) / 2;
		cv::Rect roi = cv::Rect(pad, pad, BP_cell[i].cols, BP_cell[i].rows);
		cv::Mat temp = BP_cell[i] + BP(roi);
		temp.copyTo(BP(roi));
	}

	// 4: A^H * B * dP
	ECO_FEATS fBP, shBP;

	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		vector<cv::Mat> vfBP, vshBP;
		for (size_t j = 0; j < hf_out1[i].size(); j++)
		{
			int pad = (output_sz.height - hf_out1[i][0].rows) / 2;
			cv::Rect roi = cv::Rect(pad, pad,
									hf_out1[i][0].cols, hf_out1[i][0].rows);
			cv::Mat temp =
				complexMultiplication(BP(roi),
									  mat_conj(samplesf[i][j].clone()));
			// A^H * A * f + W^H * W * f + A^H * B * dP
			hf_out1[i][j] += temp;

			vfBP.push_back(complexMultiplication(mat_conj(init_hf[i][j].clone()), BP(roi)));  // B^H * BP
			vshBP.push_back(complexMultiplication(mat_conj(init_hf[i][j].clone()), sh(roi))); // B^H * BP
		}
		fBP.push_back(vfBP);
		shBP.push_back(vshBP);
	}

	std::vector<cv::Mat> hf_out2;
	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		// the index of last frequency colunm starts
		int fi = hf_out1[i][0].rows * (hf_out1[i][0].cols - 1) + 0;

		// B^H * BP
		int c_len = XH[i].cols;
		cv::Mat part1 = XH[i] * FeatureVectorization(fBP)[i].t() -
						XH[i].colRange(fi, c_len) * FeatureVectorization(fBP)[i].colRange(fi, c_len).t();
		part1 = 2 * real2complex(real(part1)) + proj_reg * deltaP[i];

		// Compute proj matrix part : B^H * A_m * f
		cv::Mat part2 = XH[i] * FeatureVectorization(shBP)[i].t() -
						XH[i].colRange(fi, c_len) *
							FeatureVectorization(shBP)[i].colRange(fi, c_len).t();
		part2 = 2 * real2complex(real(part2));

		hf_out2.push_back(part1 + part2); // B^H * A * f + B^H * B * dp
	}

	AX.up_part = hf_out1;
	AX.low_part = hf_out2;
	return AX;
}

//************************************************************************
//                     this part is for filter training
//************************************************************************

void EcoTrain::train_filter(const vector<ECO_FEATS> &samplesf,
							const vector<float> &sample_weights,
							const ECO_FEATS &sample_energy)
{
	//double timereco = (double)cv::getTickCount();
	//float fpseco = 0;

	//1:  Construct the right hand side vector
	ECO_FEATS rhs_samplef = FeatureScale(samplesf[0], sample_weights[0]);
	for (size_t i = 1; i < samplesf.size(); i++)
	{
		rhs_samplef = FeatureScale(samplesf[i], sample_weights[i]) +
					  rhs_samplef;
	}
	rhs_samplef = FeatureVectorMultiply(rhs_samplef, yf_, 1);

	//2: Construct preconditioner
	ECO_FEATS diag_M;
	float precond_reg_param = params_.precond_reg_param,
		  precond_data_param = params_.precond_data_param;
	for (size_t i = 0; i < sample_energy.size(); i++)
	{
		cv::Mat mean(cv::Mat::zeros(sample_energy[i][0].size(), CV_32FC2));
		for (size_t j = 0; j < sample_energy[i].size(); j++)
			mean += sample_energy[i][j];
		mean = mean / sample_energy[i].size();

		vector<cv::Mat> temp_vec;
		for (size_t j = 0; j < sample_energy[i].size(); j++)
		{
			cv::Mat m;
			m = (1 - precond_data_param) * mean +
				precond_data_param * sample_energy[i][j];
			m = m * (1 - precond_reg_param) +
				precond_reg_param * reg_energy_[i] *
					cv::Mat::ones(sample_energy[i][0].size(), CV_32FC2);
			temp_vec.push_back(m);
		}
		diag_M.push_back(temp_vec);
	}

	//3: do conjugate gradient, get the filter updated
	hf_ = pcg_eco_filter(samplesf,
						 reg_filter_,
						 sample_weights,
						 rhs_samplef,
						 diag_M,
						 hf_);

	//fpseco = ((double)cv::getTickCount() - timereco) / 1000000;
	//debug("training time: %f", fpseco);
}

ECO_FEATS EcoTrain::pcg_eco_filter(const vector<ECO_FEATS> &samplesf,
								   const vector<cv::Mat> &reg_filter,
								   const vector<float> &sample_weights,
								   const ECO_FEATS &rhs_samplef,
								   const ECO_FEATS &diag_M,
								   ECO_FEATS &hf)
{
	ECO_FEATS res;

	int maxit = 5; // max iteration of conjugate gradient

	bool existM1 = true; // exist preconditoner
	if (diag_M.empty())
		existM1 = false; // no preconditioner

	ECO_FEATS x = hf; // initialization of CG

	// Load the CG state
	ECO_FEATS p, r_prev;
	float rho = 1, rho1, alpha, beta;
	for (size_t i = 0; i < hf.size(); ++i)
	{
		r_prev.push_back(vector<cv::Mat>(hf[i].size(),
										 cv::Mat::zeros(hf[i][0].size(), CV_32FC2)));
	}

	if (!state_.p.empty())
	{
		p = state_.p;
		rho = state_.rho / 0.5076;
		r_prev = state_.r_prev;
	}

	// calculate A(x)
	ECO_FEATS Ax = lhs_operation(x, samplesf, reg_filter, sample_weights);
	ECO_FEATS r = rhs_samplef - Ax;

	for (size_t ii = 0; ii < (size_t)maxit; ii++)
	{
		ECO_FEATS y, z;
		if (existM1) // exist preconditioner
			y = FeatureDotDivide(r, diag_M);
		else
			y = r;

		if (0) // exist M2
			z = y;
		else
			z = y;

		rho1 = rho;
		rho = inner_product(r, z);
		if ((rho == 0) || (std::abs(rho) >= INT_MAX) || (rho == NAN) || std::isnan(rho))
		{
			break;
		}
		if (ii == 0 && p.empty())
			p = z;
		else
		{
			float rho2 = inner_product(r_prev, z);
			beta = (rho - rho2) / rho1;

			if ((beta == 0) || (std::abs(beta) >= INT_MAX) || (beta == NAN) || std::isnan(beta))
				break;

			beta = cv::max(0.0f, beta);
			//p = FeatureAdd(z, FeatureScale(p, beta));
			p = z + FeatureScale(p, beta);
		}

		ECO_FEATS q = lhs_operation(p, samplesf, reg_filter, sample_weights);
		float pq = inner_product(p, q);

		if (pq <= 0 || (std::abs(pq) > INT_MAX) || (pq == NAN) || std::isnan(pq))
		{
			assert("GC condition is not matched");
			break;
		}
		else
			alpha = rho / pq; // standard alpha

		if ((std::abs(alpha) > INT_MAX) || (alpha == NAN) || std::isnan(alpha))
		{
			assert("GC condition alpha is not matched");
			break;
		}
		r_prev = r;

		// form new iterate
		//x = FeatureAdd(x, FeatureScale(p, alpha));
		x = x + FeatureScale(p, alpha);

		if (ii < (size_t)maxit - 1)
			//r = FeautreMinus(r, FeatureScale(q, alpha));
			r = r - FeatureScale(q, alpha);
	}

	state_.p = p;
	state_.rho = rho;
	state_.r_prev = r_prev;

	return x;
}

ECO_FEATS EcoTrain::lhs_operation(ECO_FEATS &hf,
								  const vector<ECO_FEATS> &samplesf,
								  const vector<cv::Mat> &reg_filter,
								  const vector<float> &sample_weights)
{
	ECO_FEATS res;
	int num_features = hf.size();
	vector<cv::Size> filter_sz;
	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		filter_sz.push_back(hf[i][0].size());
	}

	//1: find the maximum of size and its index
	vector<cv::Size>::iterator pos = max_element(filter_sz.begin(), filter_sz.end(), SizeCompare);
	size_t k1 = pos - filter_sz.begin();
	cv::Size output_sz = cv::Size(2 * pos->width - 1, pos->height);

	//2: sum over all features and feature blocks
	vector<cv::Mat> sh;
	for (size_t s = 0; s < samplesf.size(); s++)
	{
		vector<cv::Mat> scores = FeatureComputeScores(samplesf[s], hf);
		cv::Mat sh_tmp(cv::Mat::zeros(scores[k1].size(), scores[k1].type()));
		for (size_t i = 0; i < scores.size(); i++)
		{
			int pad = (output_sz.height - scores[i].rows) / 2;
			cv::Rect roi = cv::Rect(pad, pad, scores[i].cols, scores[i].rows);
			cv::Mat temp = scores[i] + sh_tmp(roi);
			temp.copyTo(sh_tmp(roi));
		}
		sh_tmp = sh_tmp * sample_weights[s];
		sh.push_back(sh_tmp);
	}

	//3: multiply with the transpose : A^H * A * f
	ECO_FEATS hf_out;
	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < hf[i].size(); j++)
		{
			int pad = (output_sz.height - hf[i][j].rows) / 2;
			cv::Mat res(cv::Mat::zeros(hf[i][j].size(), hf[i][j].type()));
			for (size_t s = 0; s < sh.size(); s++)
			{
				cv::Mat roi =
					sh[s](cv::Rect(pad, pad, hf[i][j].cols, hf[i][j].rows));
				res += complexMultiplication(mat_conj(roi), samplesf[s][i][j]);
			}
			tmp.push_back(mat_conj(res));
		}
		hf_out.push_back(tmp);
	}
	//4: compute the operation corresponding to the regularization term
	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		int reg_pad = cv::min(reg_filter[i].cols - 1, hf[i][0].cols - 1);
		vector<cv::Mat> hf_conv;
		for (size_t j = 0; j < hf[i].size(); j++)
		{
			int c = hf[i][j].cols;
			cv::Mat temp = hf[i][j].colRange(c - reg_pad - 1, c - 1).clone();
			rot90(temp, 3);

			cv::hconcat(hf[i][j], mat_conj(temp), temp);

			cv::Mat res1 = conv_complex(temp, reg_filter[i]);
			temp = res1;

			temp = conv_complex(temp.colRange(0, temp.cols - reg_pad), reg_filter[i], 1);

			hf_out[i][j] += temp;
		}
	}
	res = hf_out;
	return res;
}
//==============================================================================
EcoTrain::joint_out EcoTrain::joint_minus(const joint_out &a,
										  const joint_out &b)
{
	joint_out residual;
	ECO_FEATS up_rs;
	std::vector<cv::Mat> low_rs;
	for (size_t i = 0; i < a.up_part.size(); i++) // for each feature
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < a.up_part[i].size(); j++) // for each dimension
		{
			tmp.push_back(a.up_part[i][j] - b.up_part[i][j]);
		}
		up_rs.push_back(tmp);
		low_rs.push_back(a.low_part[i] - b.low_part[i]);
	}
	residual.up_part = up_rs;
	residual.low_part = low_rs;
	return residual;
}

EcoTrain::joint_out EcoTrain::diag_precond(const joint_out &a,
										   const joint_out &b)
{
	joint_out res;
	ECO_FEATS up_rs;
	std::vector<cv::Mat> low_rs;
	for (size_t i = 0; i < a.up_part.size(); i++) // for each feature
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < a.up_part[i].size(); j++) // for each dimension
		{
			tmp.push_back(complexDivision(a.up_part[i][j], b.up_part[i][j]));
		}
		up_rs.push_back(tmp);
		low_rs.push_back(complexDivision(a.low_part[i], b.low_part[i]));
	}
	res.up_part = up_rs;
	res.low_part = low_rs;
	return res;
}

float EcoTrain::inner_product_joint(const joint_out &a, const joint_out &b)
{
	float ip = 0;
	for (size_t i = 0; i < a.up_part.size(); i++) // for each feature
	{
		for (size_t j = 0; j < a.up_part[i].size(); j++) // for each dimension
		{
			int clen = a.up_part[i][j].cols;
			ip +=
				2 * mat_sum(real(complexMultiplication(
						mat_conj(a.up_part[i][j].clone()), b.up_part[i][j]))) -
				mat_sum(real(complexMultiplication(
					mat_conj(a.up_part[i][j].col(clen - 1).clone()),
					b.up_part[i][j].col(clen - 1))));
		}
		ip += mat_sum(real(complexMultiplication(
			mat_conj(a.low_part[i].clone()), b.low_part[i])));
	}
	return ip;
}

float EcoTrain::inner_product(const ECO_FEATS &a, const ECO_FEATS &b)
{
	float ip = 0;
	for (size_t i = 0; i < a.size(); i++) // for each feature
	{
		for (size_t j = 0; j < a[i].size(); j++) // for each dimension
		{
			int clen = a[i][j].cols;
			ip +=
				2 * mat_sum(real(complexMultiplication(
						mat_conj(a[i][j].clone()), b[i][j]))) -
				mat_sum(real(complexMultiplication(
					mat_conj(a[i][j].col(clen - 1).clone()),
					b[i][j].col(clen - 1))));
		}
	}
	return ip;
}
//==============================================================================
EcoTrain::rl_out EcoTrain::rl_out::operator+(rl_out data2)
{
	rl_out res;
	//res.up_part = FeatureAdd(up_part, data2.up_part);
	res.up_part = up_part + data2.up_part;
	res.low_part = low_part + data2.low_part;
	//res.low_part = ProjAdd(low_part, data2.low_part);
	return res;
}

EcoTrain::rl_out EcoTrain::rl_out::operator-(rl_out data)
{
	rl_out res;
	//res.up_part = FeautreMinus(up_part, data.up_part);
	res.up_part = up_part - data.up_part;
	res.low_part = low_part - data.low_part;
	//res.low_part = ProjMinus(low_part, data.low_part);
	return res;
}

EcoTrain::rl_out EcoTrain::rl_out::operator*(float scale)
{
	rl_out res;
	res.up_part = FeatureScale(up_part, scale);
	res.low_part = low_part * scale;
	//res.low_part = ProjScale(low_part, scale);
	return res;
}

} // namespace eco