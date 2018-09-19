#include "training.hpp"

namespace eco
{
EcoTrain::EcoTrain() {}
EcoTrain::~EcoTrain() {}
void EcoTrain::train_init(const ECO_FEATS &hf,
						  const ECO_FEATS &hf_inc,
						  const vector<cv::Mat> &proj_matrix,
						  const ECO_FEATS &xlf,
						  const vector<cv::Mat> &yf,
						  const vector<cv::Mat> &reg_filter,
						  const ECO_FEATS &sample_energy,
						  const vector<float> &reg_energy,
						  const vector<cv::Mat> &proj_energy,
						  const EcoParameters &params)
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

//************************************************************************
//     	Filter training and Projection updating(for the 1st Frame)
//************************************************************************
void EcoTrain::train_joint()
{
	// 1. Initial Gauss-Newton optimization of the filter and projection matrix.

	// Index for the start of the last column of frequencies
	std::vector<int> lf_ind;
	for (size_t i = 0; i < hf_.size(); i++)
	{
		lf_ind.push_back(hf_[i][0].rows * (hf_[i][0].cols - 1));
	}
	// debug:
	/*for (size_t i = 0; i < lf_ind.size(); i++)
	{
		debug("lf_ind:%lu, %lu, %d", i, lf_ind.size(), lf_ind[i]);
	}
	*/
	// Construct stuff for the proj matrix part
	ECO_FEATS init_samplesf = xlf_;
	vector<cv::Mat> init_samplesf_H;
	for (size_t i = 0; i < xlf_.size(); i++) // for each feature
	{
		cv::Mat temp;
		for (size_t j = 0; j < xlf_[i].size(); j++) // for each dimension
		{
			cv::Mat temp_h = xlf_[i][j].t();
			temp.push_back(cv::Mat(1, xlf_[i][j].size().area(), CV_32FC2, temp_h.data));
		}
		init_samplesf_H.push_back(mat_conj(temp));
	}
	//debug: init_samplesf_H: 0, 31 x 325
	/*for (size_t i = 0; i < init_samplesf_H.size(); i++)
	{
		debug("init_samplesf_H %lu 's size: %d x %d", i, init_samplesf_H[i].rows, init_samplesf_H[i].cols);
	}
	*/
	// Construct preconditioner diag_M
	ECO_FEATS diag_M1;
	for (size_t i = 0; i < sample_energy_.size(); i++) // for features
	{
		cv::Mat mean(cv::Mat::zeros(sample_energy_[i][0].size(), CV_32FC2));
		for (size_t j = 0; j < sample_energy_[i].size(); j++) // for dimensions
		{
			mean += sample_energy_[i][j];
		}
		mean = mean / sample_energy_[i].size();

		vector<cv::Mat> temp_vec;
		for (size_t j = 0; j < sample_energy_[i].size(); j++) // for dimensions
		{
			cv::Mat temp = (1 - params_.precond_data_param) * mean + params_.precond_data_param * sample_energy_[i][j];
			temp = temp * (1 - params_.precond_reg_param) + params_.precond_reg_param * reg_energy_[i] * cv::Mat::ones(sample_energy_[i][0].size(), CV_32FC2);
			temp_vec.push_back(temp);
		}
		diag_M1.push_back(temp_vec);
	}
	vector<cv::Mat> diag_M2;
	for (size_t i = 0; i < proj_energy_.size(); i++)
	{
		diag_M2.push_back(real2complex(params_.precond_proj_param * (proj_energy_[i] + params_.projection_reg)));
	}
	ECO_EQ diag_M(diag_M1, diag_M2);
	// debug:
	// diag_M1: 0, 10, 25 x 13
	// diag_M2: 0,  1, 31 x 10
	/*debug("diag_M1:");
	printECO_FEATS(diag_M1);
	debug("diag_M2:");
	printVector_Mat(diag_M2);
	*/
	// 2. Training with Gauss-Newton optimization
	for (size_t i = 0; i < (size_t)params_.init_GN_iter; i++)
	{
		//  Project sample with new matrix
		ECO_FEATS init_samplef_proj = FeatureProjection(init_samplesf, projection_matrix_);
		ECO_FEATS init_hf = hf_;
		/*
		// debug: 0, 10, 25 x 13
		debug("init_samplef_proj:");
		printECO_FEATS(init_samplef_proj);
		debug("init_hf:");
		printECO_FEATS(init_hf);
*/
		// Construct the right hand side vector for the filter part
		// A^H * y
		ECO_FEATS rhs_samplef1 = FeatureVectorMultiply(init_samplef_proj, yf_, 1);
		// Construct the right hand side vector for the projection matrix part
		// B^H * y - lambda * P
		vector<cv::Mat> rhs_samplef2;
		ECO_FEATS fyf = FeatureVectorMultiply(hf_, yf_, 1);
		vector<cv::Mat> fyf_vec = FeatureVectorization(fyf);
		for (size_t i = 0; i < init_samplesf_H.size(); i++)
		{
			cv::Mat fyf_vec_T = fyf_vec[i].t();
			cv::Mat l1 = complexMatrixMultiplication(init_samplesf_H[i],
													 fyf_vec_T);
			cv::Mat col = init_samplesf_H[i].colRange(lf_ind[i],
													  init_samplesf_H[i].cols);
			cv::Mat row = fyf_vec_T.rowRange(lf_ind[i], fyf_vec_T.rows);
			//31 x 25, 25 x 10
			//debug("col, row:");printMat(col);printMat(row);
			cv::Mat l2 = complexMatrixMultiplication(col, row);
			cv::Mat temp = real2complex(2 * real(l1 - l2)) -
						   params_.projection_reg * projection_matrix_[i];
			rhs_samplef2.push_back(temp);
		}
		ECO_EQ rhs_samplef(rhs_samplef1, rhs_samplef2);
		/*
		// debug:
		// rhs_samplef1: 0, 10, 25 x 13
		// rhs_samplef2: 0,  1, 31 x 10
		debug("rhs_samplef1:");
		printECO_FEATS(rhs_samplef1);
		debug("rhs_samplef2:");
		printVector_Mat(rhs_samplef2);
*/
		// Initialize the projection matrix increment to zero
		vector<cv::Mat> deltaP;
		for (size_t i = 0; i < projection_matrix_.size(); i++)
		{
			deltaP.push_back(cv::Mat::zeros(projection_matrix_[i].size(),
											projection_matrix_[i].type()));
		}
		ECO_EQ jointFP(hf_, deltaP);

		// Do conjugate gradient
		ECO_EQ outFP = pcg_eco_joint(init_samplef_proj,
									 reg_filter_,
									 init_samplesf,
									 init_samplesf_H,
									 init_hf,
									 rhs_samplef,
									 diag_M,
									 jointFP);
		// Make the filter symmetric(avoid roundoff errors)
		FilterSymmetrize(outFP.up_part_);
		hf_ = outFP.up_part_; // update the filter f

		// Add to the projection matrix
		projection_matrix_ = projection_matrix_ + outFP.low_part_; // update P
	}
}

EcoTrain::ECO_EQ EcoTrain::pcg_eco_joint(const ECO_FEATS &init_samplef_proj,
										 const vector<cv::Mat> &reg_filter,
										 const ECO_FEATS &init_samplef,
										 const vector<cv::Mat> &init_samplesf_H,
										 const ECO_FEATS &init_hf,
										 const ECO_EQ &rhs_samplef,
										 const ECO_EQ &diag_M,
										 const ECO_EQ &hf)
{
	int maxit = params_.CG_opts.maxit;
	bool existM1 = true;
	if (diag_M.low_part_.empty())
	{
		existM1 = false;
	}
	ECO_EQ x = hf; // initialization of CG

	ECO_EQ p, r_prev;
	float rho = 1, rho1, alpha, beta;

	// calculate A(x)
	ECO_EQ Ax = lhs_operation_joint(x,
									init_samplef_proj,
									reg_filter,
									init_samplef,
									init_samplesf_H,
									init_hf);
	ECO_EQ r = rhs_samplef; // rhs_samplef is const, needs to seperate to 2
	r = r - Ax;

	for (size_t ii = 0; ii < (size_t)maxit; ii++)
	{
		ECO_EQ y, z;
		if (existM1) // exist preconditioner
		{
			y = jointDotDivision(r, diag_M);
		}
		else
		{
			y = r;
		}
		z = y;

		rho1 = rho;
		rho = inner_product_joint(r, z);
		if ((rho == 0) || (std::abs(rho) >= INT_MAX) || std::isnan(rho))
		{
			break;
		}

		if (ii == 0 && p.low_part_.empty())
		{
			p = z;
		}
		else
		{
			if (params_.CG_opts.CG_use_FR) // Use Fletcher-Reeves
			{
				beta = rho / rho1;
			}
			else // Use Polak-Ribiere
			{
				float rho2 = inner_product_joint(r_prev, z);
				beta = (rho - rho2) / rho1;
			}
			if ((beta == 0) || (std::abs(beta) >= INT_MAX) || std::isnan(beta))
			{
				break;
			}
			beta = cv::max(0.0f, beta);
			p = z + p * beta;
		}
		ECO_EQ q = lhs_operation_joint(p,
									   init_samplef_proj,
									   reg_filter,
									   init_samplef,
									   init_samplesf_H,
									   init_hf);

		float pq = inner_product_joint(p, q);

		if (pq <= 0 || (std::abs(pq) > INT_MAX) || std::isnan(pq))
		{
			assert(0 && "error: GC condition is not matched");
		}
		else
		{
			if (params_.CG_opts.CG_standard_alpha)
			{
				alpha = rho / pq; // standard alpha
			}
			else
			{
				alpha = inner_product_joint(p, r) / pq;
			}
		}
		if ((std::abs(alpha) > INT_MAX) || std::isnan(alpha))
		{
			assert(0 && "GC condition alpha is not matched");
		}

		// Save old r if not using FR formula for beta
		if (!params_.CG_opts.CG_use_FR) // Use Polak-Ribiere
		{
			r_prev = r;
		}

		// form new iterate
		x = x + p * alpha;

		if (ii < (size_t)maxit)
		{
			r = r - q * alpha;
		}
	}
	return x;
}

// This is the left-hand-side operation in Conjugate Gradient
EcoTrain::ECO_EQ EcoTrain::lhs_operation_joint(const ECO_EQ &hf,
											   const ECO_FEATS &samplesf,
											   const vector<cv::Mat> &reg_filter,
											   const ECO_FEATS &init_samplef,
											   const vector<cv::Mat> &XH,
											   const ECO_FEATS &init_hf)
{
	// Extract projection matrix and filter separately
	ECO_FEATS fAndDel = hf.up_part_;	   // f + delta(f)
	vector<cv::Mat> deltaP = hf.low_part_; // delta(P)
	// fAndDel: 0, 10, 25 x 13
	// deltaP: 0, 1, 31 x 10
	/*
	debug("fAndDel:");
	printECO_FEATS(fAndDel);
	debug("deltaP:");
	printVector_Mat(deltaP);*/

	// 1: Get sizes of each feature----------------------------------------
	int num_features = fAndDel.size();
	vector<cv::Size> filter_sz;
	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		filter_sz.push_back(fAndDel[i][0].size());
	}
	// find the maximum of size and its index
	vector<cv::Size>::iterator pos = max_element(filter_sz.begin(), filter_sz.end(), SizeCompare);
	size_t k1 = pos - filter_sz.begin(); // index
	cv::Size output_sz = cv::Size(2 * pos->width - 1, pos->height);

	// Compute the operation corresponding to the data term in the optimization
	// (blockwise matrix multiplications)

	// 2 :sum over all features and feature blocks: A * f---------------------
	vector<cv::Mat> scores = FeatureComputeScores(samplesf, fAndDel);

	cv::Mat sh(cv::Mat::zeros(scores[k1].size(), scores[k1].type()));
	for (size_t i = 0; i < scores.size(); i++)
	{
		int pad = (output_sz.height - scores[i].rows) / 2;
		cv::Rect roi = cv::Rect(pad, pad, scores[i].cols, scores[i].rows);
		sh(roi) = scores[i] + sh(roi);
	}
	// sh: 25 x 13
	//debug("sh:");
	//printMat(sh);

	// 3: multiply with the transpose : A^H * A * f----------------------------
	ECO_FEATS hf_out1;
	for (size_t i = 0; i < (size_t)num_features; i++) // for each feature
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < fAndDel[i].size(); j++) // for each dimension
		{
			int pad = (output_sz.height - scores[i].rows) / 2;
			cv::Mat roi = sh(cv::Rect(pad, pad, scores[i].cols, scores[i].rows));
			cv::Mat res = complexDotMultiplication(mat_conj(roi), samplesf[i][j]);
			tmp.push_back(mat_conj(res));
		}
		hf_out1.push_back(tmp);
	}
	// hf_out1: 0, 10, 25 x 13
	//debug("hf_out1:");
	//printECO_FEATS(hf_out1);

	// 4:compute the operation corresponding to the regularization term(convolve
	// each feature dimension with the DFT of w, and the tramsposed operation)
	// add the regularization part hf_conv = cell(1, 1, num_features);
	for (size_t i = 0; i < (size_t)num_features; i++) // for each feature
	{
		int reg_pad = cv::min(reg_filter[i].cols - 1, fAndDel[i][0].cols - 1);
		for (size_t j = 0; j < fAndDel[i].size(); j++) // for each dimension
		{
			// add part needed for convolution
			int c = fAndDel[i][j].cols;
			cv::Mat hf_conv;
			if (reg_pad == 0) // 
			{
				hf_conv = fAndDel[i][j];
			}
			else
			{
				hf_conv = fAndDel[i][j].colRange(c - reg_pad - 1, c - 1).clone();
				rot90(hf_conv, 3);
				cv::hconcat(fAndDel[i][j], mat_conj(hf_conv), hf_conv);
			}

			// do first convolution: W * f
			hf_conv = complexConvolution(hf_conv, reg_filter[i]);

			// do final convolution and put together result
			hf_conv =
				complexConvolution(hf_conv.colRange(0, hf_conv.cols - reg_pad),
								   reg_filter[i], 1);

			// A^H * A * f + W^H * W * f
			hf_out1[i][j] += hf_conv;
		}
	}

	// Stuff related to the projection matrix------------------------------

	// B * deltaP = X * inti(f)(before GC . previous  NG) * delta(P)
	vector<cv::Mat> BP_cell =
		FeatureComputeScores(FeatureProjection(init_samplef, deltaP), init_hf);

	cv::Mat BP(cv::Mat::zeros(BP_cell[k1].size(), BP_cell[k1].type()));
	for (size_t i = 0; i < scores.size(); i++)
	{
		int pad = (output_sz.height - BP_cell[i].rows) / 2;
		cv::Rect roi = cv::Rect(pad, pad, BP_cell[i].cols, BP_cell[i].rows);
		//cv::Mat temp = BP_cell[i] + BP(roi);
		//temp.copyTo(BP(roi));
		BP(roi) = BP_cell[i] + BP(roi);
	}

	// A^H * B * dP
	ECO_FEATS fBP, shBP;
	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		vector<cv::Mat> vfBP, vshBP;
		for (size_t j = 0; j < hf_out1[i].size(); j++)
		{
			int pad = (output_sz.height - hf_out1[i][0].rows) / 2;
			cv::Rect roi = cv::Rect(pad, pad, hf_out1[i][0].cols, hf_out1[i][0].rows);
			cv::Mat temp =
				complexDotMultiplication(BP(roi),
										 mat_conj(samplesf[i][j].clone()));
			// A^H * A * f + W^H * W * f + A^H * B * dp
			hf_out1[i][j] += temp;

			vfBP.push_back(
				complexDotMultiplication(mat_conj(init_hf[i][j].clone()),
										 BP(roi))); // B^H * BP
			vshBP.push_back(
				complexDotMultiplication(mat_conj(init_hf[i][j].clone()),
										 sh(roi))); // B^H * A * f
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
		part1 = 2 * real2complex(real(part1)) +
				params_.projection_reg * deltaP[i];

		// Compute proj matrix part : B^H * A * f
		cv::Mat part2 = XH[i] * FeatureVectorization(shBP)[i].t() -
						XH[i].colRange(fi, c_len) * FeatureVectorization(shBP)[i].colRange(fi, c_len).t();
		part2 = 2 * real2complex(real(part2));

		hf_out2.push_back(part1 + part2); // B^H * A * f + B^H * B * dp
	}

	ECO_EQ AX;
	AX.up_part_ = hf_out1;
	AX.low_part_ = hf_out2;
	return AX;
}

//************************************************************************
//      			Only filter training(for tracker update)
//************************************************************************
void EcoTrain::train_filter(const vector<ECO_FEATS> &samplesf,
							const vector<float> &sample_weights,
							const ECO_FEATS &sample_energy)
{
	//1: Construct the right hand side vector
	// sum up all the samples with the weights.
	ECO_FEATS rhs_samplef = samplesf[0] * sample_weights[0];
	for (size_t i = 1; i < samplesf.size(); i++)
	{
		rhs_samplef = samplesf[i] * sample_weights[i] +
					  rhs_samplef;
	}
	rhs_samplef = FeatureVectorMultiply(rhs_samplef, yf_, 1); //A^H * y

	//2: Construct preconditioner diag_M(exactly the same as in train_joint())
	ECO_FEATS diag_M;
	for (size_t i = 0; i < sample_energy.size(); i++)
	{
		cv::Mat mean(cv::Mat::zeros(sample_energy[i][0].size(), CV_32FC2));
		for (size_t j = 0; j < sample_energy[i].size(); j++)
		{
			mean += sample_energy[i][j];
		}
		mean = mean / sample_energy[i].size();

		vector<cv::Mat> temp_vec;
		for (size_t j = 0; j < sample_energy[i].size(); j++)
		{
			cv::Mat temp = (1 - params_.precond_data_param) * mean + params_.precond_data_param * sample_energy[i][j];
			temp = temp * (1 - params_.precond_reg_param) + params_.precond_reg_param * reg_energy_[i] * cv::Mat::ones(sample_energy[i][0].size(), CV_32FC2);
			temp_vec.push_back(temp);
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
}

ECO_FEATS EcoTrain::pcg_eco_filter(const vector<ECO_FEATS> &samplesf,
								   const vector<cv::Mat> &reg_filter,
								   const vector<float> &sample_weights,
								   const ECO_FEATS &rhs_samplef,
								   const ECO_FEATS &diag_M,
								   const ECO_FEATS &hf)
{
	int maxit = params_.CG_opts.maxit;
	bool existM1 = true;
	if (diag_M.empty())
	{
		existM1 = false;
	}
	ECO_FEATS x = hf; // initialization of CG

	// Load the CG state
	ECO_FEATS p, r_prev;
	float rho = 1, rho1, alpha, beta;

	if (!state_.p.empty())
	{
		p = state_.p;
		rho = state_.rho / params_.CG_opts.init_forget_factor;
		if (!params_.CG_opts.CG_use_FR) // Use Polak-Ribiere
		{
			r_prev = state_.r_prev;
		}
	}

	// calculate A(x)
	ECO_FEATS Ax = lhs_operation_filter(x,
										samplesf,
										reg_filter,
										sample_weights);
	ECO_FEATS r = rhs_samplef - Ax;

	for (size_t ii = 0; ii < (size_t)maxit; ii++)
	{
		ECO_FEATS y, z;
		if (existM1) // exist preconditioner
		{
			y = FeatureDotDivide(r, diag_M);
		}
		else
		{
			y = r;
		}
		z = y;

		rho1 = rho;
		rho = inner_product_filter(r, z);
		if ((rho == 0) || (std::abs(rho) >= INT_MAX) || std::isnan(rho))
		{
			break;
		}

		if (ii == 0 && p.empty())
		{
			p = z;
		}
		else
		{
			if (params_.CG_opts.CG_use_FR) // Use Fletcher-Reeves
			{
				beta = rho / rho1;
			}
			else // Use Polak-Ribiere
			{
				float rho2 = inner_product_filter(r_prev, z);
				beta = (rho - rho2) / rho1;
			}
			if ((beta == 0) || (std::abs(beta) >= INT_MAX) || std::isnan(beta))
			{
				break;
			}
			beta = cv::max(0.0f, beta);
			p = z + p * beta;
		}

		ECO_FEATS q = lhs_operation_filter(p,
										   samplesf,
										   reg_filter,
										   sample_weights);

		float pq = inner_product_filter(p, q);

		if (pq <= 0 || (std::abs(pq) > INT_MAX) || std::isnan(pq))
		{
			assert(0 && "error: GC condition is not matched");
		}
		else
		{
			if (params_.CG_opts.CG_standard_alpha)
			{
				alpha = rho / pq; // standard alpha
			}
			else
			{
				alpha = inner_product_filter(p, r) / pq;
			}
		}
		if ((std::abs(alpha) > INT_MAX) || std::isnan(alpha))
		{
			assert(0 && "GC condition alpha is not matched");
		}

		// Save old r if not using FR formula for beta
		if (!params_.CG_opts.CG_use_FR) // Use Polak-Ribiere
		{
			r_prev = r;
		}
		// form new iterate
		x = x + p * alpha;

		if (ii < (size_t)maxit)
		{
			r = r - q * alpha;
		}
	}

	state_.p = p;
	state_.rho = rho;
	if (!params_.CG_opts.CG_use_FR) // Use Polak-Ribiere
	{
		state_.r_prev = r_prev;
	}
	return x;
}
// This is the left-hand-side operation in Conjugate Gradient
ECO_FEATS EcoTrain::lhs_operation_filter(const ECO_FEATS &hf,
										 const vector<ECO_FEATS> &samplesf,
										 const vector<cv::Mat> &reg_filter,
										 const vector<float> &sample_weights)
{
	// 1: Get sizes of each feature----------------------------------------
	int num_features = hf.size();
	vector<cv::Size> filter_sz;
	for (size_t i = 0; i < (size_t)num_features; i++)
	{
		filter_sz.push_back(hf[i][0].size());
	}
	// find the maximum of size and its index
	vector<cv::Size>::iterator pos = max_element(filter_sz.begin(), filter_sz.end(), SizeCompare);
	size_t k1 = pos - filter_sz.begin(); // index
	cv::Size output_sz = cv::Size(2 * pos->width - 1, pos->height);

	//2: sum over all features for each sample: A * f  #SLOW#-------------------
	// a. FeatureDotMultiply: dot multiply for each mat
	// b. FeatureComputeScores: sum up all the dimensions for each feature
	// c. sh: sum up all the features
	/*
	// samplesf: 30 x 1 x 10 x 25 x 13
	debug("samplesf: %lu samples", samplesf.size());
	for (size_t i = 0; i < samplesf[0].size(); i++)
	{
		debug("samplesf: %lu, %lu, %d x %d", i, samplesf[0][i].size(),
			  samplesf[0][i][0].rows, samplesf[0][i][0].cols);
	}
	// hf: 1 x 10 x 25 x 13
	for (size_t i = 0; i < hf.size(); i++)
	{
		debug("hf: %lu, %lu, %d x %d", i, hf[i].size(),
			  hf[i][0].rows, hf[i][0].cols);
	}
*/
	vector<cv::Mat> sh;							 // sum of all the features for each sample
	for (size_t s = 0; s < samplesf.size(); s++) // for each sample
	{
		vector<cv::Mat> scores = FeatureComputeScores(samplesf[s], hf);
		cv::Mat sh_one(cv::Mat::zeros(scores[k1].size(), scores[k1].type()));
		for (size_t i = 0; i < scores.size(); i++) // for each feature
		{
			int pad = (output_sz.height - scores[i].rows) / 2;
			cv::Rect roi = cv::Rect(pad, pad, scores[i].cols, scores[i].rows);
			sh_one(roi) = scores[i] + sh_one(roi);
		}
		sh_one = mat_conj(sh_one * sample_weights[s]);
		sh.push_back(sh_one);
	}
	// sh: 30 x 25 x 13
	//	debug("sh: %lu x %d x %d", sh.size(), sh[0].rows, sh[0].cols);

	//3: multiply with the transpose : A^H * A * f  #SLOW#---------------------
	// update train time3_1_3: 0.004738
	ECO_FEATS hf_out;
	//	debug("num_features:%d", num_features);
	for (size_t i = 0; i < (size_t)num_features; i++) // for each feature
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < hf[i].size(); j++) // for each dimension
		{
			int pad = (output_sz.height - hf[i][j].rows) / 2;
			cv::Mat res(cv::Mat::zeros(hf[i][j].size(), hf[i][j].type()));
			for (size_t s = 0; s < sh.size(); s++) // for each sample
			{
				cv::Mat roi = sh[s](cv::Rect(pad, pad, hf[i][j].cols, hf[i][j].rows));
				res += complexDotMultiplication(roi, samplesf[s][i][j]);
			}
			tmp.push_back(mat_conj(res));
		}
		hf_out.push_back(tmp);
	}
	/*
	// hf_out: 1 x 10 x 25 x 13
	for (size_t i = 0; i < hf.size(); i++)
	{
		debug("hf_out: %lu, %lu, %d x %d", i, hf_out[i].size(),
			  hf_out[i][0].rows, hf_out[i][0].cols);
	}
*/

	//4: compute the operation corresponding to the regularization term--------
	for (size_t i = 0; i < (size_t)num_features; i++) // for each feature
	{
		int reg_pad = cv::min(reg_filter[i].cols - 1, hf[i][0].cols - 1);
		for (size_t j = 0; j < hf[i].size(); j++) // for each dimension
		{
			// add part needed for convolution
			int c = hf[i][j].cols;
			cv::Mat hf_conv;
			if (reg_pad == 0)
			{
				hf_conv = hf[i][j];
			}
			else
			{
				hf_conv = hf[i][j].colRange(c - reg_pad - 1, c - 1).clone();
				rot90(hf_conv, 3);
				cv::hconcat(hf[i][j], mat_conj(hf_conv), hf_conv);
			}
			// do first convolution: W * f
			hf_conv = complexConvolution(hf_conv, reg_filter[i]);

			// do final convolution and put toghether result
			hf_conv =
				complexConvolution(hf_conv.colRange(0, hf_conv.cols - reg_pad),
								   reg_filter[i], 1);

			// A^H * A * f + W^H * W * f
			hf_out[i][j] += hf_conv;
		}
	}
	// 1 x 10 x 25 x 13
	//	debug("hf_out: %lu x %lu x %d x %d", hf_out.size(), hf_out[0].size(),
	//		  hf_out[0][0].rows, hf_out[0][0].cols);
	return hf_out;
}

//************************************************************************
//      			Joint structure basic operation
//************************************************************************
EcoTrain::ECO_EQ EcoTrain::jointDotDivision(const ECO_EQ &a,
											const ECO_EQ &b)
{
	ECO_EQ res;
	ECO_FEATS up_rs;
	std::vector<cv::Mat> low_rs;
	for (size_t i = 0; i < a.up_part_.size(); i++) // for each feature
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < a.up_part_[i].size(); j++) // for each dimension
		{
			tmp.push_back(complexDotDivision(a.up_part_[i][j], b.up_part_[i][j]));
		}
		up_rs.push_back(tmp);
		low_rs.push_back(complexDotDivision(a.low_part_[i], b.low_part_[i]));
	}
	res.up_part_ = up_rs;
	res.low_part_ = low_rs;
	return res;
}

float EcoTrain::inner_product_joint(const ECO_EQ &a, const ECO_EQ &b)
{
	float ip = 0;
	for (size_t i = 0; i < a.up_part_.size(); i++) // for each feature
	{
		for (size_t j = 0; j < a.up_part_[i].size(); j++) // for each dimension
		{
			int clen = a.up_part_[i][j].cols;
			ip +=
				2 * mat_sum_f(real(complexDotMultiplication(
						mat_conj(a.up_part_[i][j].clone()), b.up_part_[i][j]))) -
				mat_sum_f(real(complexDotMultiplication(
					mat_conj(a.up_part_[i][j].col(clen - 1).clone()),
					b.up_part_[i][j].col(clen - 1))));
		}
		ip += mat_sum_f(real(complexDotMultiplication(
			mat_conj(a.low_part_[i].clone()), b.low_part_[i])));
	}
	return ip;
}

float EcoTrain::inner_product_filter(const ECO_FEATS &a, const ECO_FEATS &b)
{
	float ip = 0;
	for (size_t i = 0; i < a.size(); i++) // for each feature
	{
		for (size_t j = 0; j < a[i].size(); j++) // for each dimension
		{
			int clen = a[i][j].cols;
			ip +=
				2 * mat_sum_f(real(complexDotMultiplication(
						mat_conj(a[i][j].clone()), b[i][j]))) -
				mat_sum_f(real(complexDotMultiplication(
					mat_conj(a[i][j].col(clen - 1).clone()),
					b[i][j].col(clen - 1))));
		}
	}
	return ip;
}
//==============================================================================
EcoTrain::ECO_EQ EcoTrain::ECO_EQ::operator+(const ECO_EQ data)
{
	ECO_EQ res;
	res.up_part_ = up_part_ + data.up_part_;
	res.low_part_ = low_part_ + data.low_part_;
	return res;
}

EcoTrain::ECO_EQ EcoTrain::ECO_EQ::operator-(const ECO_EQ data)
{
	ECO_EQ res;
	res.up_part_ = up_part_ - data.up_part_;
	res.low_part_ = low_part_ - data.low_part_;
	return res;
}

EcoTrain::ECO_EQ EcoTrain::ECO_EQ::operator*(const float scale)
{
	ECO_EQ res;
	res.up_part_ = up_part_ * scale;
	res.low_part_ = low_part_ * scale;
	return res;
}

} // namespace eco