#include "eco.hpp"

namespace eco
{
void ECO::init(cv::Mat &im, const cv::Rect2f &rect)
{
	printf("\n=========================================================\n");
	// 1. Initialize all the parameters.
	// Image infomations
	imgInfo(im);
	debug("rect: %f, %f, %f, %f", rect.x, rect.y, rect.width, rect.height);

	// Get the ini position
	pos_.x = rect.x + (rect.width - 1.0) / 2.0;
	pos_.y = rect.y + (rect.height - 1.0) / 2.0;
	debug("pos_:%f, %f", pos_.y, pos_.x);

	// Calculate search area and initial scale factor
	float search_area = rect.area() * std::pow(params_.search_area_scale, 2);
	debug("search_area:%f", search_area);
	if (search_area > params_.max_image_sample_size)
		currentScaleFactor_ = sqrt((float)search_area / params_.max_image_sample_size);
	else if (search_area < params_.min_image_sample_size)
		currentScaleFactor_ = sqrt((float)search_area / params_.min_image_sample_size);
	else
		currentScaleFactor_ = 1.0;
	debug("currentscale:%f", currentScaleFactor_);

	// target size at the initial scale
	base_target_size_ = cv::Size2f(rect.size().width / currentScaleFactor_, rect.size().height / currentScaleFactor_);
	debug("base_target_size_:%f x %f", base_target_size_.height, base_target_size_.width);
	// window size, taking padding into account
	float img_sample_size__tmp = sqrt(base_target_size_.area() * std::pow(params_.search_area_scale, 2));
	img_sample_size_ = cv::Size2i(img_sample_size__tmp, img_sample_size__tmp);
	debug("img_sample_size_: %d x %d", img_sample_size_.height, img_sample_size_.width);

	init_features();

	// Number of Fourier coefficients to save for each filter layer. This will be an odd number.
	output_index_ = 0;
	output_size_ = 0;
	// The size of the label function DFT. Equal to the maximum filter size
	for (size_t i = 0; i != feature_size_.size(); ++i)
	{
		size_t size = feature_size_[i].width + (feature_size_[i].width + 1) % 2; //=63, to make it as an odd number;
		filter_size_.push_back(cv::Size(size, size));
		debug("filter_size_ %lu: %d x %d", i, filter_size_[i].height, filter_size_[i].width);
		// get the largest feature and it's index;
		output_index_ = size > output_size_ ? i : output_index_;
		output_size_ = std::max(size, output_size_);
	}
	debug("output_index_:%lu, output_size_:%lu", output_index_, output_size_);

	// Compute the 2d Fourier series indices by kx and ky.
	for (size_t i = 0; i < filter_size_.size(); ++i) // for each filter
	{
		cv::Mat_<float> tempy(filter_size_[i].height, 1, CV_32FC1);
		cv::Mat_<float> tempx(1, filter_size_[i].height / 2 + 1, CV_32FC1); // why 1/2 in x?===========????

		// ky in [-(N-1)/2, (N-1)/2], because N = filter_size_[i].height is odd (check above), N x 1;
		for (int j = 0; j < tempy.rows; j++)
		{
			tempy.at<float>(j, 0) = j - (tempy.rows / 2); // y index
		}
		ky_.push_back(tempy);
		// kx in [-N/2, 0], 1 x (N / 2 + 1)
		for (int j = 0; j < tempx.cols; j++)
		{
			tempx.at<float>(0, j) = j - (filter_size_[i].height / 2);
		}
		kx_.push_back(tempx);
		debug("For filter i: %lu, N: %d, ky:%d x %d, kx:%d x %d", i, filter_size_[i].height,
			  ky_[i].size().height, ky_[i].size().width, kx_[i].size().height, kx_[i].size().width);
	}

	// Construct the Gaussian label function using Poisson formula
	yf_gaussian();

	// Construct cosine window
	cos_window();

	// Compute Fourier series of interpolation function, refer C-COT
	for (size_t i = 0; i < filter_size_.size(); ++i)
	{
		cv::Mat interp1_fs1, interp2_fs1;
		Interpolator::get_interp_fourier(filter_size_[i], interp1_fs1, interp2_fs1, params_.interpolation_bicubic_a);
		interp1_fs_.push_back(interp1_fs1);
		interp2_fs_.push_back(interp2_fs1);
		//imgInfo(interp1_fs1);
		//showmat2chall(interp1_fs1, 2);
		//showmat2chall(interp2_fs1, 2);
	}

	// Construct spatial regularization filter, refer SRDCF
	for (size_t i = 0; i < filter_size_.size(); i++)
	{
		cv::Mat temp_d = get_regularization_filter(img_support_size_, base_target_size_, params_);
		cv::Mat temp_f;
		temp_d.convertTo(temp_f, CV_32FC1);
		reg_filter_.push_back(temp_f);
		debug("reg_filter_ %lu:", i);
		showmat1chall(temp_f, 2);

		// Compute the energy of the filter (used for preconditioner)drone_flip
		cv::Mat_<double> t = temp_d.mul(temp_d); //element-wise multiply
		float energy = mat_sumd(t);				 //sum up all the values of each points of the mat
		reg_energy_.push_back(energy);
		debug("reg_energy_ %lu: %f", i, energy);
	}

	// scale factor, 5 scales, refer SAMF
	if (params_.number_of_scales % 2 == 0)
	{
		params_.number_of_scales++;
	}
	int scalemin = floor((1.0 - (float)params_.number_of_scales) / 2.0);
	int scalemax = floor(((float)params_.number_of_scales - 1.0) / 2.0);
	for (int i = scalemin; i <= scalemax; i++)
	{
		scale_factors_.push_back(std::pow(params_.scale_step, i));
	}
	if (params_.number_of_scales > 0)
	{
		params_.min_scale_factor = //0.01;
			std::pow(params_.scale_step, std::ceil(std::log(std::fmax(5 / (float)img_support_size_.width,
																	 5 / (float)img_support_size_.height)) /
												  std::log(params_.scale_step)));
		params_.max_scale_factor = //10;
			std::pow(params_.scale_step, std::floor(std::log(std::fmin(im.cols / (float)base_target_size_.width,
																	  im.rows / (float)base_target_size_.height)) /
												   std::log(params_.scale_step)));
	}
	debug("scale:%d, %d", scalemin, scalemax);
	debug("scalefactor min: %f max: %f", params_.min_scale_factor, params_.max_scale_factor);
	debug("scale_factors_:");
	for (size_t i = 0; i < params_.number_of_scales; i++)
	{
		printf("%lu:%f; ", i, scale_factors_[i]);
	}
	printf("\n======================================================================\n");
	ECO_FEATS xl, xlw, xlf, xlf_porj;

	// 2. Extract features from the first frame.
	xl = feature_extractor_.extractor(im, pos_, vector<float>(1, currentScaleFactor_), params_, deep_mean_mat_, net_);
	debug("xl size: %lu, %lu, %d x %d", xl.size(), xl[0].size(), xl[0][0].rows, xl[0][0].cols);

	// 3. Do windowing of features.
	xl = do_windows(xl, cos_window_);

	// 4. Compute the fourier series.
	xlf = do_dft(xl);

	// 5. Interpolate features to the continuous domain.
	xlf = interpolate_dft(xlf, interp1_fs_, interp2_fs_);

	
	xlf = compact_fourier_coeff(xlf); // take half of the cols
	for (size_t i = 0; i < xlf.size(); i++)
	{
		debug("xlf feature %lu 's size: %lu, %d x %d", i, xlf[i].size(), xlf[i][0].rows, xlf[i][0].cols);
	}
	// 6. Initialize projection matrix P.
	projection_matrix_ = init_projection_matrix(xl, compressed_dim_, feature_dim_); // 32FC2 31x10
	for (size_t i = 0; i < projection_matrix_.size(); i++)
	{
		debug("projection_matrix %lu 's size: %d x %d", i, projection_matrix_[i].rows, projection_matrix_[i].cols);
	}
	// 7. project sample, feature reduction.
	xlf_porj = FeatureProjection(xlf, projection_matrix_);
	for (size_t i = 0; i < xlf.size(); i++)
	{
		debug("xlf_porj feature %lu 's size: %lu, %d x %d", i, xlf_porj[i].size(), xlf_porj[i][0].rows, xlf_porj[i][0].cols);
	}
	// 8. Initialize and update sample space.
	// The distance matrix, kernel matrix and prior weight are also updated
	sample_update_.init(filter_size_, compressed_dim_, params_.nSamples);
	sample_update_.update_sample_space_model(xlf_porj);

	// 9. Calculate sample energy and projection map energy.
	sample_energy_ = FeautreComputePower2(xlf_porj);

	vector<cv::Mat> proj_energy = project_mat_energy(projection_matrix_, yf_);

	// 10. Initialize filter and it's derivative.
	ECO_FEATS hf, hf_inc;
	for (size_t i = 0; i < xlf.size(); i++)
	{
		hf.push_back(vector<cv::Mat>(xlf_porj[i].size(), cv::Mat::zeros(xlf_porj[i][0].size(), CV_32FC2)));
		hf_inc.push_back(vector<cv::Mat>(xlf_porj[i].size(), cv::Mat::zeros(xlf_porj[i][0].size(), CV_32FC2)));
	}
	// 11. Train the tracker.
	eco_trainer_.train_init(hf, hf_inc, projection_matrix_, xlf, yf_, reg_filter_,
						   sample_energy_, reg_energy_, proj_energy, params_);

	eco_trainer_.train_joint();

	// 12. Update project matrix P. 
	projection_matrix_ = eco_trainer_.get_proj(); //*** exect to matlab tracker
	for (size_t i = 0; i < projection_matrix_.size(); ++i)
	{
		double maxValue = 0, minValue = 0;
		cv::minMaxLoc(projection_matrix_[i], &minValue, &maxValue, NULL, NULL);
		debug("projection_matrix_ %lu: value: %lf %lf", i, minValue, maxValue);
	}
	// 13. Re-project the sample and update the sample space.
	xlf_porj = FeatureProjection(xlf, projection_matrix_);
	debug("xlf_porj size: %lu, %lu, %d x %d", xlf_porj.size(), xlf_porj[0].size(), xlf_porj[0][0].rows, xlf_porj[0][0].cols);

	sample_update_.replace_sample(xlf_porj, 0);

	// 14. Update distance matrix of sample space. Find the norm of the reprojected sample
	float new_sample_norm = FeatureComputeEnergy(xlf_porj);
	sample_update_.set_gram_matrix(0, 0, 2 * new_sample_norm);

	// 15. Update filter f.
	hf_full_ = full_fourier_coeff(eco_trainer_.get_hf());
	for (size_t i = 0; i < hf_full_.size(); i++)
	{
		debug("hf_full_: %lu, %lu, %d x %d", i, hf_full_[i].size(), hf_full_[i][0].rows, hf_full_[i][0].cols);
	}
	frames_since_last_train_ = 0;
}

bool ECO::update(const cv::Mat &frame, cv::Rect2f &roi)
{
	//*****************************************************************************
	//*****                     Localization
	//*****************************************************************************
	double timereco = (double)cv::getTickCount();
	float fpseco = 0;

	cv::Point sample_pos = cv::Point(pos_);
	vector<float> samples_scales;
	for (size_t i = 0; i < scale_factors_.size(); ++i)
	{
		samples_scales.push_back(currentScaleFactor_ * scale_factors_[i]);
	}
	
	// 1: Extract features at multiple resolutions
	ECO_FEATS xt = feature_extractor_.extractor(frame, sample_pos, samples_scales, params_, deep_mean_mat_, net_);
	//debug("xt size: %lu, %lu, %d x %d", xt.size(), xt[0].size(), xt[0][0].rows, xt[0][0].cols);

	// 2:  project sample *****
	ECO_FEATS xt_proj = FeatureProjectionMultScale(xt, projection_matrix_);
	//debug("xt_proj size: %lu, %lu, %d x %d", xt_proj.size(), xt_proj[0].size(), xt_proj[0][0].rows, xt_proj[0][0].cols);

	// 3: Do windowing of features ***
	xt_proj = do_windows(xt_proj, cos_window_);

	// 4: Compute the fourier series ***
	ECO_FEATS xtf_proj = do_dft(xt_proj);

	// 5: Interpolate features to the continuous domain
	xtf_proj = interpolate_dft(xtf_proj, interp1_fs_, interp2_fs_);
	//debug("xtf_proj size: %lu, %lu, %d x %d", xtf_proj.size(), xtf_proj[0].size(), xtf_proj[0][0].rows, xtf_proj[0][0].cols);
	
	// 6: Compute the scores in Fourier domain for different scales of target
	vector<cv::Mat> scores_fs_sum;
	for (size_t i = 0; i < scale_factors_.size(); i++)
		scores_fs_sum.push_back(cv::Mat::zeros(filter_size_[output_index_], CV_32FC2));
	//debug("scores_fs_sum: %lu, %d x %d", scores_fs_sum.size(), scores_fs_sum[0].rows, scores_fs_sum[0].cols);

	for (size_t i = 0; i < xtf_proj.size(); i++) // for each feature
	{
		int pad = (filter_size_[output_index_].height - xtf_proj[i][0].rows) / 2;
		cv::Rect temp_roi = cv::Rect(pad, pad, xtf_proj[i][0].cols, xtf_proj[i][0].rows);

		for (size_t j = 0; j < xtf_proj[i].size(); j++) // for each dimension of the feature
		{
			// debug("%lu, %lu", j, j / hf_full_[i].size());
			scores_fs_sum[j / hf_full_[i].size()](temp_roi) +=
				complexMultiplication(xtf_proj[i][j], hf_full_[i][j % hf_full_[i].size()]);
		}
	}

	// 7: Calculate score by inverse DFT:

	// 8: Optimize the continuous score function with Newton's method.
	OptimizeScores scores(scores_fs_sum, params_.newton_iterations);
	scores.compute_scores();

	// Compute the translation vector in pixel-coordinates and round to the closest integer pixel.
	int scale_change_factor = scores.get_scale_ind();
	float resize_scores_width = (img_support_size_.width / output_size_) * currentScaleFactor_ * scale_factors_[scale_change_factor];
	float resize_scores_height = (img_support_size_.height / output_size_) * currentScaleFactor_ * scale_factors_[scale_change_factor];
	float dx = scores.get_disp_col() * resize_scores_width;
	float dy = scores.get_disp_row() * resize_scores_height;
	//debug("scale_change_factor:%d, get_disp_col: %f, get_disp_row: %f, dx: %f, dy: %f",
	//	  scale_change_factor, scores.get_disp_col(), scores.get_disp_row(), dx, dy);

	// 9: Update position and scale
	pos_ = cv::Point2f(sample_pos) + cv::Point2f(dx, dy);
	currentScaleFactor_ = currentScaleFactor_ * scale_factors_[scale_change_factor];

	// Adjust the scale to make sure we are not too large or too small
	if (currentScaleFactor_ < params_.min_scale_factor)
	{
		currentScaleFactor_ = params_.min_scale_factor;
	}
	else if (currentScaleFactor_ > params_.max_scale_factor)
	{
		currentScaleFactor_ = params_.max_scale_factor;
	}

	fpseco = ((double)cv::getTickCount() - timereco) / 1000000;
	//debug("localization time: %f", fpseco);
	//*****************************************************************************
	//*****                       Visualization
	//*****************************************************************************
	if (DEBUG == 1)
	{
		cv::Mat resframe = frame.clone();
		cv::rectangle(resframe, roi, cv::Scalar(0, 255, 0));

		// Apply the colormap
		std::vector<cv::Mat> scores_sum; //= scores_fs_sum;

		// Do inverse fft to the scores in the Fourier domain back to the spacial domain
		for (size_t i = 0; i < scores_fs_sum.size(); ++i)
		{
			int area = scores_fs_sum[i].size().area();
			// debug("area: %d", area);
			cv::Mat tmp = fftf(fftshift(scores_fs_sum[i], 1, 1, 1), 1); // inverse dft
			tmp = fftshift(tmp, 1, 1, 1);
			scores_sum.push_back(real(tmp * area)); // spacial domain only contains real part
		}

		cv::Mat cm_tmp, cm_img;
		cm_tmp = magnitude(scores_sum[scale_change_factor]) * 1000;
		cm_tmp.convertTo(cm_tmp, CV_8U);

		cv::resize(cm_tmp, cm_img, cv::Size(cm_tmp.cols * resize_scores_width, cm_tmp.rows * resize_scores_height),
				   0, 0, cv::INTER_LINEAR);
		cv::applyColorMap(cm_img, cm_img, cv::COLORMAP_JET);

		 
		// Merge these two images
		float alpha_vis = 0.5;
		int x_vis = std::max(0, sample_pos.x - cm_img.cols / 2);
		int y_vis = std::max(0, sample_pos.y - cm_img.rows / 2);
		int w_vis = 0;
		int h_vis = 0;
		if (sample_pos.x + cm_img.cols / 2 > resframe.cols)
		{
			w_vis = resframe.cols - x_vis;
		}
		else
		{
			w_vis = sample_pos.x + cm_img.cols / 2 - x_vis;
		}
		if (sample_pos.y + cm_img.rows / 2 > resframe.rows)
		{
			h_vis = resframe.rows - y_vis;
		}
		else
		{
			h_vis = sample_pos.y + cm_img.rows / 2 - y_vis;
		}
		//debug("%d %d %d %d",x_vis, y_vis, w_vis, h_vis);
		cv::Mat roi_vis = resframe(cv::Rect(x_vis, y_vis, w_vis, h_vis));
		 
		if (x_vis == 0)
		{
			x_vis = cm_img.cols / 2 - sample_pos.x;
		}
		else
		{
			x_vis = 0;
		}
		if (y_vis == 0)
		{
			y_vis = cm_img.rows / 2 - sample_pos.y;
		}
		else
		{
			y_vis = 0;
		}
		//debug("%d %d %d %d",x_vis, y_vis, w_vis, h_vis);
		cv::Mat cm_img_vis = cm_img(cv::Rect(x_vis, y_vis, w_vis, h_vis));
		 
		cv::addWeighted(cm_img_vis, alpha_vis, roi_vis, 1.0 - alpha_vis, 0.0, roi_vis);
		// Add a circle to the expected next position
		cv::circle(resframe, pos_, 5, cv::Scalar(0, 255, 0));
		cv::circle(resframe, sample_pos, 5, cv::Scalar(0, 255, 225));

		cv::imshow("Tracking", resframe);
		int c = cvWaitKey(1);
		if (c != -1)
			c = c % 256;
		if (c == 27)
		{
			cvDestroyWindow("Tracking");
			assert(0);
		}
		//cv::waitKey(0);
	}
	//*****************************************************************************
	//*****                     Training
	//*****************************************************************************
	timereco = (double)cv::getTickCount();
	// 1: Get the sample calculated in localization
	ECO_FEATS xlf_proj;
	for (size_t i = 0; i < xtf_proj.size(); ++i)
	{
		std::vector<cv::Mat> tmp;
		int start_ind = scale_change_factor * projection_matrix_[i].cols;
		int end_ind = (scale_change_factor + 1) * projection_matrix_[i].cols;
		for (size_t j = start_ind; j < (size_t)end_ind; ++j)
		{
			tmp.push_back(xtf_proj[i][j].colRange(0, xtf_proj[i][j].rows / 2 + 1));
		}
		xlf_proj.push_back(tmp);
	}
	//debug("xlf_proj size: %lu, %lu, %d x %d", xlf_proj.size(), xlf_proj[0].size(), xlf_proj[0][0].rows, xlf_proj[0][0].cols);

	// 2: Shift the sample so that the target is centered,
	//  A shift in spatial domain means multiply by exp(i pi L k), according to shift property of Fourier transformation.
	cv::Point2f shift_samp = 2.0f * CV_PI * cv::Point2f(pos_ - cv::Point2f(sample_pos)) * (1.0f / (currentScaleFactor_ * img_support_size_.width));
	xlf_proj = shift_sample(xlf_proj, shift_samp, kx_, ky_);
	//debug("shift_sample: %f %f", shift_samp.x, shift_samp.y);
	 
	// 3: Update the samples space to include the new sample, the distance matrix,
	// kernel matrix and prior weight are also updated
	sample_update_.update_sample_space_model(xlf_proj);
	 
	// merge new sample or replace
	if (sample_update_.get_merge_id() > 0)
	{
		sample_update_.replace_sample(xlf_proj, sample_update_.get_merge_id());
	}
	if (sample_update_.get_new_id() > 0)
	{
		sample_update_.replace_sample(xlf_proj, sample_update_.get_new_id());
	}
	 
	// 4: Train the tracker every Nsth frame, Ns in ECO paper
	bool train_tracker = frames_since_last_train_ >= (size_t)params_.train_gap;
	if (train_tracker)
	{
		//debug("%lu %lu", sample_energy_.size(), FeautreComputePower2(xlf_proj).size());
		sample_energy_ = FeatureScale(sample_energy_, 1 - params_.learning_rate) +
						FeatureScale(FeautreComputePower2(xlf_proj), params_.learning_rate);
		eco_trainer_.train_filter(sample_update_.get_samples(), sample_update_.get_samples_weight(), sample_energy_);
		frames_since_last_train_ = 0;
	}
	else
	{
		++frames_since_last_train_;
	}
	// 5: Update projection matrix P.
	projection_matrix_ = eco_trainer_.get_proj(); 

	// 6: Update filter f.
	hf_full_ = full_fourier_coeff(eco_trainer_.get_hf());

	fpseco = ((double)cv::getTickCount() - timereco) / 1000000;
	debug("training time: %f", fpseco);
	//*****************************************************************************
	//*****                    			Return
	//******************************************************************************
	roi.width = base_target_size_.width * currentScaleFactor_;
	roi.height = base_target_size_.height * currentScaleFactor_;
	roi.x = pos_.x - roi.width / 2;
	roi.y = pos_.y - roi.height / 2;
	//debug("roi:%f, %f, %f, %f", roi.x, roi.y, roi.width, roi.height);

	return true;
}

void ECO::init_features()
{
	// Init features parameters---------------------------------------
	if (params_.useDeepFeature)
	{
		if (params_.use_gpu)
		{
			printf("Setting up Caffe in GPU mode with ID: %d\n", params_.gpu_id);
			caffe::Caffe::set_mode(caffe::Caffe::GPU);
			caffe::Caffe::SetDevice(params_.gpu_id);
		}
		else
		{
			printf("Setting up Caffe in CPU mode\n");
			caffe::Caffe::set_mode(caffe::Caffe::CPU);
		}

		net_.reset(new Net<float>(params_.cnn_features.fparams.proto, TEST)); // Read prototxt
		net_->CopyTrainedLayersFrom(params_.cnn_features.fparams.model);		// Read model
		read_deep_mean(params_.cnn_features.fparams.mean_file);				// Read mean file

		//	showmat3ch(deep_mean_mat_, 2);
		//	showmat3ch(deep_mean_mean_mat_, 2);

		params_.cnn_features.img_input_sz = img_sample_size_; //250
		params_.cnn_features.img_sample_sz = img_sample_size_;

		// Calculate the output size of the 2 output layer;
		// matlab version pad can be unbalanced, but caffe cannot for the moment;
		int cnn_output_sz0 = (int)((img_sample_size_.width - 7 + 0 + 0) / 2) + 1; //122
		//int cnn_output_sz1 = (int)((cnn_output_sz0 - 3 + 0 + 1) / 2) + 1;	  //61 matlab version
		int cnn_output_sz1 = (int)((cnn_output_sz0 - 3 + 0 + 0) / 2) + 1; //61
		cnn_output_sz1 = (int)((cnn_output_sz1 - 3 + 1 + 1) / 2) + 1;	 //15
		//cnn_output_sz1 = (int)((cnn_output_sz1 - 3 + 0 + 1) / 2) + 1;		   //15 matlab version
		cnn_output_sz1 = (int)((cnn_output_sz1 - 3 + 0 + 0) / 2) + 1; //15
		int total_feature_sz0 = cnn_output_sz0;
		int total_feature_sz1 = cnn_output_sz1;
		debug("total_feature_sz: %d %d", total_feature_sz0, total_feature_sz1);
		// Re-calculate the output size of the 1st output layer;
		int support_sz = params_.cnn_features.fparams.stride[1] * cnn_output_sz1;	// 16 x 15 = 240
		cnn_output_sz0 = (int)(support_sz / params_.cnn_features.fparams.stride[0]); // 240 / 2 = 120
		debug("cnn_output_sz: %d %d", cnn_output_sz0, cnn_output_sz1);

		int start_ind0 = (int)((total_feature_sz0 - cnn_output_sz0) / 2) + 1; // 2
		int start_ind1 = (int)((total_feature_sz1 - cnn_output_sz1) / 2) + 1; // 1
		int end_ind0 = start_ind0 + cnn_output_sz0 - 1;						  // 121
		int end_ind1 = start_ind1 + cnn_output_sz1 - 1;						  // 15
		//debug("ind: %d %d %d %d", start_ind0, start_ind1, end_ind0, end_ind1);

		params_.cnn_features.fparams.start_ind = {start_ind0, start_ind0, start_ind1, start_ind1};

		params_.cnn_features.fparams.end_ind = {end_ind0, end_ind0, end_ind1, end_ind1};

		params_.cnn_features.data_sz_block0 = cv::Size(cnn_output_sz0 / params_.cnn_features.fparams.downsample_factor[0],
													  cnn_output_sz0 / params_.cnn_features.fparams.downsample_factor[0]);
		params_.cnn_features.data_sz_block1 = cv::Size(cnn_output_sz1 / params_.cnn_features.fparams.downsample_factor[1],
													  cnn_output_sz1 / params_.cnn_features.fparams.downsample_factor[1]);

		params_.cnn_features.mean = deep_mean_mean_mat_;

		img_support_size_ = cv::Size(support_sz, support_sz);

		debug("cnn parameters--------------:");
		debug("img_input_sz: %d, img_sample_size_: %d", params_.cnn_features.img_input_sz.width, params_.cnn_features.img_sample_sz.width);
		debug("data_sz_block0: %d, data_sz_block1: %d", params_.cnn_features.data_sz_block0.width, params_.cnn_features.data_sz_block1.width);
		debug("start_ind0: %d, start_ind1: %d, end_ind0: %d, end_ind1: %d", params_.cnn_features.fparams.start_ind[0],
			  params_.cnn_features.fparams.start_ind[2], params_.cnn_features.fparams.end_ind[0], params_.cnn_features.fparams.end_ind[2]);
	}
	else
	{
		net_ = boost::shared_ptr<Net<float>>();
		img_support_size_ = img_sample_size_;
	}
	if (params_.useHogFeature) // just HOG feature;
	{
		params_.hog_features.img_input_sz = img_support_size_;
		params_.hog_features.img_sample_sz = img_support_size_;
		params_.hog_features.data_sz_block0 = cv::Size(params_.hog_features.img_sample_sz.width / params_.hog_features.fparams.cell_size,
													  params_.hog_features.img_sample_sz.height / params_.hog_features.fparams.cell_size);

		debug("HOG parameters---------------:");
		debug("img_input_sz: %d, img_sample_size_: %d", params_.hog_features.img_input_sz.width, params_.hog_features.img_sample_sz.width);
		debug("data_sz_block0: %d", params_.hog_features.data_sz_block0.width);
		debug("Finish------------------------");
	}
	if (params_.useCnFeature)
	{
	}
	debug("img_support_size_:%d x %d", img_support_size_.width, img_support_size_.height);

	// features setting-----------------------------------------------------
	if (params_.useDeepFeature)
	{
		feature_size_.push_back(params_.cnn_features.data_sz_block0);
		feature_size_.push_back(params_.cnn_features.data_sz_block1);
		feature_dim_ = params_.cnn_features.fparams.nDim;
		compressed_dim_ = params_.cnn_features.fparams.compressed_dim;
	}
	if (params_.useHogFeature)
	{
		feature_size_.push_back(params_.hog_features.data_sz_block0);			  //=62x62
		feature_dim_.push_back(params_.hog_features.fparams.nDim);			  //=31
		compressed_dim_.push_back(params_.hog_features.fparams.compressed_dim); //=10
	}
	if (params_.useCnFeature)
	{
	}
	for (size_t i = 0; i < feature_size_.size(); i++)
	{
		debug("features %lu: %d %d %d x %d", i, feature_dim_[i], compressed_dim_[i], feature_size_[i].height, feature_size_[i].width);
	}
}

void ECO::read_deep_mean(const string &mean_file)
{
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	// Convert from BlobProto to Blob<float>
	int num_channels_ = 3;
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	//* The format of the mean file is planar 32-bit float BGR or grayscale.
	std::vector<cv::Mat> channels;
	float *data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i)
	{
		// Extract an individual channel.
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	// Merge the separate channels into a single image.
	cv::merge(channels, deep_mean_mat_);

	// Get the mean for each channel.
	deep_mean_mean_mat_ = cv::Mat(cv::Size(224, 224), deep_mean_mat_.type(), cv::mean(deep_mean_mat_));
}

void ECO::yf_gaussian() // real part of (9) in paper C-COT
{
	// sig_y is sigma in (9)
	double sig_y = sqrt(int(base_target_size_.width) * int(base_target_size_.height)) *
				   (params_.output_sigma_factor) * (float(output_size_) / img_support_size_.width);
	debug("sig_y:%lf", sig_y);
	for (unsigned int i = 0; i < ky_.size(); i++) // for each filter
	{
		// 2 dimension version of (9)
		cv::Mat tempy(ky_[i].size(), CV_32FC1);
		tempy = CV_PI * sig_y * ky_[i] / output_size_;
		cv::exp(-2 * tempy.mul(tempy), tempy);
		tempy = sqrt(2 * CV_PI) * sig_y / output_size_ * tempy;

		cv::Mat tempx(kx_[i].size(), CV_32FC1);
		tempx = CV_PI * sig_y * kx_[i] / output_size_;
		cv::exp(-2 * tempx.mul(tempx), tempx);
		tempx = sqrt(2 * CV_PI) * sig_y / output_size_ * tempx;

		yf_.push_back(cv::Mat(tempy * tempx)); // matrix multiplication
											  /*
		showmat1chall(tempy, 2);
		showmat1chall(tempx, 2);
		showmat1chall(yf_[i], 2);
		*/
	}
}

void ECO::cos_window()
{
	for (size_t i = 0; i < feature_size_.size(); i++)
	{
		cv::Mat hann1t = cv::Mat(cv::Size(feature_size_[i].width + 2, 1), CV_32F, cv::Scalar(0));
		cv::Mat hann2t = cv::Mat(cv::Size(1, feature_size_[i].height + 2), CV_32F, cv::Scalar(0));
		for (int i = 0; i < hann1t.cols; i++)
			hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
		for (int i = 0; i < hann2t.rows; i++)
			hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));
		cv::Mat hann2d = hann2t * hann1t;
		cos_window_.push_back(hann2d(cv::Range(1, hann2d.rows - 1), cv::Range(1, hann2d.cols - 1)));
		/*
		showmat1ch(cos_window_[i],2);
		assert(0);
		*/
	}
}

ECO_FEATS ECO::interpolate_dft(const ECO_FEATS &xlf, vector<cv::Mat> &interp1_fs, vector<cv::Mat> &interp2_fs)
{
	ECO_FEATS result;

	for (size_t i = 0; i < xlf.size(); i++)
	{
		cv::Mat interp1_fs_mat = subwindow(interp1_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp1_fs[i].rows, interp1_fs[i].rows)), IPL_BORDER_REPLICATE);
		cv::Mat interp2_fs_mat = subwindow(interp2_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp2_fs[i].cols, interp2_fs[i].cols)), IPL_BORDER_REPLICATE);
		vector<cv::Mat> temp;
		for (size_t j = 0; j < xlf[i].size(); j++)
		{
			temp.push_back(complexMultiplication(complexMultiplication(interp1_fs_mat, xlf[i][j]), interp2_fs_mat));
		}
		result.push_back(temp);
	}
	return result;
}

ECO_FEATS ECO::compact_fourier_coeff(const ECO_FEATS &xf)
{
	ECO_FEATS result;
	for (size_t i = 0; i < xf.size(); i++) // for each feature
	{
		vector<cv::Mat> temp;
		for (size_t j = 0; j < xf[i].size(); j++) // for each dimension of the feature
			temp.push_back(xf[i][j].colRange(0, (xf[i][j].cols + 1) / 2));
		result.push_back(temp);
	}
	return result;
}

ECO_FEATS ECO::full_fourier_coeff(const ECO_FEATS &xf)
{
	ECO_FEATS res;
	for (size_t i = 0; i < xf.size(); i++) // for each feature
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < xf[i].size(); j++) // for each dimension of the feature
		{
			cv::Mat temp = xf[i][j].colRange(0, xf[i][j].cols - 1).clone();
			rot90(temp, 3);
			cv::hconcat(xf[i][j], mat_conj(temp), temp);
			tmp.push_back(temp);
		}
		res.push_back(tmp);
	}

	return res;
}

vector<cv::Mat> ECO::project_mat_energy(vector<cv::Mat> proj, vector<cv::Mat> yf)
{
	vector<cv::Mat> result;

	for (size_t i = 0; i < yf.size(); i++)
	{
		cv::Mat temp(proj[i].size(), CV_32FC1), temp_compelx;
		float sum_dim = std::accumulate(feature_dim_.begin(), feature_dim_.end(), 0.0f);
		cv::Mat x = yf[i].mul(yf[i]);
		temp = 2 * mat_sum(x) / sum_dim * cv::Mat::ones(proj[i].size(), CV_32FC1);
		result.push_back(temp);
	}
	return result;
}

// Shift a sample in the Fourier domain. The shift should be normalized to the range [-pi, pi]
ECO_FEATS ECO::shift_sample(ECO_FEATS &xf, cv::Point2f shift, std::vector<cv::Mat> kx, std::vector<cv::Mat> ky)
{
	ECO_FEATS res;

	for (size_t i = 0; i < xf.size(); ++i) // for each feature
	{
		cv::Mat shift_exp_y(ky[i].size(), CV_32FC2), shift_exp_x(kx[i].size(), CV_32FC2);
		for (size_t j = 0; j < (size_t)ky[i].rows; j++)
		{
			shift_exp_y.at<COMPLEX>(j, 0) = COMPLEX(cos(shift.y * ky[i].at<float>(j, 0)), sin(shift.y * ky[i].at<float>(j, 0)));
		}
		for (size_t j = 0; j < (size_t)kx[i].cols; j++)
		{
			shift_exp_x.at<COMPLEX>(0, j) = COMPLEX(cos(shift.x * kx[i].at<float>(0, j)), sin(shift.x * kx[i].at<float>(0, j)));
		}
		/*
		debug("shift_exp_y:");
		showmat2chall(shift_exp_y, 2);
		debug("shift_exp_x:");
		showmat2chall(shift_exp_x, 2);
		assert(0);
		*/
		cv::Mat shift_exp_y_mat = subwindow(shift_exp_y, cv::Rect(cv::Point(0, 0), xf[i][0].size()), IPL_BORDER_REPLICATE);
		cv::Mat shift_exp_x_mat = subwindow(shift_exp_x, cv::Rect(cv::Point(0, 0), xf[i][0].size()), IPL_BORDER_REPLICATE);

		vector<cv::Mat> tmp;
		for (size_t j = 0; j < xf[i].size(); j++) // for each dimension of the feature, do complex element-wise multiplication
		{
			tmp.push_back(complexMultiplication(complexMultiplication(shift_exp_y_mat, xf[i][j]), shift_exp_x_mat));
		}
		res.push_back(tmp);
	}
	return res;
}
} // namespace eco
