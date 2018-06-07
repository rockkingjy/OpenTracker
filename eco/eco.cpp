
#include "eco.h"

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)
#define ddebug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

namespace eco
{
void ECO::init(cv::Mat &im, const cv::Rect &rect)
{
	if (params.useDeepFeature)
	{
		yml_mean = meanMatFromYML(params.cnn_features.fparams.mean_yml);
		if (params.cnn_features.fparams.proto.empty() || params.cnn_features.fparams.model.empty())
			assert("the proto or model is empty");

		if (params.use_gpu)
		{
			printf("Setting up Caffe in GPU mode with ID: %d\n", params.gpu_id);
			caffe::Caffe::set_mode(caffe::Caffe::GPU);
			caffe::Caffe::SetDevice(params.gpu_id);
		}
		else
		{
			printf("Setting up Caffe in CPU mode\n");
			caffe::Caffe::set_mode(caffe::Caffe::CPU);
		}

		net.reset(new Net<float>(params.cnn_features.fparams.proto, TEST));
		net->CopyTrainedLayersFrom(params.cnn_features.fparams.model);

		// read mean file
		Blob<float> image_mean;
		BlobProto blob_proto;
		//const float *mean_ptr;
		//unsigned int num_pixel;
		bool succeed = ReadProtoFromBinaryFile(params.cnn_features.fparams.mean_file, &blob_proto);
		if (succeed)
		{
			image_mean.FromProto(blob_proto);
			//num_pixel = image_mean.count(); /* NCHW=1x3x224x224=196608 */
			//mean_ptr = (const float *)image_mean.cpu_data();
		}
		deep_mean_mat = deep_mean(params.cnn_features.fparams.mean_file);
	}
	else
	{
		net = boost::shared_ptr<Net<float>>();
	}

	// get the ini position
	pos.x = rect.x + (float)rect.width / 2;
	pos.y = rect.y + (float)rect.height / 2;
	debug("pos:%f, %f", pos.x, pos.y);
	target_sz = rect.size();

	// *** Calculate search area and initial scale factor ****
	int search_area = rect.area() * pow(params.search_area_scale, 2);
	debug("search_area:%d, %d", rect.area(), search_area);
	if (search_area > params.max_image_sample_size)
		currentScaleFactor = sqrt((float)search_area / params.max_image_sample_size);
	else if (search_area < params.min_image_sample_size)
		currentScaleFactor = sqrt((float)search_area / params.min_image_sample_size);
	else
		currentScaleFactor = 1.0;

	debug("currentscale:%f", currentScaleFactor);
	// *** target size at the initial scale ***
	base_target_sz = cv::Size2f(target_sz.width / currentScaleFactor, target_sz.height / currentScaleFactor);

	// *** window size, taking padding into account ***
	img_sample_sz = cv::Size(sqrt(base_target_sz.area() * pow(params.search_area_scale, 2)),
							sqrt(base_target_sz.area() * pow(params.search_area_scale, 2)));
	debug("img_sample_sz: %d, %d", img_sample_sz.width, img_sample_sz.height);
/*	if (currentScaleFactor > 1)
		img_sample_sz = cv::Size(250, 250);
	else
		img_sample_sz = cv::Size(200, 200); */

	init_features();

	if (params.useDeepFeature)
	{
		feature_sz.push_back(params.cnn_features.data_sz_block1);
		feature_sz.push_back(params.cnn_features.data_sz_block2);
		feature_dim = params.cnn_features.fparams.nDim;
		compressed_dim = params.cnn_features.fparams.compressed_dim;
	}
	feature_sz.push_back(params.hog_features.data_sz_block1);			   //=63x63
	feature_dim.push_back(params.hog_features.fparams.nDim);			   //=31
	compressed_dim.push_back(params.hog_features.fparams.compressed_dim); //=10
	for(size_t i = 0; i < feature_sz.size(); i++)
	{
		debug("features %lu: %d %d %d %d", i, feature_sz[i].width, feature_sz[i].height, feature_dim[i], compressed_dim[i]);
	}
	// *** Number of Fourier coefficients to save for each filter layer. This will be an odd number.
	max_output_index = 0;
	output_sz = 0;
	// The size of the label function DFT. Equal to the maximum filter size
	for (size_t i = 0; i != feature_sz.size(); ++i)
	{
		size_t size = feature_sz[i].width + (feature_sz[i].width + 1) % 2; //=63, to make it as an odd number;
		filter_sz.push_back(cv::Size(size, size));
		debug("filter_sz %lu: %d, %d", i, filter_sz[i].height, filter_sz[i].width);
		// get the largest feature and it's index;
		max_output_index = size > output_sz ? i : max_output_index;
		output_sz = std::max(size, output_sz);
	}
	debug("index:%lu,output:%lu", max_output_index, output_sz);

	// *** Compute the Fourier series indices k.
	for (size_t i = 0; i < filter_sz.size(); ++i) // for each filter
	{
		cv::Mat_<float> tempy(filter_sz[i].height, 1, CV_32FC1);
		cv::Mat_<float> tempx(1, filter_sz[i].height / 2 + 1, CV_32FC1); // why 1/2 in x?===========????

		// ky in [-(N-1)/2, (N-1)/2], because N = filter_sz[i].height is odd (check above), 63x1
		for (int j = 0; j < tempy.rows; j++)
		{
			tempy.at<float>(j, 0) = j - (tempy.rows / 2); // y index
		}
		ky.push_back(tempy);
		// kx in [-N/2, 0], 1x32
		for (int j = 0; j < tempx.cols; j++)
		{
			tempx.at<float>(0, j) = j - (filter_sz[i].height / 2);
		}
		kx.push_back(tempx);
		debug("i: %lu, N: %d, ky:%d %d, kx:%d %d", i, filter_sz[i].height,
					 ky[i].size().height, ky[i].size().width, kx[i].size().height, kx[i].size().width);
	}

	// *** construct the Gaussian label function using Poisson formula
	yf_gaussian();

	// *** construct cosine window
	cos_wind();

	//*** Compute Fourier series of interpolation function, refer C-COT
	for (size_t i = 0; i < filter_sz.size(); ++i)
	{
		cv::Mat interp1_fs1, interp2_fs1;
		interpolator::get_interp_fourier(filter_sz[i], interp1_fs1, interp2_fs1, params.interpolation_bicubic_a);

		interp1_fs.push_back(interp1_fs1);
		interp2_fs.push_back(interp2_fs1);
	}

	//*** Construct spatial regularization filter, refer SRDCF
	for (size_t i = 0; i < filter_sz.size(); i++)
	{
		cv::Mat temp = get_reg_filter(img_support_sz, base_target_sz, params);
		reg_filter.push_back(temp);
		// Compute the energy of the filter (used for preconditioner)drone_flip
		cv::Mat_<float> t = temp.mul(temp);  //element-wise multiply
		float energy = FFTTools::mat_sum(t); //sum up all the values of each points of the mat
		reg_energy.push_back(energy);
	}

	//*** scale factor, 5 scales, refer SAMF
	int scalemin = floor((1.0 - (float)params.number_of_scales) / 2.0);
	int scalemax = floor(((float)params.number_of_scales - 1.0) / 2.0);
	debug("scale:%d, %d", scalemin, scalemax);
	for (int i = scalemin; i <= scalemax; i++)
	{
		scaleFactors.push_back(pow(params.scale_step, i));
	}
	if(params.number_of_scales > 0)
	{
		params.min_scale_factor = //0.01;
		std::pow(params.scale_step, std::ceil(std::log(std::fmax(5 / (float)img_support_sz.width, 
								5 / (float)img_support_sz.height)) / std::log(params.scale_step)));
    	params.max_scale_factor = //10;
		std::pow(params.scale_step, std::floor(std::log(std::fmin(im.rows / (float)img_support_sz.width, 
								im.cols / (float)img_support_sz.height)) / std::log(params.scale_step)));

	}
	debug("scalefactor min: %f max: %f", params.min_scale_factor, params.max_scale_factor);

	ECO_FEATS xl, xlw, xlf, xlf_porj;
	
	xl = feat_extrator.extractor(im, pos, vector<float>(1, currentScaleFactor), params, yml_mean, net);
	
	//*** Do windowing of features ***
	xl = do_windows_x(xl, cos_window);
	
	//*** Compute the fourier series ***
	xlf = do_dft(xl);
	
	//*** Interpolate features to the continuous domain **
	xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);

	//*** New sample to be added
	xlf = compact_fourier_coeff(xlf);

	//*** Compress feature dementional projection matrix
	projection_matrix = init_projection_matrix(xl, compressed_dim, feature_dim); //*** EXACT EQUAL TO MATLAB

	//*** project sample *****
	xlf_porj = project_sample(xlf, projection_matrix);

	//*** Update the samples to include the new sample.
	// The distance matrix, kernel matrix and prior weight are also updated
	SampleUpdate.init(filter_sz, compressed_dim, params.nSamples);

	SampleUpdate.update_sample_space_model(xlf_porj);

	//**** used for precondition ******
	ECO_FEATS new_sample_energy = feats_pow2(xlf_porj);
	sample_energy = new_sample_energy;

	vector<cv::Mat> proj_energy = project_mat_energy(projection_matrix, yf);

	ECO_FEATS hf, hf_inc;
	for (size_t i = 0; i < xlf.size(); i++)
	{
		hf.push_back(vector<cv::Mat>(xlf_porj[i].size(), cv::Mat::zeros(xlf_porj[i][0].size(), CV_32FC2)));
		hf_inc.push_back(vector<cv::Mat>(xlf_porj[i].size(), cv::Mat::zeros(xlf_porj[i][0].size(), CV_32FC2)));
	}

	eco_trainer.train_init(hf, hf_inc, projection_matrix, xlf, yf, reg_filter,
						   new_sample_energy, reg_energy, proj_energy, params);

	eco_trainer.train_joint();

	//   reproject sample and updata sample space
	projection_matrix = eco_trainer.get_proj(); //*** exect to matlab tracker

	xlf_porj = project_sample(xlf, projection_matrix);

	SampleUpdate.replace_sample(xlf_porj, 0);

	//  Find the norm of the reprojected sample
	float new_sample_norm = FeatEnergy(xlf_porj); // equal to matlab
	SampleUpdate.set_gram_matrix(0, 0, 2 * new_sample_norm);

	frames_since_last_train = 0;

	hf_full = full_fourier_coeff(eco_trainer.get_hf());
}

bool ECO::update(const cv::Mat &frame, cv::Rect2d &roi)
{
	cv::Point sample_pos = cv::Point(pos);
	vector<float> det_samples_pos;
	for (size_t i = 0; i < scaleFactors.size(); ++i)
	{
		det_samples_pos.push_back(currentScaleFactor * scaleFactors[i]);
	}
	// 1: Extract features at multiple resolutions
	ECO_FEATS xt = feat_extrator.extractor(frame, sample_pos, det_samples_pos, params, yml_mean, net);

	// 2:  project sample *****
	ECO_FEATS xt_proj = FeatProjMultScale(xt, projection_matrix);
	
	// 3: Do windowing of features ***
	xt_proj = do_windows_x(xt_proj, cos_window);
	
	// 4: Compute the fourier series ***
	xt_proj = do_dft(xt_proj);
	
	// 5: Interpolate features to the continuous domain
	xt_proj = interpolate_dft(xt_proj, interp1_fs, interp2_fs);
	
	// 6: compute the scores for different scales of target
	// Compute convolution for each feature block in the Fourier domain and the sum over all blocks.
	// Also sum over all feature blocks. Gives the fourier coefficients of the convolution response.
	vector<cv::Mat> scores_fs_sum;
	for (size_t i = 0; i < scaleFactors.size(); i++)
		scores_fs_sum.push_back(cv::Mat::zeros(filter_sz[max_output_index], CV_32FC2));

	for (size_t i = 0; i < xt_proj.size(); i++)
	{
		int pad = (filter_sz[max_output_index].height - xt_proj[i][0].rows) / 2;
		cv::Rect temp_roi = cv::Rect(pad, pad, xt_proj[i][0].cols, xt_proj[i][0].rows);
	
		for (size_t j = 0; j < xt_proj[i].size(); j++)
		{
		/*	cv::Mat temp_score = complexMultiplication(xt_proj[i][j], hf_full[i][j % hf_full[i].size()]);
			temp_score += scores_fs_sum[j / hf_full[i].size()](temp_roi);
			temp_score.copyTo(scores_fs_sum[j / hf_full[i].size()](temp_roi)); */
			scores_fs_sum[j / hf_full[i].size()](temp_roi) += 
					complexMultiplication(xt_proj[i][j], hf_full[i][j % hf_full[i].size()]);
		}
	}
	
	// 7: Optimize the continuous score function with Newton's method.
	optimize_scores scores(scores_fs_sum, params.newton_iterations);
	scores.compute_scores();
	
	// 8: Compute the translation vector in pixel-coordinates and round to the closest integer pixel.
	int scale_change_factor = scores.get_scale_ind();
	float dx = scores.get_disp_col() * (img_support_sz.width / output_sz) * 
					currentScaleFactor * scaleFactors[scale_change_factor];
	float dy = scores.get_disp_row() * (img_support_sz.height / output_sz) * 
					currentScaleFactor * scaleFactors[scale_change_factor];
	
	// 9: Update position
	pos = cv::Point2f(sample_pos) + cv::Point2f(dx, dy);
	
	// 10: Update the scale
	currentScaleFactor = currentScaleFactor * scaleFactors[scale_change_factor];

	// 11: Adjust to make sure we are not too large or too small
	if(currentScaleFactor < params.min_scale_factor)
	{
		currentScaleFactor = params.min_scale_factor;
	}
	else if(currentScaleFactor > params.max_scale_factor)
	{
		currentScaleFactor = params.max_scale_factor;
	}

	//*****************************************************************************
	//*****                     Model update step
	//*****************************************************************************
	
	// 1: Use the sample that was used for detection
	ECO_FEATS xlf_proj;
	for (size_t i = 0; i < xt_proj.size(); ++i)
	{
		std::vector<cv::Mat> tmp;
		int start_ind = scale_change_factor * projection_matrix[i].cols;
		int end_ind = (scale_change_factor + 1) * projection_matrix[i].cols;
		for (size_t j = start_ind; j < (size_t)end_ind; ++j)
		{
			tmp.push_back(xt_proj[i][j].colRange(0, xt_proj[i][j].rows / 2 + 1));
		}
		xlf_proj.push_back(tmp);
	}

	// 2: Shift the sample so that the target is centered
	cv::Point2f shift_samp = 2 * CV_PI * cv::Point2f(pos - cv::Point2f(sample_pos))
							 * (1 / (currentScaleFactor * img_support_sz.width));
	xlf_proj = shift_sample(xlf_proj, shift_samp, kx, ky);

	// 3: Update the samples to include the new sample, the distance matrix, 
	// kernel matrix and prior weight are also updated
	SampleUpdate.update_sample_space_model(xlf_proj);
	ddebug();
	// 4: insert new sample
	if (SampleUpdate.get_merge_id() > 0)
	{
		SampleUpdate.replace_sample(xlf_proj, SampleUpdate.get_merge_id());
	}
	if (SampleUpdate.get_new_id() > 0)
	{
		SampleUpdate.replace_sample(xlf_proj, SampleUpdate.get_new_id());
	}
	// 5: update filter parameters, Ns in ECO paper
	bool train_tracker = frames_since_last_train >= (size_t)params.train_gap;
	if (train_tracker)
	{
		sample_energy = FeatScale(sample_energy, 1 - params.learning_rate) +
						FeatScale(feats_pow2(xlf_proj), params.learning_rate);
		eco_trainer.train_filter(SampleUpdate.get_samples(), SampleUpdate.get_samples_weight(), sample_energy);
		frames_since_last_train = 0;
	}
	else
	{
		++frames_since_last_train;
	}
	projection_matrix = eco_trainer.get_proj(); //*** exect to matlab tracker
	hf_full = full_fourier_coeff(eco_trainer.get_hf());
	//*****************************************************************************
	//*****                    return
	//******************************************************************************
	roi.width = base_target_sz.width * currentScaleFactor;
	roi.height = base_target_sz.height * currentScaleFactor;
	roi.x = pos.x - roi.width / 2;
	roi.y = pos.y - roi.height / 2;

	//roi = resbox;
	debug("roi:%f, %f, %f, %f", roi.x, roi.y, roi.width, roi.height);

	return true;
}

void ECO::init_features()
{
	if (params.useDeepFeature)
	{
		//******** the cnn feature intialization **********
		//params.cnn_features.fparams.start_ind = vector<int>({3, 3, 1, 1});
		//params.cnn_features.fparams.end_ind = vector<int>({106, 106, 13, 13});

		//params.cnn_features.img_input_sz = cv::Size(224, 224);
		params.cnn_features.img_sample_sz = img_sample_sz;

		std::vector<cv::Size> cnn_output_sz;
		cnn_output_sz.push_back(cv::Size(208 / 2, 208 / 2));   //**** the size of conv1, 109x109;
		cnn_output_sz.push_back(cv::Size(208 / 16, 208 / 16)); //**** the size of conv5, 13x13;
		params.cnn_features.data_sz_block1 = cv::Size(cnn_output_sz[0].width / params.cnn_features.fparams.downsample_factor[0],
											   cnn_output_sz[0].height / params.cnn_features.fparams.downsample_factor[0]);
		params.cnn_features.data_sz_block2 = cv::Size(cnn_output_sz[1].width / params.cnn_features.fparams.downsample_factor[1],
											   cnn_output_sz[1].height / params.cnn_features.fparams.downsample_factor[1]);
		params.cnn_features.mean = deep_mean_mat;

		//params.cnn_feat = params.cnn_features;

		img_support_sz = cv::Size(round((float)208 * img_sample_sz.width / params.cnn_features.img_input_sz.width),
								  round((float)208 * img_sample_sz.height / params.cnn_features.img_input_sz.height));
	}
	else // just HOG feature;
	{
		int new_sample_sz = (1 + img_sample_sz.width / params.hog_features.fparams.cell_size) * params.hog_features.fparams.cell_size;
		/*	debug("sample_sz:%d, %d", img_sample_sz.width, new_sample_sz);//=250,252
		int max_odd = -1, max_idx = -1;
		for (size_t i = 0; i < (size_t)params.hog_features.fparams.cell_size; i++)
		{
			int sz = (new_sample_sz + i) / params.hog_features.fparams.cell_size;
			if (sz % 2 == 1) // find the max odd value and it's id;
			{
				max_idx = max_odd >= sz ? max_idx : i;
				max_odd = max_odd >= sz ? max_odd : sz;
			}
		}

		new_sample_sz += max_idx; 
		debug("max:%d, %d, %d", max_idx, max_odd, new_sample_sz); //=0,63
	*/
		img_support_sz = cv::Size(new_sample_sz, new_sample_sz); //63x63
	}
	//******** the HOG feature intialization **********
	params.hog_features.img_input_sz = img_support_sz; //*** the imput-img-size of hog is the same as support size
	params.hog_features.img_sample_sz = img_support_sz;
	params.hog_features.data_sz_block1 = cv::Size(img_support_sz.width / params.hog_features.fparams.cell_size,
										   img_support_sz.height / params.hog_features.fparams.cell_size);
	debug("params.hog_features: %d, %d, %d, %d",img_support_sz.width, img_support_sz.height, 
				params.hog_features.data_sz_block1.width, params.hog_features.data_sz_block1.height);
	//params.hog_feat = params.hog_features;
}

cv::Mat ECO::meanMatFromYML(string path)
{
	vector<cv::Mat> resmat;
	string maen1 = path;
	cv::FileStorage fsDemo(maen1, cv::FileStorage::READ);

	cv::Mat mean1, mean2, mean3;
	fsDemo["x1"] >> mean1;
	fsDemo["x2"] >> mean2;
	fsDemo["x3"] >> mean3;
	resmat.push_back(mean1);
	resmat.push_back(mean2);
	resmat.push_back(mean3);

	cv::Mat res;
	cv::merge(resmat, res);

	return res;
}

cv::Mat ECO::deep_mean(const string &mean_file)
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

	//  Merge the separate channels into a single image.
	cv::Mat mean;
	cv::merge(channels, mean);

	cv::Scalar channel_mean = cv::mean(mean); //get the mean for each channel.
	return cv::Mat(cv::Size(224, 224), mean.type(), channel_mean);
}

void ECO::yf_gaussian() // real part of (9) in paper C-COT
{
	// sig_y is sigma in (9)
	float sig_y = sqrt(int(base_target_sz.width) * int(base_target_sz.height)) *
				  (params.output_sigma_factor) * (float(output_sz) / img_support_sz.width);

	for (unsigned int i = 0; i < ky.size(); i++) // for each filter
	{
		// 2 dimension version of (9)
		cv::Mat tempy(ky[i].size(), CV_32FC1);
		tempy = CV_PI * sig_y * ky[i] / output_sz;
		cv::exp(-2 * tempy.mul(tempy), tempy);
		tempy = sqrt(2 * CV_PI) * sig_y / output_sz * tempy;

		cv::Mat tempx(kx[i].size(), CV_32FC1);
		tempx = CV_PI * sig_y * kx[i] / output_sz;
		cv::exp(-2 * tempx.mul(tempx), tempx);
		tempx = sqrt(2 * CV_PI) * sig_y / output_sz * tempx;

		yf.push_back(cv::Mat(tempy * tempx));
	}
}

void ECO::cos_wind()
{
	for (size_t i = 0; i < feature_sz.size(); i++)
	{
		cv::Mat hann1t = cv::Mat(cv::Size(feature_sz[i].width + 2, 1), CV_32F, cv::Scalar(0));
		cv::Mat hann2t = cv::Mat(cv::Size(1, feature_sz[i].height + 2), CV_32F, cv::Scalar(0));
		for (int i = 0; i < hann1t.cols; i++)
			hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
		for (int i = 0; i < hann2t.rows; i++)
			hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));
		cv::Mat hann2d = hann2t * hann1t;
		cos_window.push_back(hann2d(cv::Range(1, hann2d.rows - 1), cv::Range(1, hann2d.cols - 1)));
	}
}

ECO_FEATS ECO::do_windows_x(const ECO_FEATS &xl, vector<cv::Mat> &cos_win)
{
	ECO_FEATS xlw;
	for (size_t i = 0; i < xl.size(); i++)
	{
		vector<cv::Mat> temp;
		for (size_t j = 0; j < xl[i].size(); j++)
			temp.push_back(cos_win[i].mul(xl[i][j]));
		xlw.push_back(temp);
	}
	return xlw;
}

ECO_FEATS ECO::interpolate_dft(const ECO_FEATS &xlf, vector<cv::Mat> &interp1_fs, vector<cv::Mat> &interp2_fs)
{
	ECO_FEATS result;

	for (size_t i = 0; i < xlf.size(); i++)
	{
		cv::Mat interp1_fs_mat = RectTools::subwindow(interp1_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp1_fs[i].rows, interp1_fs[i].rows)), IPL_BORDER_REPLICATE);
		cv::Mat interp2_fs_mat = RectTools::subwindow(interp2_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp2_fs[i].cols, interp2_fs[i].cols)), IPL_BORDER_REPLICATE);
		vector<cv::Mat> temp;
		for (size_t j = 0; j < xlf[i].size(); j++)
		{
			temp.push_back(precision(complexMultiplication(complexMultiplication(interp1_fs_mat, xlf[i][j]), interp2_fs_mat)));
		}
		result.push_back(temp);
	}
	return result;
}

ECO_FEATS ECO::compact_fourier_coeff(const ECO_FEATS &xf)
{
	ECO_FEATS result;
	for (size_t i = 0; i < xf.size(); i++)
	{
		vector<cv::Mat> temp;
		for (size_t j = 0; j < xf[i].size(); j++)
			temp.push_back(xf[i][j].colRange(0, (xf[i][j].cols + 1) / 2));
		result.push_back(temp);
	}
	return result;
}

vector<cv::Mat> ECO::init_projection_matrix(const ECO_FEATS &init_sample, const vector<int> &compressed_dim, const vector<int> &feature_dim)
{
	vector<cv::Mat> result;
	for (size_t i = 0; i < init_sample.size(); i++)
	{
		cv::Mat feat_vec(init_sample[i][0].size().area(), feature_dim[i], CV_32FC1);
		cv::Mat mean(init_sample[i][0].size().area(), feature_dim[i], CV_32FC1);
		for (unsigned int j = 0; j < init_sample[i].size(); j++)
		{
			float mean = cv::mean(init_sample[i][j])[0];
			for (size_t r = 0; r < (size_t)init_sample[i][j].rows; r++)
				for (size_t c = 0; c < (size_t)init_sample[i][j].cols; c++)
					feat_vec.at<float>(c * init_sample[i][j].rows + r, j) = init_sample[i][j].at<float>(r, c) - mean;
		}
		result.push_back(feat_vec);
	}

	vector<cv::Mat> proj_mat;
	//****** svd operation ******
	for (size_t i = 0; i < result.size(); i++)
	{
		cv::Mat S, V, D;
		cv::SVD::compute(result[i].t() * result[i], S, V, D);
		vector<cv::Mat> V_;
		V_.push_back(V);
		V_.push_back(cv::Mat::zeros(V.size(), CV_32FC1));
		cv::merge(V_, V);
		proj_mat.push_back(V.colRange(0, compressed_dim[i])); //** two channels : complex
	}

	return proj_mat;
}

//***** the train part ****
vector<cv::Mat> ECO::project_mat_energy(vector<cv::Mat> proj, vector<cv::Mat> yf)
{
	vector<cv::Mat> result;

	for (size_t i = 0; i < yf.size(); i++)
	{
		cv::Mat temp(proj[i].size(), CV_32FC1), temp_compelx;
		float sum_dim = std::accumulate(feature_dim.begin(), feature_dim.end(), 0);
		cv::Mat x = yf[i].mul(yf[i]);
		temp = 2 * FFTTools::mat_sum(x) / sum_dim * cv::Mat::ones(proj[i].size(), CV_32FC1);
		result.push_back(temp);
	}
	return result;
}

ECO_FEATS ECO::full_fourier_coeff(const ECO_FEATS &xf)
{
	ECO_FEATS res;
	for (size_t i = 0; i < xf.size(); i++)
	{
		vector<cv::Mat> tmp;
		for (size_t j = 0; j < xf[i].size(); j++)
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

ECO_FEATS ECO::shift_sample(ECO_FEATS &xf, cv::Point2f shift, std::vector<cv::Mat> kx, std::vector<cv::Mat> ky)
{
	ECO_FEATS res;

	for (size_t i = 0; i < xf.size(); ++i)
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

		cv::Mat shift_exp_y_mat = RectTools::subwindow(shift_exp_y, cv::Rect(cv::Point(0, 0), xf[i][0].size()), IPL_BORDER_REPLICATE);
		cv::Mat shift_exp_x_mat = RectTools::subwindow(shift_exp_x, cv::Rect(cv::Point(0, 0), xf[i][0].size()), IPL_BORDER_REPLICATE);

		vector<cv::Mat> tmp;
		for (size_t j = 0; j < xf[i].size(); j++)
		{
			tmp.push_back(complexMultiplication(complexMultiplication(shift_exp_y_mat, xf[i][j]), shift_exp_x_mat));
		}
		res.push_back(tmp);
	}
	return res;
}
} // namespace eco
