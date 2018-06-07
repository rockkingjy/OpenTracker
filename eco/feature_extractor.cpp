#include "feature_extractor.h"

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)
#define ddebug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

ECO_FEATS feature_extractor::extractor(cv::Mat image,
									   cv::Point2f pos,
									   vector<float> scales,
									   const eco_params &params,
									   const cv::Mat &yml_mean,
									   const boost::shared_ptr<Net<float>> &net)
{
	int num_features = 1, num_scales = scales.size();
	hog_features = params.hog_features;
	if (params.useDeepFeature)
	{
		num_features = 2;
		cnn_features = params.cnn_features;
		this->net = net;
	}

	// extract image path for different kinds of feautures
	vector<vector<cv::Mat>> img_samples;
	for (int i = 0; i < num_features; ++i)
	{
		vector<cv::Mat> img_samples_temp(num_scales);
		for (unsigned int j = 0; j < scales.size(); ++j) //for different scales
		{
			cv::Size2f img_sample_sz = (i == 0) && params.useDeepFeature ? cnn_features.img_sample_sz : hog_features.img_sample_sz;
			cv::Size2f img_input_sz = (i == 0) && params.useDeepFeature ? cnn_features.img_input_sz : hog_features.img_input_sz;
			img_sample_sz.width *= scales[j];
			img_sample_sz.height *= scales[j];
			img_samples_temp[j] = sample_patch(image, pos, img_sample_sz, img_input_sz, params);
		}
		img_samples.push_back(img_samples_temp);
	}

	// Extract image patches features(all kinds of features)
	ECO_FEATS sum_features;
	if (params.useDeepFeature)
	{
		sum_features = get_cnn_layers(img_samples[0], yml_mean);
		cnn_feature_normalization(sum_features);
	}
	hog_feat_maps = get_hog(img_samples[img_samples.size() - 1]);
	vector<cv::Mat> hog_maps_vec = hog_feature_normalization(hog_feat_maps);

	sum_features.push_back(hog_maps_vec);
	return sum_features;
}

cv::Mat feature_extractor::sample_patch(const cv::Mat &im,
										const cv::Point2f &poss,
										cv::Size2f sample_sz,
										cv::Size2f output_sz,
										const eco_params &gparams)
{
	cv::Point pos(poss.operator cv::Point());

	// Downsample factor
	float resize_factor = std::min(sample_sz.width / output_sz.width, sample_sz.height / output_sz.height);
	int df = std::max((float)floor(resize_factor - 0.1), float(1));
	cv::Mat new_im;
	im.copyTo(new_im);
	if (df > 1)
	{
		cv::Point os((pos.x - 0) % df, ((pos.y - 0) % df));
		pos.x = (pos.x - os.x - 1) / df + 1;
		pos.y = (pos.y - os.y - 1) / df + 1;

		sample_sz.width = sample_sz.width / df;
		sample_sz.height = sample_sz.height / df;

		int r = (im.rows - os.y) / df + 1, c = (im.cols - os.x) / df;
		cv::Mat new_im2(r, c, im.type());

		new_im = new_im2;
		//int m, n;
		for (size_t i = 0 + os.y, m = 0; i < (size_t)im.rows && m < (size_t)new_im.rows; i += df, ++m)
			for (size_t j = 0 + os.x, n = 0; j < (size_t)im.cols && n < (size_t)new_im.cols; j += df, ++n)
				if (im.channels() == 1)
					new_im.at<uchar>(m, n) = im.at<uchar>(i, j);
				else
					new_im.at<cv::Vec3b>(m, n) = im.at<cv::Vec3b>(i, j);
	}

	// *** extract image ***
	sample_sz.width = round(sample_sz.width);
	sample_sz.height = round(sample_sz.height);
	cv::Point pos2(pos.x - floor((sample_sz.width + 1) / 2) + 1, pos.y - floor((sample_sz.height + 1) / 2) + 1);

	//debug("new_im:%d, %d, pos:%d, %d, sample_sz:%f, %f", new_im.cols, new_im.rows, pos2.x, pos2.y, sample_sz.width, sample_sz.height);
	cv::Mat im_patch;
	if (sample_sz.width - pos2.x > 0 && sample_sz.height - pos2.y > 0)
	{
		im_patch = RectTools::subwindow(new_im, cv::Rect(pos2, sample_sz), IPL_BORDER_REPLICATE);
	}
	else
	{
		im_patch = RectTools::subwindow(new_im, cv::Rect(cv::Point(0, 0), new_im.size()), IPL_BORDER_REPLICATE);
	}

	cv::Mat resized_patch;
	cv::resize(im_patch, resized_patch, output_sz);
	return resized_patch;
}

vector<cv::Mat> feature_extractor::get_hog(vector<cv::Mat> ims)
{
	if (ims.empty())
		return vector<cv::Mat>();

	vector<cv::Mat> hog_feats;
	for (unsigned int i = 0; i < ims.size(); i++)
	{
		cv::Mat temp;
		ims[i].convertTo(temp, CV_32FC3);
		// Create object
		HogFeature FHOG(hog_features.fparams.cell_size, 1);
		cv::Mat features = FHOG.getFeature(temp); //*** Extract FHOG features****
		hog_feats.push_back(features);
	}
	return hog_feats;
}

ECO_FEATS feature_extractor::get_cnn_layers(vector<cv::Mat> im, const cv::Mat &yml_mean)
{
	for (unsigned int i = 0; i < im.size(); ++i)
	{
		im[i].convertTo(im[i], CV_32FC3);
		cv::cvtColor(im[i], im[i], CV_BGR2RGB);
		im[i] = im[i].t();
		im[i] = im[i] - yml_mean;
	}
	cv::Mat input_imgs;
	cv::merge(im, input_imgs);
	//**** forward computation and exrtract cnn1 and output data ***************
	Blob<float> *input_layer = net->input_blobs()[0];

	input_layer->Reshape(im.size(), im[0].channels(), 224, 224);
	net->Reshape();
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);
	cv::split(input_imgs, input_channels);

	net->Forward();
	ECO_FEATS feature_map;
	for (size_t idx = 0; idx < cnn_features.fparams.output_layer.size(); ++idx)
	{
		const float *pstart = NULL;
		vector<int> shape;
		if (cnn_features.fparams.downsample_factor[idx] == 2)
		{
			boost::shared_ptr<caffe::Blob<float>> layerData = net->blob_by_name("norm1");
			pstart = layerData->cpu_data();
			shape = layerData->shape();
		}
		else
		{
			Blob<float> *output_layer = net->output_blobs()[0];
			pstart = output_layer->cpu_data();
			shape = output_layer->shape();
		}
		vector<cv::Mat> merge_feature;
		for (size_t i = 0; i < (size_t)(shape[0] * shape[1]); i++) //  CNN into single channel******
		{
			cv::Mat feat_map(shape[2], shape[3], CV_32FC(1), (void *)pstart);
			feat_map = feat_map.t();
			//const float* inData = feat_map.ptr<float>(0);
			//inData = pstart;
			pstart += shape[2] * shape[3];

			//  extract features according to fparamss
			cnn_params fparams = cnn_features.fparams;
			cv::Mat extract_map = feat_map(cv::Range(fparams.start_ind[0 + 2 * idx] - 1, fparams.end_ind[0 + 2 * idx]),
										   cv::Range(fparams.start_ind[0 + 2 * idx] - 1, fparams.end_ind[0 + 2 * idx]));

			extract_map = (cnn_features.fparams.downsample_factor[idx] == 1) ? extract_map : sample_pool(extract_map, 2, 2);
			merge_feature.push_back(extract_map);
		} //end for
		feature_map.push_back(merge_feature);
	}
	return feature_map;
}

void feature_extractor::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
	Blob<float> *input_layer = net->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float *input_data = input_layer->mutable_cpu_data();

	for (int i = 0; i < input_layer->channels() * input_layer->shape()[0]; ++i)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

cv::Mat feature_extractor::sample_pool(const cv::Mat &im, int smaple_factor, int stride)
{
	if (im.empty())
		return cv::Mat();
	cv::Mat new_im(im.cols / 2, im.cols / 2, CV_32FC1);
	for (size_t i = 0; i < (size_t)new_im.rows; i++)
	{
		for (size_t j = 0; j < (size_t)new_im.cols; j++)
			new_im.at<float>(i, j) = 0.25 * (im.at<float>(2 * i, 2 * j) + im.at<float>(2 * i, 2 * j + 1) +
											 im.at<float>(2 * i + 1, 2 * j) + im.at<float>(2 * i + 1, 2 * j + 1));
	}
	return new_im;
}

void feature_extractor::cnn_feature_normalization(ECO_FEATS &cnn_feat_maps)
{
	for (size_t i = 0; i < cnn_feat_maps.size(); i++)
	{
		//*** the normalization scale ****
		vector<cv::Mat> temp = cnn_feat_maps[i]; // *** conv1 norm1 *****
		vector<float> sum_scales;
		for (size_t s = 0; s < temp.size(); s += cnn_features.fparams.nDim[i]) // *** for an scale ***
		{
			float sum = 0.0f;
			for (size_t j = s; j < s + cnn_features.fparams.nDim[i]; j++)
				sum += cv::sum(temp[j].mul(temp[j]))[0];
			sum_scales.push_back(sum);
		}
		//*** the normalization para ****
		float para = 0.0f;
		if (i == 0)
			para = cnn_features.data_sz_block1.area() * cnn_features.fparams.nDim[i];
		else
			para = cnn_features.data_sz_block2.area() * cnn_features.fparams.nDim[i];

		for (unsigned int k = 0; k < temp.size(); k++) //*** normalization ****
			cnn_feat_maps[i][k] /= sqrt(sum_scales[k / cnn_features.fparams.nDim[i]] / para);
	}
}

vector<cv::Mat> feature_extractor::hog_feature_normalization(vector<cv::Mat> &hog_feat_maps)
{
	vector<cv::Mat> hog_maps_vec;
	for (size_t i = 0; i < hog_feat_maps.size(); i++)
	{
		cv::Mat temp = hog_feat_maps[i];
		temp = temp.mul(temp);
		// float  sum_scales = cv::sum(temp)[0]; *** sum can not work !! while dimension exceeding 3
		vector<cv::Mat> temp_vec, result_vec;
		float sum = 0;
		cv::split(temp, temp_vec);
		for (int j = 0; j < temp.channels(); j++)
			sum += cv::sum(temp_vec[j].mul(temp_vec[j]))[0];
		float para = hog_features.data_sz_block1.area() * hog_features.fparams.nDim;
		hog_feat_maps[i] /= sqrt(sum / para);

		cv::split(hog_feat_maps[i], result_vec);
		hog_maps_vec.insert(hog_maps_vec.end(), result_vec.begin(), result_vec.end());
	}

	return hog_maps_vec;
}