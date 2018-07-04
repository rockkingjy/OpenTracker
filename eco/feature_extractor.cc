#include "feature_extractor.hpp"
namespace eco{
ECO_FEATS FeatureExtractor::extractor(cv::Mat image,
									   cv::Point2f pos,
									   vector<float> scales,
									   const EcoParameters &params,
									   const cv::Mat &deep_mean_mat,
									   const boost::shared_ptr<Net<float>> &net)
{
	//double timereco = (double)cv::getTickCount();
	//float fpseco = 0;

	int num_features = 0, num_scales = scales.size();
	cv::Mat new_deep_mean_mat;
	if (params.useDeepFeature)
	{
		cnn_feat_ind = num_features;
		num_features++;
		cnn_features = params.cnn_features;
		this->net = net;
		resize(deep_mean_mat, new_deep_mean_mat, params.cnn_features.img_input_sz, 0, 0, cv::INTER_CUBIC);
	}
	if (params.useHogFeature)
	{
		hog_feat_ind = num_features;
		num_features++;
		hog_features = params.hog_features;
	}
	if (params.useCnFeature)
	{
	}

	// extract image path for different feautures
	vector<vector<cv::Mat>> img_samples;
	for (int i = 0; i < num_features; ++i) // for each feature
	{
		vector<cv::Mat> img_samples_temp(num_scales);

		for (unsigned int j = 0; j < scales.size(); ++j) // for each scale
		{
			cv::Size2f img_sample_sz = (i == 0) && params.useDeepFeature ? cnn_features.img_sample_sz : hog_features.img_sample_sz;
			cv::Size2f img_input_sz = (i == 0) && params.useDeepFeature ? cnn_features.img_input_sz : hog_features.img_input_sz;
			img_sample_sz.width *= scales[j];
			img_sample_sz.height *= scales[j];
			img_samples_temp[j] = sample_patch(image, pos, img_sample_sz, img_input_sz, params);
		}
		img_samples.push_back(img_samples_temp);
	}

	// Extract feature maps for each feature in the list
	ECO_FEATS sum_features;
	if (params.useDeepFeature)
	{
		sum_features = get_cnn_layers(img_samples[cnn_feat_ind], new_deep_mean_mat);
		cnn_feature_normalization(sum_features);
	}

	//cv::waitKey(0);
	if (params.useHogFeature)
	{
		hog_feat_maps = get_hog(img_samples[hog_feat_ind]);
		vector<cv::Mat> hog_maps_vec = hog_feature_normalization(hog_feat_maps);
		sum_features.push_back(hog_maps_vec);
	}

	if (params.useCnFeature)
	{
	}

	//fpseco = ((double)cv::getTickCount() - timereco) / 1000000;
	//debug("feature extra time: %f", fpseco);
	return sum_features;
}

cv::Mat FeatureExtractor::sample_patch(const cv::Mat &im,
										const cv::Point2f &posf,
										cv::Size2f sample_sz,
										cv::Size2f input_sz,
										const EcoParameters &gparams)
{
	// Pos should be integer when input, but floor in just in case.
	cv::Point pos(posf.operator cv::Point());

	// Downsample factor
	float resize_factor = std::min(sample_sz.width / input_sz.width, sample_sz.height / input_sz.height);
	int df = std::max((float)floor(resize_factor - 0.1), float(1));
	//	debug("resize_factor: %f,df: %d,sample_sz: %f x %f,input_sz: %f x %f,pos: %d %d", resize_factor, df,
	//			sample_sz.width, sample_sz.height, input_sz.width, input_sz.height, pos.y, pos.x);

	cv::Mat new_im;
	im.copyTo(new_im);
	//debug("new_im:%d x %d", new_im.rows, new_im.cols);
	if (df > 1)
	{
		cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
		pos.x = (pos.x - os.x - 1) / df + 1;
		pos.y = (pos.y - os.y - 1) / df + 1;

		sample_sz.width = sample_sz.width / df;
		sample_sz.height = sample_sz.height / df;

		int r = (im.rows - os.y) / df + 1;
		int c = (im.cols - os.x) / df;
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

	// make sure the size is not too small and round it
	sample_sz.width = std::max(round(sample_sz.width), 2.0f);
	sample_sz.height = std::max(round(sample_sz.height), 2.0f);

	cv::Point pos2(pos.x - floor((sample_sz.width + 1) / 2),
				   pos.y - floor((sample_sz.height + 1) / 2));
	//debug("new_im:%d x %d, pos2:%d %d, sample_sz:%f x %f", new_im.rows, new_im.cols, pos2.y, pos2.x,
	//	  sample_sz.height, sample_sz.width);
	//showmatNch(new_im, 0);

	cv::Mat im_patch;
	im_patch = subwindow(new_im, cv::Rect(pos2, sample_sz), IPL_BORDER_REPLICATE);

	cv::Mat resized_patch;
	cv::resize(im_patch, resized_patch, input_sz);

	return resized_patch;
}

vector<cv::Mat> FeatureExtractor::get_hog(vector<cv::Mat> ims)
{
	if (ims.empty())
		return vector<cv::Mat>();

	//	double timereco = (double)cv::getTickCount();
	//	float fpseco = 0;
	// debug("ims.size: %lu", ims.size());
	vector<cv::Mat> hog_feats;
	for (unsigned int i = 0; i < ims.size(); i++) // ims.size()=5
	{
		cv::Mat ims_f;
		ims[i].convertTo(ims_f, CV_32FC3);
		/*
		 
		imgInfo(ims_f);
	
		printImage(ims_f, 2);
		assert(0);
*/
		cv::Size _tmpl_sz;
		_tmpl_sz.width = ims_f.cols;
		_tmpl_sz.height = ims_f.rows;

		//		debug("%d, %d", _tmpl_sz.width, _tmpl_sz.height);

		int _cell_size = hog_features.fparams.cell_size;
		// Round to cell size and also make it even
		if (int(_tmpl_sz.width / (_cell_size)) % 2 == 0)
		{
			_tmpl_sz.width = ((int)(_tmpl_sz.width / (2 * _cell_size)) * 2 * _cell_size) + _cell_size * 2;
			_tmpl_sz.height = ((int)(_tmpl_sz.height / (2 * _cell_size)) * 2 * _cell_size) + _cell_size * 2;
		}
		else
		{
			_tmpl_sz.width = ((int)(_tmpl_sz.width / (2 * _cell_size)) * 2 * _cell_size) + _cell_size * 3;
			_tmpl_sz.height = ((int)(_tmpl_sz.height / (2 * _cell_size)) * 2 * _cell_size) + _cell_size * 3;
		}

		//		debug("%d, %d", _tmpl_sz.width, _tmpl_sz.height);

		// Add extra cell filled with zeros around the image
		cv::Mat featurePaddingMat(_tmpl_sz.height + _cell_size * 2,
								  _tmpl_sz.width + _cell_size * 2,
								  CV_32FC3, cvScalar(0, 0, 0));

		if (ims_f.cols != _tmpl_sz.width || ims_f.rows != _tmpl_sz.height)
		{
			resize(ims_f, ims_f, _tmpl_sz);
		}
		ims_f.copyTo(featurePaddingMat);
		/*
		 
		imgInfo(featurePaddingMat);
		assert(0);

		fpseco = ((double)cv::getTickCount() - timereco) / 1000000;
		debug("hog extra time: %f", fpseco);
		timereco = (double)cv::getTickCount();
		debug("%d %d", featurePaddingMat.cols, featurePaddingMat.rows);
*/
		IplImage zz = featurePaddingMat;
		CvLSVMFeatureMapCaskade *map_temp;
		getFeatureMaps(&zz, _cell_size, &map_temp); // dimension: 27
		normalizeAndTruncate(map_temp, 0.2f);		// dimension: 108
		PCAFeatureMaps(map_temp);					// dimension: 31

		cv::Mat featuresMap = cv::Mat(cv::Size(map_temp->sizeX, map_temp->sizeY), // Procedure do deal with cv::Mat multichannel bug
									  CV_32FC(map_temp->numFeatures), map_temp->map);

		// clone because map_temp will be free.
		featuresMap = featuresMap.clone();

		freeFeatureMapObject(&map_temp);
		/*
		 
		imgInfo(featuresMap);
		printFeature(featuresMap, 2);
		assert(0);
*/
		hog_feats.push_back(featuresMap);
	}

	//	fpseco = ((double)cv::getTickCount() - timereco) / 1000000;
	//	debug("hog extra time: %f", fpseco);

	return hog_feats;
}

ECO_FEATS FeatureExtractor::get_cnn_layers(vector<cv::Mat> im, const cv::Mat &deep_mean_mat)
{
	Blob<float> *input_layer = net->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float *input_data = input_layer->mutable_cpu_data(); 
	input_layer->Reshape(im.size(), im[0].channels(), im[0].rows, im[0].cols); // Reshape input_layer.
	net->Reshape();															   // Forward dimension change to all layers.

	// Preprocess the images
	for (unsigned int i = 0; i < im.size(); ++i)
	{
		im[i].convertTo(im[i], CV_32FC3);
		im[i] = im[i] - deep_mean_mat;
	}
	// Put the images to the input_data.
	std::vector<cv::Mat> input_channels;
	for (int i = 0; i < input_layer->channels() * input_layer->shape()[0]; ++i)
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}
	// Split each image and merge all together, then split to input_channels.
	std::vector<cv::Mat> im_split;
	for (unsigned int i = 0; i < im.size(); i++)
	{
		std::vector<cv::Mat> tmp_split;
		cv::split(im[i], tmp_split);
		im_split.insert(im_split.end(), tmp_split.begin(), tmp_split.end());
	}
	cv::Mat im_merge;
	cv::merge(im_split, im_merge);
	cv::split(im_merge, input_channels);

	/*
	debug("im size: %lu", im.size());
	imgInfo(im[0]);
	for (unsigned int i = 0; i < im.size(); i++)
	{
		cv::imshow("Tracking", im[i]);
		debug("i:%u", i);
		cv::waitKey(0);
	}

	debug("input_channels size: %lu", input_channels.size());
	for (unsigned int i = 0; i < input_channels.size(); i++)
	{
		cv::imshow("Tracking", input_channels[i]);
		debug("i:%u", i);
		cv::waitKey(0);
	}
*/
	net->Forward();

	ECO_FEATS feature_map;
	for (size_t idx = 0; idx < cnn_features.fparams.output_layer.size(); ++idx)
	{
		const float *pstart = NULL;
		vector<int> shape;
		if (cnn_features.fparams.output_layer[idx] == 3)
		{
			boost::shared_ptr<caffe::Blob<float>> layerData = net->blob_by_name("norm1");
			pstart = layerData->cpu_data();
			shape = layerData->shape();
		}
		else if (cnn_features.fparams.output_layer[idx] == 14)
		{
			boost::shared_ptr<caffe::Blob<float>> layerData = net->blob_by_name("conv5");
			//Blob<float>* layerData = net->output_blobs()[0]; // read "relu5" in the method above will cause error
			pstart = layerData->cpu_data();
			shape = layerData->shape();
		}

//		debug("shape: %d, %d, %d, %d", shape[0], shape[1], shape[2], shape[3]); //num, channel, height, width
//		debug("%d %d", cnn_features.fparams.start_ind[0 + 2 * idx] - 1, cnn_features.fparams.end_ind[0 + 2 * idx]);

		vector<cv::Mat> merge_feature;
		for (size_t i = 0; i < (size_t)(shape[0] * shape[1]); i++) //  CNN into single channel
		{
			cv::Mat feat_map(shape[2], shape[3], CV_32FC1, (void *)pstart);
			pstart += shape[2] * shape[3];
			//  extract features according to fparams
			CnnParameters fparams = cnn_features.fparams;
			cv::Mat extract_map = feat_map(cv::Range(fparams.start_ind[0 + 2 * idx] - 1, fparams.end_ind[0 + 2 * idx]),
										   cv::Range(fparams.start_ind[0 + 2 * idx] - 1, fparams.end_ind[0 + 2 * idx]));
			extract_map = (cnn_features.fparams.downsample_factor[idx] == 1) ? extract_map : sample_pool(extract_map, 2, 2);
			merge_feature.push_back(extract_map);
		} //end for

		feature_map.push_back(merge_feature);
	}

	return feature_map;
}

cv::Mat FeatureExtractor::sample_pool(const cv::Mat &im, int smaple_factor, int stride)
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

void FeatureExtractor::cnn_feature_normalization(ECO_FEATS &cnn_feat_maps)
{
	//debug("cnn_feat_maps: %lu", cnn_feat_maps.size());
	for (size_t i = 0; i < cnn_feat_maps.size(); i++) // for each layer = 2
	{
		vector<cv::Mat> temp = cnn_feat_maps[i];
		vector<float> sum_scales;
		//debug("temp: %lu", temp.size());
		for (size_t s = 0; s < temp.size(); s += cnn_features.fparams.nDim[i]) // for each scale , {96, 512} x scale
		{
			float sum = 0.0f;
			for (size_t j = s; j < s + cnn_features.fparams.nDim[i]; j++) // for all the dimension
				sum += cv::sum(temp[j].mul(temp[j]))[0];
			sum_scales.push_back(sum);
		}

		float para = 0.0f;
		if (i == 0)
			para = cnn_features.data_sz_block0.area() * cnn_features.fparams.nDim[i];
		else if (i == 1)
			para = cnn_features.data_sz_block1.area() * cnn_features.fparams.nDim[i];
		//debug("para: %f", para);
		for (unsigned int k = 0; k < temp.size(); k++)
			cnn_feat_maps[i][k] /= sqrt(sum_scales[k / cnn_features.fparams.nDim[i]] / para);
	}
}

vector<cv::Mat> FeatureExtractor::hog_feature_normalization(vector<cv::Mat> &hog_feat_maps)
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
		float para = hog_features.data_sz_block0.area() * hog_features.fparams.nDim;
		hog_feat_maps[i] /= sqrt(sum / para);

		cv::split(hog_feat_maps[i], result_vec);
		hog_maps_vec.insert(hog_maps_vec.end(), result_vec.begin(), result_vec.end());
	}

	return hog_maps_vec;
}
}