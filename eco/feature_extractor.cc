#include "feature_extractor.hpp"
namespace eco
{
ECO_FEATS FeatureExtractor::extractor(const cv::Mat image,
									  const cv::Point2f pos,
									  const vector<float> scales,
									  const EcoParameters &params)
{
	int num_features = 0, num_scales = scales.size();
#ifdef USE_CAFFE
	cv::Mat new_deep_mean_mat;
	if (params.useDeepFeature)
	{
		cnn_feat_ind_ = num_features;
		num_features++;
		cnn_features_ = params.cnn_features;
		this->net_ = params.cnn_features.fparams.net;
		resize(params.cnn_features.fparams.deep_mean_mat, new_deep_mean_mat,
			   params.cnn_features.img_input_sz, 0, 0, cv::INTER_CUBIC);
	}
#endif
	if (params.useHogFeature)
	{
		hog_feat_ind_ = num_features;
		num_features++;
		hog_features_ = params.hog_features;
	}
	if (params.useCnFeature)
	{
	}

	// Extract images for different feautures
	vector<vector<cv::Mat>> img_samples;
	for (int i = 0; i < num_features; ++i) // for each feature
	{
		vector<cv::Mat> img_samples_temp(num_scales);
		//debug("scales:%lu", scales.size());
		for (unsigned int j = 0; j < scales.size(); ++j) // for each scale
		{
			cv::Size2f img_sample_sz;
			cv::Size2f img_input_sz;
			if ((i == 0) && params.useDeepFeature)
			{
#ifdef USE_CAFFE
				img_sample_sz = cnn_features_.img_sample_sz;
				img_input_sz = cnn_features_.img_input_sz;
#endif
			}
			else
			{
				img_sample_sz = hog_features_.img_sample_sz;
				img_input_sz = hog_features_.img_input_sz;
			}
			img_sample_sz.width *= scales[j];
			img_sample_sz.height *= scales[j];
			img_samples_temp[j] = sample_patch(image,
											   pos,
											   img_sample_sz,
											   img_input_sz);
		}
		img_samples.push_back(img_samples_temp);
	}

	// Extract features
	ECO_FEATS sum_features;
#ifdef USE_CAFFE
	if (params.useDeepFeature)
	{
		cnn_feat_maps_ = get_cnn_layers(img_samples[cnn_feat_ind_],
										new_deep_mean_mat);
		cnn_feature_normalization(cnn_feat_maps_);
		sum_features = cnn_feat_maps_;
	}
#endif
	if (params.useHogFeature)
	{
		/* Debug
		debug("hog_feat_ind_:%d", hog_feat_ind_);
		debug("img_samples[hog_feat_ind_] size: %lu",
			  img_samples[hog_feat_ind_].size());
		imgInfo(img_samples[hog_feat_ind_][0]);
		showmat3ch(img_samples[hog_feat_ind_][0], 0); */
#ifdef USE_SIMD
		hog_feat_maps_ = get_hog_features_simd(img_samples[hog_feat_ind_]);
		//imgInfo(hog_feat_maps_[0]);]
		//debug();
		//showmatNch(hog_feat_maps_[0], 2);
		hog_feat_maps_ = hog_feature_normalization_simd(hog_feat_maps_);
		//imgInfo(hog_feat_maps_[0]);
		//debug();
		//showmatNch(hog_feat_maps_[0], 2);
#else
		// 0.006697 s
		hog_feat_maps_ = get_hog_features(img_samples[hog_feat_ind_]);
		// 32FCO 25 x 25
		// 0.000125 s
		hog_feat_maps_ = hog_feature_normalization(hog_feat_maps_);
		// 32FC1 25 x 25
#endif
		sum_features.push_back(hog_feat_maps_);
	}

	if (params.useCnFeature)
	{
	}

	return sum_features;
}

cv::Mat FeatureExtractor::sample_patch(const cv::Mat im,
									   const cv::Point2f posf,
									   cv::Size2f sample_sz,
									   cv::Size2f input_sz)
{
	// Pos should be integer when input, but floor in just in case.
	cv::Point2i pos(posf);
	//debug("%d, %d", pos.y, pos.x);

	// Downsample factor
	float resize_factor = std::min(sample_sz.width / input_sz.width,
								   sample_sz.height / input_sz.height);
	int df = std::max((float)floor(resize_factor - 0.1), float(1));
	//debug("resize_factor: %f, df: %d,sample_sz: %f x %f,input_sz: % f x % f",
	//	  resize_factor, df,
	//	  sample_sz.width, sample_sz.height,
	//	  input_sz.width, input_sz.height);

	cv::Mat new_im;
	im.copyTo(new_im);
	//debug("new_im:%d x %d", new_im.rows, new_im.cols);

	if (df > 1)
	{
		// compute offset and new center position
		cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
		pos.x = (pos.x - os.x - 1) / df + 1;
		pos.y = (pos.y - os.y - 1) / df + 1;
		// new sample size
		sample_sz.width = sample_sz.width / df;
		sample_sz.height = sample_sz.height / df;
		// down sample image
		int r = (im.rows - os.y) / df + 1;
		int c = (im.cols - os.x) / df;
		cv::Mat new_im2(r, c, im.type());
		new_im = new_im2;
		for (size_t i = 0 + os.y, m = 0;
			 i < (size_t)im.rows && m < (size_t)new_im.rows;
			 i += df, ++m)
			for (size_t j = 0 + os.x, n = 0;
				 j < (size_t)im.cols && n < (size_t)new_im.cols;
				 j += df, ++n)
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
	//debug("new_im:%d x %d, pos2:%d %d, sample_sz:%f x %f",
	//	  new_im.rows, new_im.cols, pos2.y, pos2.x,
	//	  sample_sz.height, sample_sz.width);

	cv::Mat im_patch = subwindow(new_im,
								 cv::Rect(pos2, sample_sz),
								 IPL_BORDER_REPLICATE);

	cv::Mat resized_patch;
	cv::resize(im_patch, resized_patch, input_sz);
	/* Debug
	imgInfo(resized_patch); // 8UC3 150 x 150
	showmat3ch(resized_patch, 0);
	// resized_patch(121,21,1) in matlab RGB, cv: BGR
	debug("%d", resized_patch.at<cv::Vec3b>(120,20)[2]); 
	assert(0); */
	return resized_patch;
}
#ifdef USE_SIMD
vector<cv::Mat> FeatureExtractor::get_hog_features_simd(const vector<cv::Mat> ims)
{
	if (ims.empty())
		return vector<cv::Mat>();
	//debug("ims.size: %lu", ims.size());
	vector<cv::Mat> hog_feats;
	for (unsigned int k = 0; k < ims.size(); k++)
	{
		int h, w, d, binSize, nOrients, softBin, nDim, hb, wb, useHog;
		bool full = 1;
		useHog = 2;
		h = ims[k].rows;
		w = ims[k].cols;
		d = ims[k].channels();
		binSize = 6;
		nOrients = 9;
		softBin = -1;
		nDim = useHog == 0 ? nOrients : 
						(useHog == 1 ? nOrients * 4 : nOrients * 3 + 5);
		hb = h / binSize;
		wb = w / binSize;
		float clipHog = 0.2f;

		float *I, *M, *O, *H, *Hb;
		debug();
  		I = (float*) alMalloc(h * w * d * sizeof(float),16);
  		M = (float*) alMalloc(h * w * sizeof(float),16);
  		O = (float*) alMalloc(h * w * sizeof(float),16);
  		H = (float*) alMalloc(hb * wb * nDim * sizeof(float),16);
  		Hb = (float*) alMalloc(hb * wb * (nDim - 1) * sizeof(float),16);
/*
		I = (float *)wrMalloc(h * w * d * sizeof(float));
		M = (float *)wrCalloc(h * w, sizeof(float));
		O = (float *)wrCalloc(h * w, sizeof(float));
		H = (float *)wrMalloc(hb * wb * nDim * sizeof(float));
		Hb = (float *)wrMalloc(hb * wb * (nDim - 1) * sizeof(float));
*/	
		/*
		std::vector<cv::Mat> channels;
		cv::split(ims_f, channels);
		// transpose because matlab is col-major
		cv::transpose(channels[0], channels[0]);
		cv::transpose(channels[1], channels[1]);
		cv::transpose(channels[2], channels[2]);
		memcpy(I, channels[2].ptr(), h * w * sizeof(float));
		memcpy(I + h * w, channels[1].ptr(), h * w * sizeof(float));
		memcpy(I + 2 * h * w, channels[0].ptr(), h * w * sizeof(float));
		*/
		debug();
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
			{
				*(I + i * w + j) = (float)ims[k].at<cv::Vec3b>(j, i)[2];
				*(I + h * w + i * w + j) = (float)ims[k].at<cv::Vec3b>(j, i)[1];
				*(I + 2 * h * w + i * w + j) = (float)ims[k].at<cv::Vec3b>(j, i)[0];
			}

		gradMag(I, M, O, h, w, d, 1);
		debug();
		/*
		for (int i = 0; i < h; i++)
		{
			//printf("%f ", M[h * 149 + i]);
			printf("%f ", M[i]);
		}
		printf("\nM end\n");
		for (int i = 0; i < h; i++)
		{
			//printf("%f ", M[h * 149 + i]);
			printf("%f ", M[150 + i]);
		}
		printf("\nM end\n");
		for (int i = 0; i < h; i++)
		{
			//printf("%f ", O[h * 149 + i]);
			printf("%f ", O[i]);
		}
		printf("\nO end\n");
		for (int i = 0; i < h; i++)
		{
			//printf("%f ", O[h * 149 + i]);
			printf("%f ", O[150 + i]);
		}
		printf("\nO end\n");
*/
		if (useHog == 0)
		{
			gradHist(M, O, H, h, w, binSize, nOrients, softBin, full);
		}
		else if (useHog == 1)
		{
			hog(M, O, H, h, w, binSize, nOrients, softBin, full, clipHog);
		}
		else
		{
			fhog(M, O, H, h, w, binSize, nOrients, softBin, clipHog);
		}
		debug();
		for (int i = 0; i < hb; i++)
		{
			printf("%f ", H[i]);
		}
		printf("\nH end\n");
		for (int i = 0; i < hb; i++)
		{
			printf("%f ", H[25 + i]);
		}
		printf("\nH end\n");
		for (int i = 0; i < hb; i++)
		{
			printf("%f ", H[25 * 25 * 30 + i]);
		}
		printf("\nH end\n");
		//assert(0);
		/* Debug
		for (int i = 0; i < std::floor(h / cell_size); i++)
		{
			//printf("%f ", H[25 * 25 * 30 + 25 * 24 + i]);
			printf("%f ", H[i]);
		}
		printf("\nH end\n");
		for (int i = 0; i < std::floor(h / cell_size); i++)
		{
			//printf("%f ", H[25 * 25 * 30 + 25 * 24 + i]);
			printf("%f ", H[25 + i]);
		}
		printf("\nH end\n");
	*/
/*
		// cv: ch->row->col; matlab: col->row->ch;
		for (int i = 0; i < hb; i++)
			for (int j = 0; j < wb; j++)
				for (int l = 0; l < nDim - 1; l++)
				{
					*(Hb + nDim * (i + j * wb) + l) = *(H + l * hb * wb + i * hb + j);
				}
				*/
		//cv::Mat featuresMap = cv::Mat(cv::Size(wb, hb), CV_32FC(nDim - 1)), Hb);
		//featuresMap = featuresMap.clone();
		cv::Mat featuresMap = cv::Mat(cv::Size(wb, hb), CV_32FC(nDim - 1));
		debug();
		for (int i = 0; i < hb; i++)
			for (int j = 0; j < wb; j++)
				for (int l = 0; l < nDim - 1; l++)
				{
					featuresMap.at<cv::Vec<float,31>>(i, j)[l] = *(H + l * hb * wb + i * hb + j);
				}
		//showmatNch(featuresMap, 2);
		hog_feats.push_back(featuresMap);
		debug();
		alFree(I); alFree(M); alFree(O); alFree(H); alFree(Hb);
		//wrFree(I); wrFree(M); wrFree(O); wrFree(H); wrFree(Hb);
	}
	debug();
	return hog_feats;
}

vector<cv::Mat> FeatureExtractor::hog_feature_normalization_simd(vector<cv::Mat> &hog_feat_maps)
{
	vector<cv::Mat> hog_maps_vec;
	for (size_t i = 0; i < hog_feat_maps.size(); i++)
	{
		cv::Mat temp = hog_feat_maps[i].mul(hog_feat_maps[i]);
		// float sum_scales = cv::sum(temp)[0]; // sum can not work when dimension exceeding 3
		vector<cv::Mat> temp_vec, result_vec;
		float sum = 0;
		cv::split(temp, temp_vec);
		for (int j = 0; j < temp.channels(); j++)
			sum += cv::sum(temp_vec[j])[0];
		float para = hog_features_.data_sz_block0.area() *
					 hog_features_.fparams.nDim;
		hog_feat_maps[i] *= sqrt(para / sum);
		//debug("para:%f, sum:%f, sqrt:%f", para, sum, sqrt(para / sum));
		cv::split(hog_feat_maps[i], result_vec);
		hog_maps_vec.insert(hog_maps_vec.end(),
							result_vec.begin(),
							result_vec.end());
	}
	return hog_maps_vec;
}
#endif

vector<cv::Mat> FeatureExtractor::get_hog_features(const vector<cv::Mat> ims)
{
	if (ims.empty())
		return vector<cv::Mat>();

	vector<cv::Mat> hog_feats;
	for (unsigned int i = 0; i < ims.size(); i++)
	{
		cv::Mat ims_f;
		ims[i].convertTo(ims_f, CV_32FC3);

		cv::Size _tmpl_sz;
		_tmpl_sz.width = ims_f.cols;
		_tmpl_sz.height = ims_f.rows;

		int _cell_size = hog_features_.fparams.cell_size;
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

		// Add extra cell filled with zeros around the image
		cv::Mat featurePaddingMat(_tmpl_sz.height + _cell_size * 2,
								  _tmpl_sz.width + _cell_size * 2,
								  CV_32FC3, cvScalar(0, 0, 0));

		if (ims_f.cols != _tmpl_sz.width || ims_f.rows != _tmpl_sz.height)
		{
			resize(ims_f, ims_f, _tmpl_sz);
		}
		ims_f.copyTo(featurePaddingMat);

		IplImage zz = featurePaddingMat;
		CvLSVMFeatureMapCaskade *map_temp;
		getFeatureMaps(&zz, _cell_size, &map_temp); // dimension: 27
		normalizeAndTruncate(map_temp, 0.2f);		// dimension: 108
		PCAFeatureMaps(map_temp);					// dimension: 31

		// Procedure do deal with cv::Mat multichannel bug(can not merge)
		cv::Mat featuresMap = cv::Mat(cv::Size(map_temp->sizeX, map_temp->sizeY), CV_32FC(map_temp->numFeatures), map_temp->map);

		// clone because map_temp will be free.
		featuresMap = featuresMap.clone();

		freeFeatureMapObject(&map_temp);

		hog_feats.push_back(featuresMap);
	}

	return hog_feats;
}

vector<cv::Mat> FeatureExtractor::hog_feature_normalization(vector<cv::Mat> &hog_feat_maps)
{
	vector<cv::Mat> hog_maps_vec;
	for (size_t i = 0; i < hog_feat_maps.size(); i++)
	{
		cv::Mat temp = hog_feat_maps[i];
		temp = temp.mul(temp);
		// float sum_scales = cv::sum(temp)[0]; // sum can not work when dimension exceeding 3
		vector<cv::Mat> temp_vec, result_vec;
		float sum = 0;
		cv::split(temp, temp_vec);
		for (int j = 0; j < temp.channels(); j++)
			sum += cv::sum(temp_vec[j].mul(temp_vec[j]))[0];
		float para = hog_features_.data_sz_block0.area() *
					 hog_features_.fparams.nDim;
		hog_feat_maps[i] /= sqrt(sum / para);

		cv::split(hog_feat_maps[i], result_vec);
		hog_maps_vec.insert(hog_maps_vec.end(),
							result_vec.begin(),
							result_vec.end());
	}

	return hog_maps_vec;
}
//=========================================================================
#ifdef USE_CAFFE
ECO_FEATS FeatureExtractor::get_cnn_layers(vector<cv::Mat> im, const cv::Mat &deep_mean_mat)
{
	caffe::Blob<float> *input_layer = net_->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float *input_data = input_layer->mutable_cpu_data();
	input_layer->Reshape(im.size(), im[0].channels(), im[0].rows, im[0].cols); // Reshape input_layer.
	net_->Reshape();														   // Forward dimension change to all layers.

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
	net_->Forward();

	ECO_FEATS feature_map;
	for (size_t idx = 0; idx < cnn_features_.fparams.output_layer.size(); ++idx)
	{
		const float *pstart = NULL;
		vector<int> shape;
		if (cnn_features_.fparams.output_layer[idx] == 3)
		{
			boost::shared_ptr<caffe::Blob<float>> layerData = net_->blob_by_name("norm1");
			pstart = layerData->cpu_data();
			shape = layerData->shape();
		}
		else if (cnn_features_.fparams.output_layer[idx] == 14)
		{
			boost::shared_ptr<caffe::Blob<float>> layerData = net_->blob_by_name("conv5");
			//Blob<float>* layerData = net_->output_blobs()[0]; // read "relu5" in the method above will cause error
			pstart = layerData->cpu_data();
			shape = layerData->shape();
		}

		//		debug("shape: %d, %d, %d, %d", shape[0], shape[1], shape[2], shape[3]); //num, channel, height, width
		//		debug("%d %d", cnn_features_.fparams.start_ind[0 + 2 * idx] - 1, cnn_features_.fparams.end_ind[0 + 2 * idx]);

		vector<cv::Mat> merge_feature;
		for (size_t i = 0; i < (size_t)(shape[0] * shape[1]); i++) //  CNN into single channel
		{
			cv::Mat feat_map(shape[2], shape[3], CV_32FC1, (void *)pstart);
			pstart += shape[2] * shape[3];
			//  extract features according to fparams
			CnnParameters fparams = cnn_features_.fparams;
			cv::Mat extract_map = feat_map(cv::Range(fparams.start_ind[0 + 2 * idx] - 1, fparams.end_ind[0 + 2 * idx]),
										   cv::Range(fparams.start_ind[0 + 2 * idx] - 1, fparams.end_ind[0 + 2 * idx]));
			extract_map = (cnn_features_.fparams.downsample_factor[idx] == 1) ? extract_map : sample_pool(extract_map, 2, 2);
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
		for (size_t s = 0; s < temp.size(); s += cnn_features_.fparams.nDim[i]) // for each scale , {96, 512} x scale
		{
			float sum = 0.0f;
			for (size_t j = s; j < s + cnn_features_.fparams.nDim[i]; j++) // for all the dimension
				sum += cv::sum(temp[j].mul(temp[j]))[0];
			sum_scales.push_back(sum);
		}

		float para = 0.0f;
		if (i == 0)
			para = cnn_features_.data_sz_block0.area() * cnn_features_.fparams.nDim[i];
		else if (i == 1)
			para = cnn_features_.data_sz_block1.area() * cnn_features_.fparams.nDim[i];
		//debug("para: %f", para);
		for (unsigned int k = 0; k < temp.size(); k++)
			cnn_feat_maps[i][k] /= sqrt(sum_scales[k / cnn_features_.fparams.nDim[i]] / para);
	}
}
#endif
} // namespace eco