#include "eco_sample_update.h"


namespace eco_sample_update{

	void sample_update::init(const std::vector<cv::Size>& filter, const std::vector<int>& feature_dim)
	{
		//*** distance matrix initialization memory *****
		distance_matrix.create(cv::Size(nSamples, nSamples), CV_32FC2);
		gram_matrix.create(cv::Size(nSamples, nSamples), CV_32FC2);

		for (size_t i = 0; i < (size_t)distance_matrix.rows; i++)
		{
			for (size_t j = 0; j < (size_t)distance_matrix.cols; j++)
			{
				distance_matrix.at<cv::Vec<float, 2 >>(i, j) = cv::Vec<float, 2>(INF, 0);
				gram_matrix.at<cv::Vec<float, 2 >>(i, j) = cv::Vec<float, 2>(INF, 0);
			}
		}

		//*** samples memory initialization *******
		for (size_t n = 0; n < (size_t)nSamples; n++)
		{
			ECO_FEATS temp;
			for (size_t feat_block = 0; feat_block < (size_t)feature_dim.size(); feat_block++)
			{
				std::vector<cv::Mat> single_feat;
				for (size_t i = 0; i < (size_t)feature_dim[feat_block]; i++)
					single_feat.push_back(cv::Mat::zeros(cv::Size((filter[feat_block].width + 1) / 2, filter[feat_block].width), CV_32FC2));
				temp.push_back(single_feat);
			}
			samples_f.push_back(temp);
		}

		prior_weights.resize(nSamples);
	}

	void  sample_update::update_sample_sapce_model(ECO_FEATS& new_train_sample)
	{
		if (num_training_samples == 1)
		{
			//int i = 0;
		};
		//int unm_feature_blocks = new_train_sample.size();

		//*** Find the inner product of the new sample with existing samples ***
		cv::Mat gram_vector = find_gram_vector(new_train_sample);

		float new_train_sample_norm = 2 * FeatEnergy(new_train_sample);
		cv::Mat dist_vec(nSamples, 1, CV_32FC2);
		for (int i = 0; i < nSamples; i++)
		{
			float temp = new_train_sample_norm + gram_matrix.at<cv::Vec<float, 2>>(i, i)[0] - 2 * gram_vector.at<cv::Vec<float, 2>>(i, 0)[0];
			if (i < num_training_samples)
				dist_vec.at<cv::Vec<float, 2>>(i, 0) = cv::Vec<float, 2>(std::max(temp, 0.0f), 0);
			else
				dist_vec.at<cv::Vec<float, 2>>(i, 0) = cv::Vec<float, 2>(INF, 0);
		}

		std::vector<int> merged_sample;
		
		if (num_training_samples == nSamples) //*** if memory is full   ****
		{
			float min_sample_weight = INF;
			int min_sample_id = 0;
			findMin(min_sample_weight, min_sample_id);

			if (min_sample_weight < minmum_sample_weight)//*** If any prior weight is less than the minimum allowed weight, replace that sample with the new sample
			{
				//*** Normalise the prior weights so that the new sample gets weight as
				update_distance_matrix(gram_vector, new_train_sample_norm, min_sample_id, -1, 0, 1);
				prior_weights[min_sample_id] = 0;
				float sum = accumulate(prior_weights.begin(), prior_weights.end(), 0);
				for (size_t i = 0; i < (size_t)nSamples; i++)
				{
					prior_weights[i] = prior_weights[i] * (1 - learning_rate) / sum;
				}

				prior_weights[min_sample_id] = learning_rate;

				//*** Set the new sample and new sample position in the samplesf****
				new_sample_id = min_sample_id;
				new_sample = new_train_sample;
			}
			else
			{
				//*** If no sample has low enough prior weight, then we either merge
				//*** the new sample with an existing sample, or merge two of the
				//*** existing samples and insert the new sample in the vacated position
				double new_sample_min_dist;
				cv::Point min_sample_id;
				cv::minMaxLoc(real(dist_vec), &new_sample_min_dist, 0, &min_sample_id);

				//*** Find the closest pair amongst existing samples
				cv::Mat duplicate = distance_matrix.clone();
				double existing_samples_min_dist;
				cv::Point closest_exist_sample_pair;   //*** clost location ***
				cv::minMaxLoc(real(duplicate), &existing_samples_min_dist, 0, &closest_exist_sample_pair);

				if (closest_exist_sample_pair.x == closest_exist_sample_pair.y)
					assert("distance matrix diagonal filled wrongly ");

				if (new_sample_min_dist < existing_samples_min_dist)
				{
					//*** If the min distance of the new sample to the existing samples is less than the min distance
					//*** amongst any of the existing samples, we merge the new sample with the nearest existing
					for (size_t i = 0; i < prior_weights[i]; i++)
						prior_weights[i] *= (1 - learning_rate);

					//*** Set the position of the merged sample
					merged_sample_id = min_sample_id.y;

					//*** Extract the existing sample to merge ***
					ECO_FEATS existing_sample_to_merge = samples_f[merged_sample_id];

					//*** Merge the new_train_sample with existing sample ***
					ECO_FEATS merged_sample = merge_samples(existing_sample_to_merge, new_train_sample,
						prior_weights[merged_sample_id], learning_rate, std::string("merge"));

					//*** Update distance matrix and the gram matrix
					update_distance_matrix(gram_vector, new_train_sample_norm, merged_sample_id, -1,
						prior_weights[merged_sample_id], learning_rate);

					//*** Update the prior weight of the merged sample ***
					prior_weights[min_sample_id.y] += learning_rate;

					//*** discard new sample **********
				}
				else
				{
					//*** If the min distance amongst any of the existing  samples is less than the min distance of 
					//*** the new sample to the existing samples, we merge the nearest existing samples and insert the new 
					//*** sample in the vacated position

					//*** renormalize prior weights ***
					for (size_t i = 0; i < prior_weights[i]; i++)
						prior_weights[i] *= (1 - learning_rate);

					//*** Ensure that the sample with higher prior weight is assigned id1.
					if (prior_weights[closest_exist_sample_pair.x] > prior_weights[closest_exist_sample_pair.y])
						std::swap(closest_exist_sample_pair.x, closest_exist_sample_pair.y);

					//*** Merge the existing closest samples ****
					ECO_FEATS merged_sample = merge_samples(samples_f[closest_exist_sample_pair.x], samples_f[closest_exist_sample_pair.y],
						prior_weights[closest_exist_sample_pair.x], prior_weights[closest_exist_sample_pair.y], std::string("Merge"));

					//**  Update distance matrix and the gram matrix
					update_distance_matrix(gram_vector, new_train_sample_norm, closest_exist_sample_pair.x, closest_exist_sample_pair.y,
						prior_weights[closest_exist_sample_pair.x], prior_weights[closest_exist_sample_pair.y]);

					//*** Update prior weights for the merged sample and the new sample **
					prior_weights[closest_exist_sample_pair.x] += prior_weights[closest_exist_sample_pair.y];
					prior_weights[closest_exist_sample_pair.y] = learning_rate;

					//** Set the merged sample position and new sample position **
					merged_sample_id = closest_exist_sample_pair.x;
					new_sample_id = closest_exist_sample_pair.y;

					new_sample = new_train_sample;
				}

			}
		}     //**** end if memory is full *******
		else  //*** if memory is not full ***
		{
			size_t sample_position = num_training_samples;  //*** location ****
			update_distance_matrix(gram_vector, new_train_sample_norm, sample_position, -1, 0, 1);

			if (sample_position == 0)
				prior_weights[sample_position] = 1;
			else
			{
				for (size_t i = 0; i < prior_weights[i]; i++)
					prior_weights[i] *= (1 - learning_rate);
				prior_weights[sample_position] = learning_rate;
			}

			new_sample_id = sample_position;
			new_sample = new_train_sample;

			num_training_samples++;
		}

	}


	cv::Mat sample_update::find_gram_vector(ECO_FEATS& new_train_sample)
	{
		cv::Mat result(cv::Size(1, nSamples), CV_32FC2);
		for (size_t i = 0; i < (size_t)result.rows; i++)
			result.at<cv::Vec<float, 2> >(i, 0) = cv::Vec<float, 2>(INF, 0);

		std::vector<float> dist_vec;
		for (size_t i = 0; i < (size_t)num_training_samples; i++)
		{
			dist_vec.push_back(2 * feat_dis_compute(samples_f[i], new_train_sample));
		}

		for (size_t i = 0; i < dist_vec.size(); i++)
			result.at<cv::Vec<float, 2> >(i, 0) = cv::Vec<float, 2>(dist_vec[i], 0);
		return result;
	}

	float sample_update::feat_dis_compute(std::vector<std::vector<cv::Mat> >& feat1, std::vector<std::vector<cv::Mat> >& feat2)
	{
		if (feat1.size() != feat2.size())
			return 0;

		float dist = 0;
		for (size_t i = 0; i < feat1.size(); i++)
		{
			for (size_t j = 0; j < feat1[i].size(); j++)
			{
				cv::Mat feat2_conj = FFTTools::mat_conj(feat2[i][j]);
				dist += FFTTools::mat_sum(FFTTools::real(FFTTools::complexMultiplication(feat1[i][j], feat2_conj)));
			}
		}
		return dist;
	}

	void sample_update::update_distance_matrix(cv::Mat& gram_vector, float new_sample_norm, int id1, int id2, float w1, float w2)
	{
		float alpha1 = w1 / (w1 + w2);
		float alpha2 = 1 - alpha1;

		if (id2 < 0)
		{
			COMPLEX norm_id1 = gram_matrix.at<COMPLEX>(id1, id1);

			//** update the matrix ***
			if (alpha1 == 0)
			{
				gram_vector.col(0).copyTo(gram_matrix.col(id1));
				cv::Mat tt = gram_vector.t();
				tt.row(0).copyTo(gram_matrix.row(id1));
				gram_matrix.at<COMPLEX>(id1, id1) = COMPLEX(new_sample_norm, 0);
			}
			else if (alpha2 == 0)
			{
				// *** do nothing discard new sample *****
			}
			else
			{   // *** The new sample is merge with an existing sample
				cv::Mat t = alpha1 * gram_matrix.col(id1) + alpha2 * gram_vector.col(0), t_t;
				t.col(0).copyTo(gram_matrix.col(id1));
				t_t = t.t();
				t_t.row(0).copyTo(gram_matrix.row(id1));
				gram_matrix.at<COMPLEX>(id1, id1) =
					COMPLEX(pow(alpha1, 2) * norm_id1[0] + pow(alpha2, 2) * new_sample_norm + 2 * alpha1 * alpha2 * gram_vector.at<COMPLEX>(id1)[0], 0);
			}
			//*** Update distance matrix *****
			cv::Mat dist_vec(nSamples, 1, CV_32FC2);
			for (int i = 0; i < nSamples; i++)
			{
				float temp = gram_matrix.at<COMPLEX>(id1, id1)[0] + gram_matrix.at<COMPLEX>(i, i)[0] - 2 * gram_matrix.at<COMPLEX>(i, id1)[0];
				dist_vec.at<COMPLEX>(i, 0) = COMPLEX(std::max(temp, 0.0f), 0);
			}
			dist_vec.col(0).copyTo(distance_matrix.col(id1));
			cv::Mat tt = dist_vec.t();
			tt.row(0).copyTo(distance_matrix.row(id1));
			distance_matrix.at<COMPLEX>(id1, id1) = COMPLEX(INF, 0);
		}
		else
		{
			if (alpha1 == 0 || alpha2 == 0)
				assert("wrong");

			//*** Two existing samples are merged and the new sample fills the empty **
			COMPLEX norm_id1 = gram_matrix.at<COMPLEX>(id1, id1);
			COMPLEX norm_id2 = gram_matrix.at<COMPLEX>(id2, id2);
			COMPLEX ip_id1_id2 = gram_matrix.at<COMPLEX>(id1, id2);

			//*** Handle the merge of existing samples **
			cv::Mat t = alpha1 * gram_matrix.col(id1) + alpha2 * gram_matrix.col(id2), t_t;
			t.col(0).copyTo(gram_matrix.col(id1));
			cv::Mat tt = t.t();
			tt.row(0).copyTo(gram_matrix.row(id1));
			gram_matrix.at<COMPLEX>(id1, id1) =
				COMPLEX(pow(alpha1, 2) * norm_id1[0] + pow(alpha2, 2) * norm_id2[0] + 2 * alpha1 * alpha2 *ip_id1_id2[0], 0);
			gram_vector.at<COMPLEX>(id1) =
				COMPLEX(alpha1 * gram_vector.at<COMPLEX>(id1, 0)[0] + alpha2 * gram_vector.at<COMPLEX>(id2, 0)[0], 0);

			//*** Handle the new sample ****
			gram_vector.col(0).copyTo(gram_matrix.col(id2));
			tt = gram_vector.t();
			tt.row(0).copyTo(gram_matrix.row(id2));
			gram_matrix.at<COMPLEX>(id2, id2) = new_sample_norm;

			//*** Update the distance matrix ****
			cv::Mat dist_vec(nSamples, 1, CV_32FC2);
			std::vector<int> id({ id1, id2 });
			for (size_t i = 0; i < 2; i++)
			{
				for (int j = 0; j < nSamples; j++)
				{
					float temp = gram_matrix.at<COMPLEX>(id[i], id[i])[0] + gram_matrix.at<COMPLEX>(j, j)[0] - 2 * gram_matrix.at<COMPLEX>(j, id[i])[0];
					dist_vec.at<COMPLEX>(j, 0) = COMPLEX(std::max(temp, 0.0f), 0);
				}
				dist_vec.col(0).copyTo(distance_matrix.col(id[i]));
				cv::Mat tt = dist_vec.t();
				tt.row(0).copyTo(distance_matrix.row(id[i]));
				distance_matrix.at<COMPLEX>(id[i], id[i]) = COMPLEX(INF, 0);
			}
		}//if end

	}//function end

 	void sample_update::findMin(float& min_w, size_t index)const
	{
		std::vector<float>::const_iterator pos = std::min_element(prior_weights.begin(), prior_weights.end());
		min_w = *pos;
		index = pos - prior_weights.begin();
		/*min_w = INF;
		index = 0;

		for (size_t i = 0; i < prior_weights.size(); i++)
		{
			index = prior_weights[i] < min_w ? i : index;
			min_w = prior_weights[i] < min_w ? prior_weights[i] : min_w;
		}*/
	}

	sample_update::ECO_FEATS sample_update::merge_samples(ECO_FEATS& sample1, ECO_FEATS& sample2, float w1, float w2, std::string sample_merge_type)
	{
		float alpha1 = w1 / (w1 + w2);
		float alpha2 = 1 - alpha1;

		if (sample_merge_type == std::string("replace"))
			return sample1;
		else if (sample_merge_type == std::string("merge"))
		{
			ECO_FEATS merged_sample = sample1;
			for (size_t i = 0; i < sample1.size(); i++)
				for (size_t j = 0; j < sample1[i].size(); j++)
					merged_sample[i][j] = alpha1 * sample1[i][j] + alpha2 * sample2[i][j];
			return merged_sample;
		}
		else
			assert("Invalid sample merge type");

	}

	void sample_update::replace_sample(ECO_FEATS& new_sample, size_t idx)
	{
		samples_f[idx] = new_sample;
	}


	 void sample_update::set_gram_matrix(int r, int c, float val)
	{
		gram_matrix.at<float>(r, c) = val;
	}

}
