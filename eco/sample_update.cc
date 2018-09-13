#include "sample_update.hpp"

namespace eco
{
void SampleUpdate::init(const std::vector<cv::Size> &filter,
						const std::vector<int> &feature_dim,
						const size_t nSamples,
						const float learning_rate)
{
	distance_matrix_.release();
	gram_matrix_.release();
	nSamples_ = nSamples;
	learning_rate_ = learning_rate;
	sample_weight_.clear();
	samples_f_.clear();
	num_training_samples_ = 0;
	prior_weights_.clear();
	new_sample_.clear();
	merged_sample_.clear();
	new_sample_id_ = -1;
	merged_sample_id_ = -1;

	// distance matrix initialization
	distance_matrix_.create(cv::Size(nSamples_, nSamples_), CV_32FC2);
	gram_matrix_.create(cv::Size(nSamples_, nSamples_), CV_32FC2);

	// initialization to INF
	for (size_t i = 0; i < (size_t)distance_matrix_.rows; i++)
	{
		for (size_t j = 0; j < (size_t)distance_matrix_.cols; j++)
		{
			distance_matrix_.at<cv::Vec<float, 2>>(i, j) = cv::Vec<float, 2>(INF, 0);
			gram_matrix_.at<cv::Vec<float, 2>>(i, j) = cv::Vec<float, 2>(INF, 0);
		}
	}
	// Samples memory initialization
	samples_f_.clear();

	for (size_t n = 0; n < nSamples_; n++)
	{
		ECO_FEATS temp;
		for (size_t j = 0; j < (size_t)feature_dim.size(); j++) // for each feature
		{
			std::vector<cv::Mat> temp_single_feat;
			for (size_t i = 0; i < (size_t)feature_dim[j]; i++) // for each dimension of the feature
				temp_single_feat.push_back(cv::Mat::zeros(
					cv::Size((filter[j].height + 1) / 2, filter[j].width),
					CV_32FC2));
			temp.push_back(temp_single_feat);
		}
		samples_f_.push_back(temp);
	}

	// resize prior weights to the same as nSamples_

	prior_weights_.resize(nSamples_);

	// Show debug.
	for (size_t j = 0; j < (size_t)feature_dim.size(); j++)
	{
		debug("samples: %lu, feature %lu, size: %lu, mat: %d x %d",
			  nSamples_, j, samples_f_[nSamples_ - 1][j].size(),
			  samples_f_[nSamples_ - 1][j][feature_dim[j] - 1].rows,
			  samples_f_[nSamples_ - 1][j][feature_dim[j] - 1].cols);
	}
	/*
	debug("prior_weights_ size: %lu ", prior_weights_.size());
	for (size_t j = 0; j < (size_t)prior_weights_.size(); j++)
	{
		printf("%f ", prior_weights_[j]);
	}
	printf("\n");
	*/
}

void SampleUpdate::update_sample_space_model(const ECO_FEATS &new_train_sample)
{
	// Calculate the distance
	cv::Mat gram_vector = find_gram_vector(new_train_sample);				  // 32FC2 50 x 1, 2(ac+bd)
	float new_train_sample_norm = 2 * FeatureComputeEnergy(new_train_sample); //2(c^2+d^2)
	cv::Mat distance(nSamples_, 1, CV_32FC2);
	for (size_t i = 0; i < nSamples_; i++)
	{
		// a^2 + b^2 + c^2 + d^2 - 2(ac+bd)
		float temp = new_train_sample_norm + gram_matrix_.at<cv::Vec<float, 2>>(i, i)[0] - 2 * gram_vector.at<cv::Vec<float, 2>>(i, 0)[0];
		if (i < (size_t)num_training_samples_)
			distance.at<cv::Vec<float, 2>>(i, 0) =
				cv::Vec<float, 2>(std::max(temp, 0.0f), 0);
		else
			distance.at<cv::Vec<float, 2>>(i, 0) = cv::Vec<float, 2>(INF, 0);
	}
	// End of calcualte the distance

	if (num_training_samples_ == nSamples_) // if memory is full
	{
		float min_sample_weight_ = INF;
		size_t min_sample_id = 0;
		findMin(min_sample_weight_, min_sample_id);
		//	debug("min_sample: %d %f", min_sample_id, min_sample_weight_);

		if (min_sample_weight_ < minmum_sample_weight_)
		// If any prior weight is less than the minimum allowed weight,
		// replace that sample with the new sample
		{
			update_distance_matrix(gram_vector, new_train_sample_norm, min_sample_id, -1, 0, 1);
			prior_weights_[min_sample_id] = 0;
			// normalize the prior_weights_
			float sum = std::accumulate(prior_weights_.begin(), prior_weights_.end(), 0.0f);
			for (size_t i = 0; i < (size_t)nSamples_; i++)
			{
				prior_weights_[i] = prior_weights_[i] *
									(1 - learning_rate_) / sum;
			}
			// set the new sample's weight as learning_rate_
			prior_weights_[min_sample_id] = learning_rate_;

			// update sampel space.
			merged_sample_id_ = -1;
			new_sample_id_ = min_sample_id;
			new_sample_ = new_train_sample;
			replace_sample(new_sample_, new_sample_id_);
		}
		else // If no sample has low enough prior weight, then we either merge
			 // the new sample with an existing sample, or merge two of the
			 // existing samples and insert the new sample in the vacated position
		{
			// Find the minimum distance between new sample and exsiting samples.
			double new_sample_min_dist;
			cv::Point min_sample_id;
			cv::minMaxLoc(real(distance), &new_sample_min_dist, 0, &min_sample_id);

			// Find the closest pair amongst existing samples.
			double existing_samples_min_dist;
			cv::Point closest_exist_sample_pair;
			cv::Mat duplicate = distance_matrix_.clone();
			cv::minMaxLoc(real(duplicate), &existing_samples_min_dist, 0, &closest_exist_sample_pair);

			if (closest_exist_sample_pair.x == closest_exist_sample_pair.y)
				assert(0 && "error: distance matrix diagonal filled wrongly.");

			if (new_sample_min_dist < existing_samples_min_dist)
			{
				// If the min distance of the new sample to the existing samples is less than the min distance
				// amongst any of the existing samples, we merge the new sample with the nearest existing

				// renormalize prior weights
				for (size_t i = 0; i < prior_weights_[i]; i++)
					prior_weights_[i] *= (1 - learning_rate_);

				// Set the position of the merged sample
				merged_sample_id_ = min_sample_id.y;

				// Extract the existing sample to merge
				ECO_FEATS existing_sample_to_merge = samples_f_[merged_sample_id_];

				// Merge the new_train_sample with existing sample
				merged_sample_ =
					merge_samples(existing_sample_to_merge,
								  new_train_sample,
								  prior_weights_[merged_sample_id_], learning_rate_,
								  std::string("merge"));

				// Update distance matrix and the gram matrix
				update_distance_matrix(gram_vector, new_train_sample_norm, merged_sample_id_, -1, prior_weights_[merged_sample_id_], learning_rate_);

				// Update the prior weight of the merged sample
				prior_weights_[min_sample_id.y] += learning_rate_;

				// update the merged sample and discard new sample
				replace_sample(merged_sample_, merged_sample_id_);
			}
			else
			{
				// we merge the nearest existing samples and insert the new sample in the vacated position

				// renormalize prior weights
				for (size_t i = 0; i < prior_weights_[i]; i++)
					prior_weights_[i] *= (1 - learning_rate_);

				// Ensure that the sample with higher prior weight is assigned id1.
				if (prior_weights_[closest_exist_sample_pair.x] >
					prior_weights_[closest_exist_sample_pair.y])
					std::swap(closest_exist_sample_pair.x,
							  closest_exist_sample_pair.y);

				// Merge the existing closest samples
				merged_sample_ =
					merge_samples(samples_f_[closest_exist_sample_pair.x],
								  samples_f_[closest_exist_sample_pair.y],
								  prior_weights_[closest_exist_sample_pair.x], prior_weights_[closest_exist_sample_pair.y],
								  std::string("merge"));

				// Update distance matrix and the gram matrix
				update_distance_matrix(gram_vector, new_train_sample_norm, closest_exist_sample_pair.x, closest_exist_sample_pair.y, prior_weights_[closest_exist_sample_pair.x], prior_weights_[closest_exist_sample_pair.y]);

				// Update prior weights for the merged sample and the new sample
				prior_weights_[closest_exist_sample_pair.x] +=
					prior_weights_[closest_exist_sample_pair.y];
				prior_weights_[closest_exist_sample_pair.y] = learning_rate_;

				// Update the merged sample and insert new sample
				merged_sample_id_ = closest_exist_sample_pair.x;
				new_sample_id_ = closest_exist_sample_pair.y;
				new_sample_ = new_train_sample;
				replace_sample(merged_sample_, merged_sample_id_);
				replace_sample(new_sample_, new_sample_id_);
			}
		}
	}	// end if memory is full
	else // if memory is not full
	{
		size_t sample_position = num_training_samples_;
		update_distance_matrix(gram_vector, new_train_sample_norm, sample_position, -1, 0, 1);

		if (sample_position == 0)
		{
			prior_weights_[sample_position] = 1;
		}
		else
		{
			for (size_t i = 0; i < sample_position; i++)
				prior_weights_[i] *= (1 - learning_rate_);
			prior_weights_[sample_position] = learning_rate_;
		}
		// update sample space
		new_sample_id_ = sample_position;
		new_sample_ = new_train_sample;
		replace_sample(new_sample_, new_sample_id_);

		num_training_samples_++;
	}
	//debug("num_training_samples_: %lu", num_training_samples_);
}

void SampleUpdate::update_distance_matrix(cv::Mat &gram_vector, float new_sample_norm, int id1, int id2, float w1, float w2)
{
	float alpha1 = w1 / (w1 + w2);
	float alpha2 = 1 - alpha1;
	//debug("alpha1: %f, alpha2: %f", alpha1, alpha2);
	if (id2 < 0) //
	{
		COMPLEX norm_id1 = gram_matrix_.at<COMPLEX>(id1, id1);

		// update the matrix
		if (alpha1 == 0)
		{
			gram_vector.col(0).copyTo(gram_matrix_.col(id1));
			cv::Mat tt = gram_vector.t();
			tt.row(0).copyTo(gram_matrix_.row(id1));
			gram_matrix_.at<COMPLEX>(id1, id1) = COMPLEX(new_sample_norm, 0);
		}
		else if (alpha2 == 0)
		{
			//  do nothing discard new sample
		}
		else
		{ // The new sample is merge with an existing sample
			cv::Mat t = alpha1 * gram_matrix_.col(id1) + alpha2 * gram_vector.col(0), t_t;
			t.col(0).copyTo(gram_matrix_.col(id1));
			t_t = t.t();
			t_t.row(0).copyTo(gram_matrix_.row(id1));
			gram_matrix_.at<COMPLEX>(id1, id1) =
				COMPLEX(std::pow(alpha1, 2) * norm_id1[0] + std::pow(alpha2, 2) * new_sample_norm + 2 * alpha1 * alpha2 * gram_vector.at<COMPLEX>(id1)[0], 0);
		}

		// Update distance matrix
		cv::Mat distance(nSamples_, 1, CV_32FC2);
		for (size_t i = 0; i < nSamples_; i++)
		{
			float temp = gram_matrix_.at<COMPLEX>(id1, id1)[0] +
						 gram_matrix_.at<COMPLEX>(i, i)[0] -
						 2 * gram_matrix_.at<COMPLEX>(i, id1)[0];
			distance.at<COMPLEX>(i, 0) = COMPLEX(std::max(temp, 0.0f), 0);
		}
		distance.col(0).copyTo(distance_matrix_.col(id1));
		cv::Mat tt = distance.t();
		tt.row(0).copyTo(distance_matrix_.row(id1));
		distance_matrix_.at<COMPLEX>(id1, id1) = COMPLEX(INF, 0);
	}
	else
	{
		if (alpha1 == 0 || alpha2 == 0)
		{
			assert(0 && "error: alpha1 or alpha2 equals 0");
		}
		// Two existing samples are merged and the new sample fills the empty
		COMPLEX norm_id1 = gram_matrix_.at<COMPLEX>(id1, id1);
		COMPLEX norm_id2 = gram_matrix_.at<COMPLEX>(id2, id2);
		COMPLEX ip_id1_id2 = gram_matrix_.at<COMPLEX>(id1, id2);
		//debug("%d %d, %f %f, %f %f %f", id1, id2, w1, w2, norm_id1[0], norm_id2[0], ip_id1_id2[0]);
		// Handle the merge of existing samples
		cv::Mat t = alpha1 * gram_matrix_.col(id1) +
					alpha2 * gram_matrix_.col(id2),
				t_t;

		t.col(0).copyTo(gram_matrix_.col(id1));

		cv::Mat tt = t.t();
		tt.row(0).copyTo(gram_matrix_.row(id1));

		gram_matrix_.at<COMPLEX>(id1, id1) =
			COMPLEX(std::pow(alpha1, 2) * norm_id1[0] +
						std::pow(alpha2, 2) * norm_id2[0] +
						2 * alpha1 * alpha2 * ip_id1_id2[0],
					0);
		gram_vector.at<COMPLEX>(id1) =
			COMPLEX(alpha1 * gram_vector.at<COMPLEX>(id1, 0)[0] +
						alpha2 * gram_vector.at<COMPLEX>(id2, 0)[0],
					0);

		// Handle the new sample
		gram_vector.col(0).copyTo(gram_matrix_.col(id2));
		tt = gram_vector.t();
		tt.row(0).copyTo(gram_matrix_.row(id2));
		gram_matrix_.at<COMPLEX>(id2, id2) = new_sample_norm;

		// Update the distance matrix
		cv::Mat distance(nSamples_, 1, CV_32FC2);
		std::vector<int> id({id1, id2});
		for (size_t i = 0; i < 2; i++)
		{
			for (size_t j = 0; j < nSamples_; j++)
			{
				float temp = gram_matrix_.at<COMPLEX>(id[i], id[i])[0] +
							 gram_matrix_.at<COMPLEX>(j, j)[0] -
							 2 * gram_matrix_.at<COMPLEX>(j, id[i])[0];
				distance.at<COMPLEX>(j, 0) = COMPLEX(std::max(temp, 0.0f), 0);
			}
			distance.col(0).copyTo(distance_matrix_.col(id[i]));
			cv::Mat tt = distance.t();
			tt.row(0).copyTo(distance_matrix_.row(id[i]));
			distance_matrix_.at<COMPLEX>(id[i], id[i]) = COMPLEX(INF, 0);
		}
	} //if end
} //function end
} // namespace eco
