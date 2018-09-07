#ifndef SAMPLE_UPDATE_HPP
#define SAMPLE_UPDATE_HPP

#include <numeric>
#include <vector>

#include <opencv2/opencv.hpp>

#include "parameters.hpp"
#include "ffttools.hpp"
#include "feature_operator.hpp"
#include "debug.hpp"

namespace eco
{

class SampleUpdate
{
  public:
	SampleUpdate(){};
	virtual ~SampleUpdate(){};

	void init(const std::vector<cv::Size> &filter,
			  const std::vector<int> &feature_dim,
			  const size_t nSamples,
			  const float learning_rate);

	void update_sample_space_model(const ECO_FEATS &new_train_sample); 

	void update_distance_matrix(cv::Mat &gram_vector, float new_sample_norm,
								int id1, int id2, float w1, float w2);

	inline cv::Mat find_gram_vector(const ECO_FEATS &new_train_sample) 
	{
		cv::Mat result(cv::Size(1, nSamples_), CV_32FC2);
		for (size_t i = 0; i < (size_t)result.rows; i++) // init to INF;
			result.at<cv::Vec<float, 2>>(i, 0) = cv::Vec<float, 2>(INF, 0);

		std::vector<float> distance_vector;
		for (size_t i = 0; i < num_training_samples_; i++) // calculate the distance;
			distance_vector.push_back(2 * 
			FeatureComputeInnerProduct(samples_f_[i], new_train_sample));

		for (size_t i = 0; i < distance_vector.size(); i++)
			result.at<cv::Vec<float, 2>>(i, 0) = 
				cv::Vec<float, 2>(distance_vector[i], 0);

		return result;
	};
	// find the minimum element in prior_weights_;
	inline void findMin(float &min_w, size_t &index) const
	{
		std::vector<float>::const_iterator pos = std::min_element(prior_weights_.begin(), prior_weights_.end());
		min_w = *pos;
		index = pos - prior_weights_.begin();
	};

	inline ECO_FEATS merge_samples(const ECO_FEATS &sample1,
								   const ECO_FEATS &sample2,
								   const float w1, const float w2,
								   const std::string sample_merge_type = "merge")
	{
		float alpha1 = w1 / (w1 + w2);
		float alpha2 = 1 - alpha1;
		ECO_FEATS merged_sample = sample1;

		if (sample_merge_type == std::string("replace"))
		{
		}
		else if (sample_merge_type == std::string("merge"))
		{
			for (size_t i = 0; i < sample1.size(); i++)
				for (size_t j = 0; j < sample1[i].size(); j++)
					merged_sample[i][j] = alpha1 * sample1[i][j] + alpha2 * sample2[i][j];
		}
		return merged_sample;
	};

	inline void replace_sample(const ECO_FEATS &new_sample, const size_t idx)
	{
		samples_f_[idx] = new_sample;
	};

	inline void set_gram_matrix(const int r, const int c, const float val)
	{
		gram_matrix_.at<float>(r, c) = val;
	};

	int get_merged_sample_id() const { return merged_sample_id_; }

	int get_new_sample_id() const { return new_sample_id_; }

	std::vector<float> get_prior_weights() const { return prior_weights_; }

	std::vector<ECO_FEATS> get_samples() const { return samples_f_; }

  private:
	cv::Mat distance_matrix_, gram_matrix_; // distance matrix and its kernel

	size_t nSamples_ = 50;

	float learning_rate_ = 0.009;

	const float minmum_sample_weight_ = 0.0036;

	std::vector<float> sample_weight_;

	std::vector<ECO_FEATS> samples_f_; // all samples frontier

	size_t num_training_samples_ = 0;

	std::vector<float> prior_weights_;

	ECO_FEATS new_sample_, merged_sample_;

	int new_sample_id_ = -1, merged_sample_id_ = -1;
};

} // namespace eco

#endif