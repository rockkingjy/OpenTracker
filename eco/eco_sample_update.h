#pragma once

#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>

#include "feature_type.h"
#include "fftTool.h"
#include "feature_operator.h"
#include "debug.h"

using namespace FFTTools;

namespace eco_sample_update{

	class sample_update
	{
	public:
		sample_update(){};
		virtual ~sample_update(){};

		void init(const std::vector<cv::Size>& filter, const std::vector<int>& feature_dim, size_t max_samples);
		void update_sample_space_model( ECO_FEATS& new_train_sample);
		void update_distance_matrix(cv::Mat& gram_vector, float new_sample_norm, 
										  int id1, int id2, float w1, float w2);

		inline cv::Mat find_gram_vector( ECO_FEATS& new_train_sample) 
		{
			cv::Mat result(cv::Size(1, nSamples), CV_32FC2);
			for (size_t i = 0; i < (size_t)result.rows; i++) // init to INF;
				result.at<cv::Vec<float, 2>>(i, 0) = cv::Vec<float, 2>(INF, 0);

			std::vector<float> dist_vec;
			for (size_t i = 0; i < num_training_samples; i++) // calculate the distance;
				dist_vec.push_back(2 * feat_dis_compute(samples_f[i], new_train_sample));

			for (size_t i = 0; i < dist_vec.size(); i++)
				result.at<cv::Vec<float, 2>>(i, 0) = cv::Vec<float, 2>(dist_vec[i], 0);

			return result;
		};

		inline void findMin(float& min_w, size_t index) const
		{
			std::vector<float>::const_iterator pos = std::min_element(prior_weights.begin(), prior_weights.end());
			min_w = *pos;
			index = pos - prior_weights.begin();
		};

		inline ECO_FEATS merge_samples(ECO_FEATS& sample1, ECO_FEATS& sample2, float w1, float w2, 
									std::string sample_merge_type = "merge")
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

		inline void replace_sample(ECO_FEATS& new_sample, size_t idx)
		{
			samples_f[idx] = new_sample;
		};

		inline void set_gram_matrix(int r, int c, float val)
		{
			gram_matrix.at<float>(r, c) = val;
		};

		int get_merge_id()const { return merged_sample_id; }

		int get_new_id()const   { return new_sample_id; }

		std::vector<float>      get_samples_weight()const { return prior_weights; }

		std::vector<ECO_FEATS>  get_samples() const{ return samples_f; }

	private:
		 cv::Mat                    distance_matrix, gram_matrix; //**** distance matrix and its kernel
		  
		 size_t                     nSamples = 50; 

		 const float                learning_rate = 0.009;

		 const float                _minmum_sample_weight = 0.0036;

		 std::vector<float>         sample_weight;

		 std::vector<ECO_FEATS>     samples_f; // all samples frontier

		 size_t		                num_training_samples = 0; 

		 std::vector<float>         prior_weights;

		 ECO_FEATS                  new_sample, merged_sample;

		 int                        merged_sample_id = -1, new_sample_id = -1;

	};

};
