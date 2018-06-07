#ifndef	SAMPLE_UPDATE_H
#define SAMPLE_UPDATA_H

#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>

#include "feature_type.h"
#include "fftTool.h"
#include "feature_operator.h"

using namespace FFTTools;

namespace eco_sample_update{

	class sample_update
	{
	public:
		sample_update(){};

		virtual    ~sample_update(){};

		void       init(const std::vector<cv::Size>& filter, const std::vector<int>& feature_dim, size_t max_samples);

		void       update_sample_space_model( ECO_FEATS& new_train_sample);

		cv::Mat    find_gram_vector( ECO_FEATS& new_train_sample) ;

		float      feat_dis_compute(std::vector<std::vector<cv::Mat> >& feat1, std::vector<std::vector<cv::Mat> >& feat2);

		void       update_distance_matrix(cv::Mat& gram_vector, float new_sample_norm, int id1, int id2, float w1, float w2);

		void       findMin(float& min_w, size_t index)const;

		ECO_FEATS  merge_samples(ECO_FEATS& sample1, ECO_FEATS& sample2, float w1, float w2, std::string sample_merge_type = "merge");

		void       replace_sample(ECO_FEATS& new_sample, size_t idx);

		void       set_gram_matrix(int r, int c, float val);

		int        get_merge_id()const { return merged_sample_id; }

		int        get_new_id()const   { return new_sample_id; }

		std::vector<float>      get_samples_weight()const { return prior_weights; }

		std::vector<ECO_FEATS>  get_samples() const{ return samples_f; }

	private:
		 cv::Mat                    distance_matrix, gram_matrix; //**** distance matrix and its kernel
		  
		 size_t                     nSamples = 50; 

		 const float                learning_rate = 0.009;

		 const float                minmum_sample_weight = 0.0036;

		 std::vector<float>         sample_weight;

		 std::vector<ECO_FEATS>     samples_f; // all samples frontier

		 size_t		                num_training_samples = 0; 

		 std::vector<float>         prior_weights;

		 ECO_FEATS                  new_sample, merged_sample;

		 int                        merged_sample_id = -1, new_sample_id = -1;

	};

};



#endif