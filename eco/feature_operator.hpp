#ifndef FEATURE_OPERATOR_HPP
#define FEATURE_OPERATOR_HPP

#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include "parameters.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "debug.hpp"

namespace eco
{
template <typename T>
extern std::vector<T> operator+(const std::vector<T> &a,
								const std::vector<T> &b)
{
	assert(a.size() == b.size());
	std::vector<T> result;
	for (unsigned int i = 0; i < a.size(); ++i)
	{
		result.push_back(a[i] + b[i]);
	}
	return result;
}

template <typename T>
extern std::vector<T> operator-(const std::vector<T> &a,
								const std::vector<T> &b)
{
	assert(a.size() == b.size());
	std::vector<T> result;
	for (unsigned int i = 0; i < a.size(); ++i)
	{
		result.push_back(a[i] - b[i]);
	}
	return result;
}

template <typename T>
extern std::vector<T> operator*(const std::vector<T> &a, const float scale)
{
	std::vector<T> result;
	for (unsigned int i = 0; i < a.size(); ++i)
	{
		result.push_back(a[i] * scale);
	}
	return result;
}

extern ECO_FEATS do_dft(const ECO_FEATS &xlw);
extern ECO_FEATS do_windows(const ECO_FEATS &xl, vector<cv::Mat> &cos_win);

extern void FilterSymmetrize(ECO_FEATS &hf);
extern vector<cv::Mat> init_projection_matrix(const ECO_FEATS &init_sample,
											  const vector<int> &compressed_dim,
											  const vector<int> &feature_dim);
extern ECO_FEATS FeatureProjection(const ECO_FEATS &x,
								   const std::vector<cv::Mat> &projection_matrix);
extern ECO_FEATS FeatureProjectionMultScale(const ECO_FEATS &x,
											const std::vector<cv::Mat> &projection_matrix);

extern float FeatureComputeInnerProduct(const ECO_FEATS &feat1,
										const ECO_FEATS &feat2);
extern float FeatureComputeEnergy(const ECO_FEATS &feat);
extern ECO_FEATS FeautreComputePower2(const ECO_FEATS &feats);
extern std::vector<cv::Mat> FeatureComputeScores(const ECO_FEATS &x,
												 const ECO_FEATS &f);
extern std::vector<cv::Mat> FeatureVectorization(const ECO_FEATS &x);

extern ECO_FEATS FeatureVectorMultiply(const ECO_FEATS &x,
									   const std::vector<cv::Mat> &y,
									   const bool _conj = 0); // feature * yf

extern ECO_FEATS FeatureDotMultiply(const ECO_FEATS &a, const ECO_FEATS &b);
extern ECO_FEATS FeatureDotDivide(const ECO_FEATS &a, const ECO_FEATS &b);
} // namespace eco

#endif