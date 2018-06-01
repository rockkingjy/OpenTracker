#ifndef	FEATURE_OPERATOR
#define FEATURE_OPERATOR

#include <opencv2/features2d/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>

#include "fftTool.h"
#include "feature_type.h"
#include "recttools.hpp"

using FFTTools::fftd;

//using namespace std;
template<typename T>
extern std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
	assert(a.size() == b.size());
	std::vector<T> result;
	for (unsigned int i = 0; i < a.size(); ++i)
	{
		result.push_back(a[i] + b[i]);
	}
	return result;
}

template<typename T>
extern std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
	assert(a.size() == b.size());
	std::vector<T> result;
	for (unsigned int i = 0; i < a.size(); ++i)
	{
		result.push_back(a[i] - b[i]);
	}
	return result;
}

extern ECO_FEATS   featDotMul(const ECO_FEATS& a, const ECO_FEATS& b);     // two features dot multiplication

extern ECO_FEATS   project_sample(const ECO_FEATS& x, const std::vector<cv::Mat>& projection_matrix);

extern float       FeatEnergy(ECO_FEATS& feat);

extern ECO_FEATS   feats_pow2(const ECO_FEATS& feats);

extern ECO_FEATS   do_dft(const ECO_FEATS& xlw);
extern  ECO_FEATS  featDotMul(const ECO_FEATS& a, const ECO_FEATS& b);   // two features dot multiplication
extern  ECO_FEATS  FeatDotDivide(ECO_FEATS data1, ECO_FEATS data2);

extern  std::vector<cv::Mat>   computeFeatSores(const ECO_FEATS& x, const ECO_FEATS& f); // compute socres  Sum(x * f)
extern  ECO_FEATS              computerFeatScores2(const ECO_FEATS& x, const ECO_FEATS& f);

extern  ECO_FEATS  FeatScale(ECO_FEATS data, float scale);

extern  void       symmetrize_filter(ECO_FEATS& hf);
extern  float      FeatEnergy(ECO_FEATS& feat);
extern  std::vector<cv::Mat>      FeatVec(const ECO_FEATS& x);   // vectorize features

extern  ECO_FEATS  FeatProjMultScale(const ECO_FEATS& x, const std::vector<cv::Mat>& projection_matrix);

extern  std::vector<cv::Mat>  ProjScale(std::vector<cv::Mat> data, float scale);





#endif 