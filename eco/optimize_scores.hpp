#ifndef OPTIMIZE_SCORES_H
#define OPTIMIZE_SCORES_H

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "fftTool.hpp"
#include "debug.hpp"

namespace eco
{
class OptimizeScores
{
  public:
	virtual ~OptimizeScores() {}

	OptimizeScores() {} // default constructor
	OptimizeScores(std::vector<cv::Mat> &pscores_fs, int pite)
		: scores_fs(pscores_fs), iterations(pite) {}

	void compute_scores();

	std::vector<cv::Mat> sample_fs(const std::vector<cv::Mat> &xf,
								   cv::Size grid_sz = cv::Size(0, 0));

	inline int get_scale_ind() const { return scale_ind; }
	inline float get_disp_row() const { return disp_row; }
	inline float get_disp_col() const { return disp_col; }

  private:
	std::vector<cv::Mat> scores_fs;
	int iterations;

	int scale_ind;
	float disp_row;
	float disp_col;
};
} // namespace eco
#endif
