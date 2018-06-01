#include <vector>
#include <opencv2/features2d/features2d.hpp>
//#include <core\core.hpp>
#include <opencv2/opencv.hpp>

typedef   std::vector<std::vector<cv::Mat> > ECO_FEATS; // *** one of kind of ECO features class
typedef   cv::Vec<float, 2>                  COMPLEX;   // *** the complex number reprsentation 