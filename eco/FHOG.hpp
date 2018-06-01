/*
C++ Felzenszwalb HOG extractor
 
This repository is meant to provide an easy-to-use implementation of the Felzenszwalb HOG features extractor.
This approach followed the one presented by Felzenszwalb, Pedro F., et al. "Object detection with discriminatively trained part-based models." Pattern Analysis and Machine Intelligence, IEEE Transactions on 32.9 (2010): 1627-1645. 
The OpenCV library have only the original HOG, proposed by Dalal and Triggs. However, the Latent SVM OpenCV implementation have its own FHOG extractor. This code allows you to use it without having do deal with Latent SVM objects.

To run the code you need OpenCV library.

Author: Joao Faro
Contacts: joaopfaro@gmail.com
*/

#ifndef FHOG_H
#define FHOG_H

#include <opencv2/opencv.hpp> 
#include "fhog_f.hpp"

//typedef struct{
//  int sizeX;
//  int sizeY;
//  int numFeatures;
//  float *map;
//} CvLSVMFeatureMapCaskade;

typedef unsigned int uint;

//int getFeatureMaps(const IplImage*, const int, CvLSVMFeatureMapCaskade **);
//int normalizeAndTruncate(CvLSVMFeatureMapCaskade *, const float);
//int PCAFeatureMaps(CvLSVMFeatureMapCaskade *);
//int freeFeatureMapObject(CvLSVMFeatureMapCaskade **);

class HogFeature {

	public:
	HogFeature();
	HogFeature(uint, uint);
	virtual ~HogFeature();
	virtual HogFeature* clone() const;
	virtual cv::Mat getFeature(cv::Mat);

	private:
		uint _cell_size;
		uint _scale;
		cv::Size _tmpl_sz;
		cv::Mat _featuresMap;
		cv::Mat _featurePaddingMat;
		CvLSVMFeatureMapCaskade *_map;
};

#endif
