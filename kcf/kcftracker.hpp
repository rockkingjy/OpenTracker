/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: horizontal area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_
#endif

namespace kcf
{
class KCFTracker
{
public:
    // Constructor
    KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true, bool dsst = false);

    // Initialize tracker 
    void init( const cv::Mat image, const cv::Rect2d& roi);
    
    // Update position based on the new frame
    //cv::Rect update(cv::Mat image);
    bool update( const cv::Mat image, cv::Rect2d& roi);

    float detect_thresh_kcf; // thresh hold for tracking error or not
    float sigma; // gaussian kernel bandwidth
    float lambda; // regularization
    float interp_factor; // linear interpolation factor for adaptation
    int cell_size; // HOG cell size
    int cell_sizeQ; // cell size^2, to avoid repeated operations
    float padding; // extra area surrounding the target
    float output_sigma_factor; // bandwidth of gaussian target
    int template_size; // template size

    float scale_step; // scale step for multi-scale estimation
    float scale_weight;  // to downweight detection scores of other scales for added stability

//=====dsst====
    float detect_thresh_dsst; // thresh hold for tracking error or not
    int base_width_dsst; // initial ROI widt
    int base_height_dsst; // initial ROI height
    int scale_max_area; // max ROI size before compressing
    float scale_padding; // extra area surrounding the target for scaling
//    float scale_step; // scale step for multi-scale estimation
    float scale_sigma_factor; // bandwidth of gaussian target
    int n_scales; // # of scaling windows
    float scale_lr; // scale learning rate
    float *scaleFactors; // all scale changing rate, from larger to smaller with 1 to be the middle
    int scale_model_width; // the model width for scaling
    int scale_model_height; // the model height for scaling
    float min_scale_factor; // min scaling rate
    float max_scale_factor; // max scaling rate
    float scale_lambda; // regularization
//===========

protected:
    bool update_kcf( const cv::Mat image, cv::Rect2d& roi);
    // Detect object in the current frame.
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value); // paper Algorithm 1 , _alpha updated in train();

    // train tracker with a single image, to update _alphaf;
    void train(cv::Mat x, float train_interp_factor);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2); // paper (30)

    // Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(int sizey, int sizex);

    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

    // Initialize Hanning window. Function called only in the first frame.
    void createHanningMats();

    // Calculate sub-pixel peak for one dimension
    float subPixelPeak(float left, float center, float right);

//=====dsst====
    // Initialization for scales
    void init_dsst(const cv::Mat image, const cv::Rect2d& roi);

    bool update_dsst( const cv::Mat image, cv::Rect2d& roi);
    // Detect the new scaling rate
    cv::Point2i detect_dsst(cv::Mat image);

    // Train method for scaling
    void train_dsst(cv::Mat image, bool ini = false);

    // Compute the F^l in the paper
    cv::Mat get_sample_dsst(const cv::Mat & image);

    // Compute the FFT Guassian Peak for scaling
    cv::Mat createGaussianPeak_dsst();

    // Compute the hanning window for scaling
    cv::Mat createHanningMats_dsst();
//===========

    cv::Mat _alphaf;//alpha in paper, use this to calculate the detect result, changed in train();
    cv::Mat _prob;  //Gaussian Peak(training outputs);
    cv::Mat _tmpl;  //features of image (or the normalized gray image itself when raw), changed in train();
    cv::Mat _num;   //numerator: use to update as MOSSE
    cv::Mat _den;   //denumerator: use to update as MOSSE
    cv::Mat _labCentroids;

    cv::Rect_<float> _roi;

private:
    int _size_patch[3];//0:rows;1:cols;2:numFeatures; init in getFeatures();
    cv::Mat _hann;
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    bool _hogfeatures;
    bool _labfeatures;
    float _peak_value;

//=====dsst====
    bool _dsst;
    float _scale_dsst;
    cv::Mat _den_dsst;
    cv::Mat _num_dsst;
    cv::Mat _hann_dsst;
    cv::Mat _prob_dsst;
//============    
};
}