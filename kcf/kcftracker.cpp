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


Constructor parameters, al l boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
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

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#endif

namespace kcf
{
// Constructor
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab, bool dsst)
{
    // Parameters equal in all cases
    detect_thresh_kcf = 0.13;
    detect_thresh_dsst = 0.15;
    lambda = 0.0001;
    padding = 2.5;
    output_sigma_factor = 0.125; //0.1

    if (hog)
    { // HOG - KCF
        // VOT
        interp_factor = 0.012;
        sigma = 0.6;
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5;
        cell_size = 4; //hog cell size = 4;
        _hogfeatures = true;

        if (lab)
        {
            interp_factor = 0.005;
            sigma = 0.4;
            output_sigma_factor = 0.1; //0.025;
            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data); //create lab centroids mat according to labdata.hpp
            cell_sizeQ = cell_size * cell_size;
        }
        else
        {
            _labfeatures = false;
        }
    }
    else
    { // RAW - CSK
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1; //just pixel;
        _hogfeatures = false;

        if (lab)
        {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }

    if (multiscale)
    {                       // multiscale
        template_size = 96; //100;
        scale_step = 1.05;
        scale_weight = 0.95;

        if (!fixed_window)
        {
            printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window)
    {                       // fit correction without multiscale
        template_size = 96; //100;
        scale_step = 1;
    }
    else
    {
        template_size = 1;
        scale_step = 1;
    }
    //dsst===================
    if (dsst)
    {
        scale_step = 1.05;

        _dsst = true;
        _scale_dsst = 1;
        scale_padding = 1.0;
        scale_sigma_factor = 0.25;
        n_scales = 33;
        scale_lr = 0.025;
        scale_max_area = 512;
        scale_lambda = 0.01;
    }
    else
    {
        _dsst = false;
    }
}
// Initialize tracker
void KCFTracker::init(const cv::Mat image, const cv::Rect2d &roi)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);
    _prob = createGaussianPeak(_size_patch[0], _size_patch[1]);
    _alphaf = cv::Mat(_size_patch[0], _size_patch[1], CV_32FC2, float(0));
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame

    if (_dsst)
    {
        init_dsst(image, roi);
        train_dsst(image, true);
    }
}
// Update position based on the new frame
bool KCFTracker::update(const cv::Mat image, cv::Rect2d &roi)
{
    if (_dsst)
    {
        return update_dsst(image, roi);
    }
    else
    {
        return update_kcf(image, roi);
    }
}

bool KCFTracker::update_kcf(const cv::Mat image, cv::Rect2d &roi)
{
    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 1; //let _roi.x + _roi.width = 1
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1)
        _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1)
        _roi.y = image.rows - 2;
    if (_roi.width <= 0)
        _roi.width = 2;
    if (_roi.height <= 0)
        _roi.height = 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), _peak_value); //translation estimation;

    if (scale_step != 1)
    {
        // Detect at a smaller _scale
        float new_peak_value;
        cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);
        if (scale_weight * new_peak_value > _peak_value)
        {
            res = new_res;
            _peak_value = new_peak_value;
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }
        // Detect at a bigger _scale
        new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);
        if (scale_weight * new_peak_value > _peak_value)
        {
            res = new_res;
            _peak_value = new_peak_value;
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }
    }

    //printf("kcf thresh: %f, peak: %f\n", detect_thresh_kcf, _peak_value);
    if (_peak_value >= detect_thresh_kcf)
    { // Adjust by cell size and _scale
        _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
        _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

        if (_roi.x + _roi.width <= 0)
            _roi.x = -_roi.width + 2;
        if (_roi.y + _roi.height <= 0)
            _roi.y = -_roi.height + 2;
        if (_roi.x >= image.cols - 1)
            _roi.x = image.cols - 1;
        if (_roi.y >= image.rows - 1)
            _roi.y = image.rows - 1;
        if (_roi.width <= 0)
            _roi.width = 2;
        if (_roi.height <= 0)
            _roi.height = 2;

        assert(_roi.width >= 0 && _roi.height >= 0);
        train(getFeatures(image, 0), interp_factor);

        roi = _roi;
        return true;
    }
    else
    {
        return false;
    }
}

bool KCFTracker::update_dsst(const cv::Mat image, cv::Rect2d &roi)
{
    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 1; //let _roi.x + _roi.width = 1
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1)
        _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1)
        _roi.y = image.rows - 2;
    if (_roi.width <= 0)
        _roi.width = 2;
    if (_roi.height <= 0)
        _roi.height = 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    //translation estimation;
    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), _peak_value);

    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale_dsst);
    _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale_dsst);

    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 2;
    if (_roi.x >= image.cols - 1)
        _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1)
        _roi.y = image.rows - 1;
    if (_roi.width <= 0)
        _roi.width = 2;
    if (_roi.height <= 0)
        _roi.height = 2;

    //scale estimation;
    cv::Point2i scale_pi = detect_dsst(image);

    //printf("dsst thresh: %f, peak: %f\n", detect_thresh_dsst, _peak_value);
    if (_peak_value >= detect_thresh_dsst)
    {
        _scale_dsst = _scale_dsst * scaleFactors[scale_pi.x];
        //printf("scale_pi.x:%d, _scale_dsst:%f\n", scale_pi.x, _scale_dsst);
        if (_scale_dsst < min_scale_factor)
            _scale_dsst = min_scale_factor;
        else if (_scale_dsst > max_scale_factor)
            _scale_dsst = max_scale_factor;

        // Compute new _roi
        cx = _roi.x + _roi.width / 2.0f;
        cy = _roi.y + _roi.height / 2.0f;
        _roi.width = base_width_dsst * _scale_dsst;
        _roi.height = base_height_dsst * _scale_dsst;
        _roi.x = cx - _roi.width / 2.0f;
        _roi.y = cy - _roi.height / 2.0f;

        if (_roi.x + _roi.width <= 0)
            _roi.x = -_roi.width + 2;
        if (_roi.y + _roi.height <= 0)
            _roi.y = -_roi.height + 2;
        if (_roi.x >= image.cols - 1)
            _roi.x = image.cols - 1;
        if (_roi.y >= image.rows - 1)
            _roi.y = image.rows - 1;
        if (_roi.width <= 0)
            _roi.width = 2;
        if (_roi.height <= 0)
            _roi.height = 2;

        assert(_roi.width >= 0 && _roi.height >= 0);
        train(getFeatures(image, 0), interp_factor);

        train_dsst(image);

        roi = _roi;
        return true;
    }
    else
    {
        return false;
    }
}

// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value) // KCF Algorithm 1 , _alpha updated in train();
{
    cv::Mat k = gaussianCorrelation(x, z);
    cv::Mat res = (real(dft_d(complexDotMultiplication(_alphaf, dft_d(k)), true))); // KCF (22)

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float)pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols - 1)
    {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
    }

    if (pi.y > 0 && pi.y < res.rows - 1)
    {
        p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}

// train tracker with a single image, to update _alphaf;
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{

    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDotDivision(_prob, (dft_d(k) + lambda)); // KCF (17)

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)*x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)*alphaf;

    /*//MOSSE-style update
    cv::Mat kf = dft_d(gaussianCorrelation(x, x));
    cv::Mat num = complexDotMultiplication(kf, _prob);
    cv::Mat den = complexDotMultiplication(kf, kf + lambda);
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDotDivision(_num, _den);*/
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y,
// which must both be MxN. They must also be periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2) // KCF (30)
{

    cv::Mat c = cv::Mat(cv::Size(_size_patch[1], _size_patch[0]), CV_32F, cv::Scalar(0));
    // HOG features
    if (_hogfeatures)
    {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < _size_patch[2]; i++)
        {
            x1aux = x1.row(i); // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, _size_patch[0]);
            x2aux = x2.row(i).reshape(1, _size_patch[0]);
            cv::mulSpectrums(dft_d(x1aux), dft_d(x2aux), caux, 0, true);
            caux = dft_d(caux, true);
            rearrange(caux);
            caux.convertTo(caux, CV_32F);
            c = c + real(caux);
        }
    }
    // Gray features
    else
    {
        cv::mulSpectrums(dft_d(x1), dft_d(x2), c, 0, true); //output: c
        c = dft_d(c, true);                                //ifft
        rearrange(c);                                     //KCF page 11, Figure 6;
        c = real(c);
    }
    cv::Mat d;
    //make sure >=0 and scaling for the fft; KCF page 11
    cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (_size_patch[0] * _size_patch[1] * _size_patch[2]), 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{

    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    //    printf("sizex:%d, sizey:%d\n",sizex,sizey);
    //    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    //    float mult = -(float)0.5/ (output_sigma * output_sigma);
    //
    //    output_sigma_factor = 0.1;
    //    float sigmax2 = -0.5*padding*padding/output_sigma_factor/output_sigma_factor/(sizex);
    //    float sigmay2 = -0.5*padding*padding/output_sigma_factor/output_sigma_factor/(sizey);
    //
    //
    //    for (int i = 0; i < sizey; i++)
    //        for (int j = 0; j < sizex; j++)
    //        {
    //            int ih = i - syh;
    //            int jh = j - sxh;
    //            res(i, j) = std::exp( sigmay2*ih * ih + sigmax2*jh * jh);
    //        }
    // exp(-(1/2)(sigma^2)(padding^2/(sizex*sizey)) * (x^2+y^2))
    float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
        }
    return dft_d(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getFeatures(const cv::Mat &image, bool inithann, float scale_adjust)
{
    cv::Rect extracted_roi;

    //printf("_roi:%f,%f,%f,%f\n",_roi.x,_roi.y,_roi.width,_roi.height);
    // get the centor of roi
    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;
    //printf("cxcy:%f,%f\n", cx, cy);
    if (inithann)
    {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;

        if (template_size > 1)
        {                             // Fit largest dimension to the given template size
            if (padded_w >= padded_h) //fit to width
                _scale = padded_w / (float)template_size;
            else
                _scale = padded_h / (float)template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else
        { //No template size given, use ROI size
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
            // original code from paper:
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2;
            }*/
        }
        // Round the _tmpl_sz
        if (_hogfeatures)
        {
            // Round to cell size and also make it even
            _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
            _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
        }
        else
        { //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }

    // get new extracted_roi
    //printf("c:%f,%f,%d,%d,%f,%f\n", scale_adjust, _scale, _tmpl_sz.width, _tmpl_sz.height, cx, cy);
    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    if (extracted_roi.x + extracted_roi.width <= 0)
        extracted_roi.x = -extracted_roi.width + 2;
    if (extracted_roi.y + extracted_roi.height <= 0)
        extracted_roi.y = -extracted_roi.height + 2;
    if (extracted_roi.x >= image.cols - 1)
        extracted_roi.x = image.cols - 1;
    if (extracted_roi.y >= image.rows - 1)
        extracted_roi.y = image.rows - 1;
    if (extracted_roi.width <= 0)
        extracted_roi.width = 2;
    if (extracted_roi.height <= 0)
        extracted_roi.height = 2;

    //printf("extracted_roi:%d,%d,%d,%d\n", extracted_roi.x, extracted_roi.y, extracted_roi.width, extracted_roi.height);
    cv::Mat FeaturesMap;
    cv::Mat z = subwindow(image, extracted_roi, cv::BORDER_REPLICATE);

    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height)
    {
        cv::resize(z, z, _tmpl_sz);
    }
    //printf("%d, %d \n", z.cols, z.rows);
    //double timereco = (double)cv::getTickCount();
	//float fpseco = 0;
    // HOG features
    if (_hogfeatures)
    {
        IplImage z_ipl = z;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, cell_size, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        _size_patch[0] = map->sizeY;
        _size_patch[1] = map->sizeX;
        _size_patch[2] = map->numFeatures;

        FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map); // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();                                                                // transpose
        freeFeatureMapObject(&map);

        // Lab features
        if (_labfeatures)
        {
            cv::Mat imgLab;
            cvtColor(z, imgLab, CV_BGR2Lab);
            unsigned char *input = (unsigned char *)(imgLab.data);

            // Sparse output vector
            cv::Mat outputLab = cv::Mat(_labCentroids.rows, _size_patch[0] * _size_patch[1], CV_32F, float(0));

            int cntCell = 0;
            // Iterate through each cell
            for (int cY = cell_size; cY < z.rows - cell_size; cY += cell_size)
            {
                for (int cX = cell_size; cX < z.cols - cell_size; cX += cell_size)
                {
                    // Iterate through each pixel of cell (cX,cY)
                    for (int y = cY; y < cY + cell_size; ++y)
                    {
                        for (int x = cX; x < cX + cell_size; ++x)
                        {
                            // Lab components for each pixel
                            float l = (float)input[(z.cols * y + x) * 3];
                            float a = (float)input[(z.cols * y + x) * 3 + 1];
                            float b = (float)input[(z.cols * y + x) * 3 + 2];

                            // Iterate trough each centroid, to see it belongs to which centroid;
                            float minDist = FLT_MAX;
                            int minIdx = 0;
                            float *inputCentroid = (float *)(_labCentroids.data);
                            for (int k = 0; k < _labCentroids.rows; ++k)
                            {
                                float dist = ((l - inputCentroid[3 * k]) * (l - inputCentroid[3 * k])) +
                                             ((a - inputCentroid[3 * k + 1]) * (a - inputCentroid[3 * k + 1])) +
                                             ((b - inputCentroid[3 * k + 2]) * (b - inputCentroid[3 * k + 2]));
                                if (dist < minDist)
                                {
                                    minDist = dist;
                                    minIdx = k;
                                }
                            }
                            // Store result at output
                            outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
                            //((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ;
                        }
                    }
                    cntCell++;
                }
            }
            // Update size_patch[2] and stack lab features to FeaturesMap
            _size_patch[2] += _labCentroids.rows;
            FeaturesMap.push_back(outputLab);
        }
    }
    else //CSK
    {
        FeaturesMap = getGrayImage(z);
        FeaturesMap -= (float)0.5; // CSK p10;
        _size_patch[0] = z.rows;
        _size_patch[1] = z.cols;
        _size_patch[2] = 1;
    }
	//fpseco = ((double)cv::getTickCount() - timereco) / cv::getTickFrequency();
	//printf("kcf hog extra time: %f \n", fpseco);
    if (inithann)
    {
        createHanningMats();
    }
    FeaturesMap = _hann.mul(FeaturesMap); //element-wise multiplication;
    return FeaturesMap;
}

// Initialize Hanning window. Function called only in the first frame. To make the boundary of feature(image) to be zero
void KCFTracker::createHanningMats()
{                                                                                 //Mat(size(cols,rows), type, init)
    cv::Mat hann1t = cv::Mat(cv::Size(_size_patch[1], 1), CV_32F, cv::Scalar(0)); //1 x size_patch[1]
    cv::Mat hann2t = cv::Mat(cv::Size(1, _size_patch[0]), CV_32F, cv::Scalar(0)); //size_patch[0] x 1

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t; //size_patch[0] x size_patch[1]
    // HOG features
    if (_hogfeatures)
    {
        cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

        _hann = cv::Mat(cv::Size(_size_patch[0] * _size_patch[1], _size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < _size_patch[2]; i++)
        {
            for (int j = 0; j < _size_patch[0] * _size_patch[1]; j++)
            {
                _hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
    }
    // Gray features
    else
    {
        _hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}
//DSST=========================================================================================
// Initialization for scales
void KCFTracker::init_dsst(const cv::Mat image, const cv::Rect2d &roi)
{
    // The initial size for adjusting
    base_width_dsst = roi.width;
    base_height_dsst = roi.height;

    // Guassian peak for scales (after fft)
    _prob_dsst = createGaussianPeak_dsst();
    _hann_dsst = createHanningMats_dsst();

    // Get all scale changing rate, DSST page 5;
    scaleFactors = new float[n_scales];
    float ceilS = std::ceil(n_scales / 2.0f);
    for (int i = 0; i < n_scales; i++)
    {
        scaleFactors[i] = std::pow(scale_step, ceilS - i - 1);
        //    printf("scaleFactors %d: %f; ", i, scaleFactors[i]);
    }
    printf("\n");

    // Get the scaling rate for compressing to the model size
    float scale_model_factor = 1;
    if (base_width_dsst * base_height_dsst > scale_max_area)
    {
        scale_model_factor = std::sqrt(scale_max_area / (float)(base_width_dsst * base_height_dsst));
    }

    scale_model_width = (int)(base_width_dsst * scale_model_factor);
    scale_model_height = (int)(base_height_dsst * scale_model_factor);
    //printf("%d, %d \n", scale_model_width, scale_model_height);
    // Compute min and max scaling rate
    min_scale_factor = 0.01; //std::pow(scale_step,
                             //     std::ceil(std::log((std::fmax(5 / (float)base_width, 5 / (float)base_height) * (1 + scale_padding))) / 0.0086));
    max_scale_factor = 10;   //std::pow(scale_step,
                             //      std::floor(std::log(std::fmin(image.rows / (float)base_height, image.cols / (float)base_width)) / 0.0086));
    //printf("dsstInit - min_scale_factor:%f; max_scale_factor:%f;\n", min_scale_factor, max_scale_factor);
}

// Train method for scaling
void KCFTracker::train_dsst(cv::Mat image, bool ini)
{
    cv::Mat samples = get_sample_dsst(image);

    // Adjust ysf to the same size as xsf in the first frame
    if (ini)
    {
        int totalSize = samples.rows;
        _prob_dsst = cv::repeat(_prob_dsst, totalSize, 1);
    }

    // Get new GF in the paper (delta A)
    cv::Mat new_num_dsst;
    cv::mulSpectrums(_prob_dsst, samples, new_num_dsst, 0, true);

    // Get Sigma{FF} in the paper (delta B)
    cv::Mat new_den_dsst;
    cv::mulSpectrums(samples, samples, new_den_dsst, 0, true);
    cv::reduce(real(new_den_dsst), new_den_dsst, 0, CV_REDUCE_SUM);

    if (ini)
    {
        _den_dsst = new_den_dsst;
        _num_dsst = new_num_dsst;
    }
    else
    {
        // Get new A and new B, DSST (5)
        cv::addWeighted(_den_dsst, (1 - scale_lr), new_den_dsst, scale_lr, 0, _den_dsst);
        cv::addWeighted(_num_dsst, (1 - scale_lr), new_num_dsst, scale_lr, 0, _num_dsst);
    }
}

// Detect the new scaling rate
cv::Point2i KCFTracker::detect_dsst(cv::Mat image)
{

    cv::Mat samples = KCFTracker::get_sample_dsst(image);

    // Compute AZ in the paper
    cv::Mat add_temp;
    cv::reduce(complexDotMultiplication(_num_dsst, samples), add_temp, 0, CV_REDUCE_SUM);

    // compute the final y, DSST (6);
    cv::Mat scale_response;
    cv::idft(complexDotDivisionReal(add_temp, (_den_dsst + scale_lambda)), scale_response, cv::DFT_REAL_OUTPUT);

    // Get the max point as the final scaling rate
    cv::Point2i pi; //max location
    double pv;      //max value
    cv::minMaxLoc(scale_response, NULL, &pv, NULL, &pi);

    return pi;
}

// Compute the F^l in DSST (4);
cv::Mat KCFTracker::get_sample_dsst(const cv::Mat &image)
{
	//double timereco = (double)cv::getTickCount();
	//float fpseco = 0;

    CvLSVMFeatureMapCaskade *map[n_scales]; // temporarily store FHOG result
    cv::Mat samples;                        // output
    int totalSize;                          // # of features
    // iterate for each scale
    for (int i = 0; i < n_scales; i++)
    {
        // Size of subwindow waiting to be detect
        float patch_width = base_width_dsst * scaleFactors[i] * _scale_dsst;
        float patch_height = base_height_dsst * scaleFactors[i] * _scale_dsst;

        float cx = _roi.x + _roi.width / 2.0f;
        float cy = _roi.y + _roi.height / 2.0f;

        // Get the subwindow
        cv::Mat im_patch = extractImage(image, cx, cy, patch_width, patch_height);
        cv::Mat im_patch_resized;

        //printf("cx:%f,cy:%f\n",cx,cy);
        //printf("patch_width: %f, patch_height: %f,\n",patch_width,patch_height);
        //printf("im_patch w: %d, im_path h: %d,\n", im_patch.rows, im_patch.cols);

        if (im_patch.rows == 0 || im_patch.cols == 0)
        {
            samples = dft_d(samples, 0, 1);
            return samples;
            // map[i]->map = (float *)malloc (sizeof(float));
        }

        // Scaling the subwindow
        if (scale_model_width > im_patch.cols)
            resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, 1);
        else
            resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, 3);
        
        //printf("%d, %d \n", im_patch_resized.cols, im_patch_resized.rows);
        //printf("%d, %d \n", im_patch.cols, im_patch.rows);
        // Compute the FHOG features for the subwindow
        IplImage im_ipl = im_patch_resized;
        getFeatureMaps(&im_ipl, cell_size, &map[i]);
        normalizeAndTruncate(map[i], 0.2f);
        PCAFeatureMaps(map[i]);

        if (i == 0)
        {
            //printf("numFeatures:%d, sizeX:%d,sizeY:%d\n", map[i]->numFeatures, map[i]->sizeX, map[i]->sizeY);
            totalSize = map[i]->numFeatures * map[i]->sizeX * map[i]->sizeY;

            if (totalSize <= 0) //map[i]->sizeX or Y could be 0 if the roi is too small!!!!!!!!!!!!!!!!!!!!!!!
            {
                totalSize = 1;
            }
            samples = cv::Mat(cv::Size(n_scales, totalSize), CV_32F, float(0));
        }
        cv::Mat FeaturesMap;
        if (map[i]->map != NULL)
        {
            // Multiply the FHOG results by hanning window and copy to the output
            FeaturesMap = cv::Mat(cv::Size(1, totalSize), CV_32F, map[i]->map);
            float mul = _hann_dsst.at<float>(0, i);

            FeaturesMap = mul * FeaturesMap;
            FeaturesMap.copyTo(samples.col(i));
        }
    }
	//fpseco = ((double)cv::getTickCount() - timereco) / cv::getTickFrequency();
	//printf("kcf hog extra time: %f \n", fpseco);

    // Free the temp variables
    for (int i = 0; i < n_scales; i++)
    {
        freeFeatureMapObject(&map[i]);
    }
    // Do fft to the FHOG features row by row
    samples = dft_d(samples, 0, 1);

    return samples;
}

// Compute the FFT Guassian Peak for scaling
cv::Mat KCFTracker::createGaussianPeak_dsst()
{

    float scale_sigma2 = n_scales / std::sqrt(n_scales) * scale_sigma_factor;
    scale_sigma2 = scale_sigma2 * scale_sigma2;
    cv::Mat res(cv::Size(n_scales, 1), CV_32F, float(0));
    float ceilS = std::ceil(n_scales / 2.0f);

    for (int i = 0; i < n_scales; i++)
    {
        res.at<float>(0, i) = std::exp(-0.5 * std::pow(i + 1 - ceilS, 2) / scale_sigma2);
    }

    return dft_d(res);
}

// Compute the hanning window for scaling
cv::Mat KCFTracker::createHanningMats_dsst()
{
    cv::Mat hann_s = cv::Mat(cv::Size(n_scales, 1), CV_32F, cv::Scalar(0));
    for (int i = 0; i < hann_s.cols; i++)
        hann_s.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann_s.cols - 1)));

    return hann_s;
}
}