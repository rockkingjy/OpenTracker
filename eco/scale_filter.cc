#include "scale_filter.hpp"

namespace eco
{
void ScaleFilter::init(int &nScales, float &scale_step, const EcoParameters &params)
{
    nScales = params.number_of_scales_filter;
    scale_step = params.scale_step_filter;
    float scale_sigma = params.number_of_interp_scales * params.scale_sigma_factor;
    vector<float> scale_exp, scale_exp_shift;
    int scalemin = floor((1.0 - (float)nScales) / 2.0);
    int scalemax = floor(((float)nScales - 1.0) / 2.0);
    for (int i = scalemin; i <= scalemax; i++)
    {
        scale_exp.push_back(i * params.number_of_interp_scales / (float)nScales);
    }
    for (int i = 0; i < nScales; i++)
    {
        scale_exp_shift.push_back(scale_exp[(i + nScales / 2) % nScales]);
    }
    /*    debug("scale: min:%d, max:%d", scalemin, scalemax);
    debug("scale_exp_shift:");
    for (int i = 0; i < nScales; i++)
    {
        printf("%d:%f; ", i, scale_exp_shift[i]);
    }
    printf("\n");
*/
    vector<float> interp_scale_exp, interp_scale_exp_shift;
    scalemin = floor((1.0 - (float)params.number_of_interp_scales) / 2.0);
    scalemax = floor(((float)params.number_of_interp_scales - 1.0) / 2.0);
    for (int i = scalemin; i <= scalemax; i++)
    {
        interp_scale_exp.push_back(i);
    }
    for (int i = 0; i < params.number_of_interp_scales; i++)
    {
        interp_scale_exp_shift.push_back(interp_scale_exp[(i + params.number_of_interp_scales / 2) % params.number_of_interp_scales]);
    }
    /*    debug("scale: min:%d, max:%d", scalemin, scalemax);
    debug("interp_scale_exp_shift:");
    for (int i = 0; i < params.number_of_interp_scales; i++)
    {
        printf("%d:%f; ", i, interp_scale_exp_shift[i]);
    }
    printf("\n");
*/
    for (int i = 0; i < nScales; i++)
    {
        scaleSizeFactors_.push_back(std::pow(scale_step, scale_exp[i]));
    }
    /*    debug("scaleSizeFactors_:");
    for (int i = 0; i < nScales; i++)
    {
        printf("%d:%f; ", i, scaleSizeFactors_[i]);
    }
    printf("\n");
*/
    for (int i = 0; i < params.number_of_interp_scales; i++)
    {
        interpScaleFactors_.push_back(std::pow(scale_step, interp_scale_exp_shift[i]));
    }
    /*    debug("interpScaleFactors_:");
    for (int i = 0; i < params.number_of_interp_scales; i++)
    {
        printf("%d:%f; ", i, interpScaleFactors_[i]);
    }
    printf("\n");
*/

    cv::Mat ys_mat = cv::Mat(cv::Size(nScales, 1), CV_32FC1);
    for (int i = 0; i < nScales; i++)
    {
        ys_mat.at<float>(0, i) = std::exp(-0.5f * scale_exp_shift[i] * scale_exp_shift[i] / scale_sigma / scale_sigma);
    }
    /*
    debug("ys:");
    printMat(ys_mat);
    showmat1channels(ys_mat,2);
*/
    yf_ = real(dft(ys_mat, false));
    /*
    debug("yf:");
    printMat(yf_);
    showmat1channels(yf_,2);
*/

    for (int i = 0; i < nScales; i++)
    {
        window_.push_back(0.5f * (1.0f - std::cos(2 * M_PI * i / (nScales - 1.0f))));
    }
    /*
    debug("window_:");
    for (int i = 0; i < nScales; i++)
    {
        printf("%d:%f; ", i, window_[i]);
    }
*/
    //max_scale_dim_ = !params.s_num_compressed_dim.compare("MAX");
    //debug("max_scale_dim_: %d", max_scale_dim_);
}

float ScaleFilter::scale_filter_track(const cv::Mat &im, const cv::Point2f &pos, const cv::Size2f &base_target_sz, const float &currentScaleFactor, const EcoParameters &params)
{
    debug("%f", currentScaleFactor);
    vector<float> scales;
    for (unsigned int i = 0; i < scaleSizeFactors_.size(); i++)
    {
        scales.push_back(scaleSizeFactors_[i] * currentScaleFactor);
        //printf("%f ", scaleSizeFactors_[i]);
    }
    cv::Mat xs = extract_scale_sample(im, pos, base_target_sz, scales, params.scale_model_sz);

    debug("Not finished!-------------------");
    assert(0);

    float scale_change_factor;
    return scale_change_factor;
}

cv::Mat ScaleFilter::extract_scale_sample(const cv::Mat &im, const cv::Point2f &posf, const cv::Size2f &base_target_sz, vector<float> &scaleFactors, const cv::Size &scale_model_sz)
{
    //printMat(new_im);
    //showmat3channels(new_im, 0);
    //debug("pos: %f %f", posf.x, posf.y);
    cv::Point2i pos(posf);
    int nScales = scaleFactors.size();
    int df = std::floor(*std::min_element(std::begin(scaleFactors), std::end(scaleFactors)));
    // debug("df:%d", df);

    cv::Mat new_im;
    im.copyTo(new_im);
    if (df > 1)
    {
        // compute offset and new center position
        cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
        pos.x = (pos.x - os.x - 1) / df + 1;
        pos.y = (pos.y - os.y - 1) / df + 1;

        for (unsigned int i = 0; i < scaleFactors.size(); i++)
        {
            scaleFactors[i] /= df;
        }
        // down sample image
        int r = (im.rows - os.y) / df + 1;
        int c = (im.cols - os.x) / df;
        cv::Mat new_im2(r, c, im.type());
        new_im = new_im2;
        for (size_t i = 0 + os.y, m = 0;
             i < (size_t)im.rows && m < (size_t)new_im.rows;
             i += df, ++m)
        {
            for (size_t j = 0 + os.x, n = 0;
                 j < (size_t)im.cols && n < (size_t)new_im.cols;
                 j += df, ++n)
            {

                if (im.channels() == 1)
                {
                    new_im.at<uchar>(m, n) = im.at<uchar>(i, j);
                }
                else
                {
                    new_im.at<cv::Vec3b>(m, n) = im.at<cv::Vec3b>(i, j);
                }
            }
        }
    }

    for (int s = 0; s < nScales; s++)
    {
        cv::Size patch_sz;
        patch_sz.width = std::max(std::floor(base_target_sz.width * scaleFactors[s]), 2.0f);
        patch_sz.height = std::max(std::floor(base_target_sz.height * scaleFactors[s]), 2.0f);
        //debug("patch_sz:%d %d", patch_sz.height, patch_sz.width);

        cv::Point pos2(pos.x - floor((patch_sz.width + 1) / 2),
                       pos.y - floor((patch_sz.height + 1) / 2));

        cv::Mat im_patch = subwindow(new_im, cv::Rect(pos2, patch_sz), IPL_BORDER_REPLICATE);

        cv::Mat im_patch_resized;
        if (im_patch.cols == 0 || im_patch.rows == 0)
        {
            return im_patch_resized;
        }
        cv::resize(im_patch, im_patch_resized, scale_model_sz);
        //printMat(im_patch);
        //showmat3channels(im_patch, 0);
        //printMat(im_patch_resized);
        //showmat3channels(im_patch_resized, 0);

        vector<cv::Mat> im_vector, temp_hog;
        im_vector.push_back(im_patch);
        FeatureExtractor feature_extractor;
#ifdef USE_SIMD
		temp_hog = feature_extractor.get_hog_features_simd(im_vector);
#else
		temp_hog = feature_extractor.get_hog_features(im_vector);
#endif
		temp_hog = feature_extractor.hog_feature_normalization(temp_hog);

        debug("Not finished!-------------------");
        assert(0);
    }

    cv::Mat scale_sample;
    return scale_sample;
}

} // namespace eco