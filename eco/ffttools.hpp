/* 
Author: Christian Bailer
Contact address: Christian.Bailer@dfki.de 
Department Augmented Vision DFKI 

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

#ifndef FFTTOOLS_HPP
#define FFTTOOLS_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include "debug.hpp"
/*
#ifdef USE_CUDA
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#endif 
*/
#ifdef USE_SIMD
#include "sse.hpp"
#include "wrappers.hpp"
#endif

#ifdef USE_FFTW
#include <fftw3.h>
#endif

namespace eco
{
cv::Mat dft(const cv::Mat img_org, const bool backwards = false);
cv::Mat fftshift(const cv::Mat img_org,
				   const bool rowshift = true,
				   const bool colshift = true,
				   const bool reverse = 0);

cv::Mat real(const cv::Mat img);
cv::Mat imag(const cv::Mat img);
cv::Mat magnitude(const cv::Mat img);
cv::Mat complexDotMultiplication(const cv::Mat &a, const cv::Mat &b);
cv::Mat complexDotMultiplicationCPU(const cv::Mat &a, const cv::Mat &b);
#ifdef USE_SIMD
cv::Mat complexDotMultiplicationSIMD(const cv::Mat &a, const cv::Mat &b);
#endif
/*
#ifdef USE_CUDA
cv::Mat complexDotMultiplicationGPU(const cv::Mat &a, const cv::Mat &b);
#endif
*/
cv::Mat complexDotDivision(const cv::Mat a, const cv::Mat b);
cv::Mat complexMatrixMultiplication(const cv::Mat &a, const cv::Mat &b);
cv::Mat complexConvolution(const cv::Mat a_input,
						   const cv::Mat b_input,
						   const bool valid = 0);

cv::Mat real2complex(const cv::Mat &x);
cv::Mat mat_conj(const cv::Mat &org);
float mat_sum_f(const cv::Mat &org);
double mat_sum_d(const cv::Mat &org);

inline bool SizeCompare(cv::Size &a, cv::Size &b)
{
	return a.height < b.height;
}

inline void rot90(cv::Mat &matImage, int rotflag)
{
	if (rotflag == 1)
	{
		cv::transpose(matImage, matImage);
		cv::flip(matImage, matImage, 1); // flip around y-axis
	}
	else if (rotflag == 2)
	{
		cv::transpose(matImage, matImage);
		cv::flip(matImage, matImage, 0); // flip around x-axis
	}
	else if (rotflag == 3)
	{
		cv::flip(matImage, matImage, -1); // flip around both axis
	}
	else if (rotflag != 0) // 0: keep the same
	{
		assert(0 && "error: unknown rotation flag!");
	}
}

} // namespace eco

#endif