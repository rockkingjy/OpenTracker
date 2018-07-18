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

namespace eco
{
cv::Mat fftf(const cv::Mat &img_org, const bool backwards = false);
cv::Mat fftd(const cv::Mat &img_org, const bool backwards = false);

cv::Mat real(const cv::Mat img);
cv::Mat imag(const cv::Mat img);
cv::Mat magnitude(const cv::Mat img);
cv::Mat complexMultiplication(const cv::Mat a, const cv::Mat b);
cv::Mat complexDivision(const cv::Mat a, const cv::Mat b);

void rearrange(cv::Mat &img);

cv::Mat fftshift(const cv::Mat org_img, 
				 const bool rowshift = true, 
				 const bool colshift = true, 
				 const bool reverse = 0);
cv::Mat fftshiftd(const cv::Mat org_img, 
				  const bool rowshift = true, 
				  const bool colshift = true, 
				  const bool reverse = 0);

cv::Mat mat_conj(const cv::Mat &org);
float mat_sum(const cv::Mat &org);
double mat_sumd(const cv::Mat &org);

cv::Mat cmat_multi(const cv::Mat &a, const cv::Mat &b); 
cv::Mat real2complex(const cv::Mat &x);
cv::Mat conv_complex(cv::Mat _a, cv::Mat _b, bool valid = 0); 

inline bool SizeCompare(cv::Size &a, cv::Size &b) 
{
	return a.height < b.height;
}

inline void rot90(cv::Mat &matImage, int rotflag)
{ //matrix rotation 1=CW, 2=CCW, 3=180

	if (rotflag == 1)
	{
		transpose(matImage, matImage);
		flip(matImage, matImage, 1); //transpose+flip(1)=CW
	}
	else if (rotflag == 2)
	{
		transpose(matImage, matImage);
		flip(matImage, matImage, 0); //transpose+flip(0)=CCW
	}
	else if (rotflag == 3)
	{
		flip(matImage, matImage, -1); //flip(-1)=180
	}
	else if (rotflag != 0) // 0: keep the same
	{ 
		assert("Unknown rotation flag");
	}
}

} // namespace eco

#endif