// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// A sample program demonstrating using Google C++ testing framework.
//
// Author: wan@google.com (Zhanyong Wan)

#include "gtest/gtest.h"
#include "../../eco/ffttools.hpp"
#include "debug.hpp"

// Frames:119 AvgPrecision:1 AvgIou:0.677676 SuccessRate:0.932773 AvgFps:10.3517

namespace
{

TEST(ffttoolsTest, dft_f)
{
  cv::Mat_<float> mat_float(10, 10, CV_32FC1);
  for (int j = 0; j < mat_float.rows; j++)
    for (int i = 0; i < mat_float.cols; i++)
      mat_float.at<float>(j, i) = i + j * mat_float.cols;

	double timer = (double)cv::getTickCount();
	float timedft = 0;

  //showmat1channels(mat_float, 2);
  cv::Mat res = eco::dft_f(mat_float);
  //showmat2channels(res, 2); 

  timedft = ((double)cv::getTickCount() - timer) / cv::getTickFrequency();
	debug("dft_f time: %f", timedft);

  res = eco::dft_f(mat_float, 1);
  //showmat2channels(res, 2); 

  timedft = ((double)cv::getTickCount() - timer) / cv::getTickFrequency();
	debug("dft_f time: %f", timedft);

  res = eco::dft_f(mat_float, 1);
  //showmat2channels(res, 2); 

  timedft = ((double)cv::getTickCount() - timer) / cv::getTickFrequency();
	debug("dft_f time: %f", timedft);
}

TEST(ffttoolsTest, dft_d)
{
  cv::Mat_<double> mat_double(10, 10, CV_64FC1);
  for (int j = 0; j < mat_double.rows; j++)
    for (int i = 0; i < mat_double.cols; i++)
      mat_double.at<double>(j, i) = i + j * mat_double.cols;
  
	double timer = (double)cv::getTickCount();
	float timedft = 0;

  //showmat1channels(mat_double, 3);
  cv::Mat res = eco::dft_d(mat_double);
  //showmat2channels(res, 3); 
  res = eco::dft_d(mat_double, 1);
  //showmat2channels(res, 3); 
	
  timedft = ((double)cv::getTickCount() - timer) / cv::getTickFrequency();
	debug("dft_d time: %f", timedft);

  cv::Mat_<double> mat_double2(10, 10, CV_64FC1);
  for (int j = 0; j < mat_double2.rows; j++)
    for (int i = 0; i < mat_double2.cols; i++)
      mat_double2.at<double>(j, i) = i + j * mat_double2.cols;
  res = eco::dft_d(mat_double2, 1);

  timedft = ((double)cv::getTickCount() - timer) / cv::getTickFrequency();
	debug("dft_d time: %f", timedft);
}

TEST(ffttoolsTest, fftshift_f)
{
  cv::Mat_<float> mat_float(10, 10, CV_32FC1);
  for (int j = 0; j < mat_float.rows; j++)
    for (int i = 0; i < mat_float.cols; i++)
      mat_float.at<float>(j, i) = i + j * mat_float.cols;
  debug("channels: %d", mat_float.channels());

  showmat1channels(mat_float, 2);
  cv::Mat res;
  res = eco::fftshift_f(mat_float);
  showmat1channels(res, 2); 

  res = eco::dft_f(mat_float, 1);
  showmat2channels(res, 2); 

  res = eco::fftshift_f(res);
  showmat2channels(res, 2); 
}

TEST(ffttoolsTest, fftshift_d)
{
  cv::Mat_<double> mat_double(10, 10, CV_32FC1);
  for (int j = 0; j < mat_double.rows; j++)
    for (int i = 0; i < mat_double.cols; i++)
      mat_double.at<double>(j, i) = i + j * mat_double.cols;
  debug("channels: %d", mat_double.channels());

  showmat1channels(mat_double, 3);
  cv::Mat res;
  res = eco::fftshift_d(mat_double);
  showmat1channels(res, 3); 

  res = eco::dft_d(mat_double, 1);
  showmat2channels(res, 3); 

  res = eco::fftshift_d(res);
  showmat2channels(res, 3); 
}

TEST(ffttoolsTest, complexDotMultiplication)
{
  cv::Mat mat_float(10, 10, CV_32FC2);
  for (int j = 0; j < mat_float.rows; j++)
    for (int i = 0; i < mat_float.cols; i++)
    {
      mat_float.at<cv::Vec2f>(j, i)[0] = i + j * mat_float.cols;
      mat_float.at<cv::Vec2f>(j, i)[1] = i + j * mat_float.cols;
    }
  debug("channels: %d", mat_float.channels());
  showmat2channels(mat_float, 2);

  cv::Mat mat_float1(10, 10, CV_32FC2);
  for (int j = 0; j < mat_float1.rows; j++)
    for (int i = 0; i < mat_float1.cols; i++)
    {
      mat_float1.at<cv::Vec2f>(j, i)[0] = i + j * mat_float1.cols;
      mat_float1.at<cv::Vec2f>(j, i)[1] = -i;
    }
  debug("channels: %d", mat_float1.channels());
  showmat2channels(mat_float1, 2);

  cv::Mat res;
  res = eco::complexDotMultiplication(mat_float, mat_float1);
  showmat2channels(res, 2);
}

TEST(ffttoolsTest, complexDotDivision)
{
  cv::Mat mat_float(10, 10, CV_32FC2);
  for (int j = 0; j < mat_float.rows; j++)
    for (int i = 0; i < mat_float.cols; i++)
    {
      mat_float.at<cv::Vec2f>(j, i)[0] = i + j * mat_float.cols;
      mat_float.at<cv::Vec2f>(j, i)[1] = i + j * mat_float.cols;
    }
  debug("channels: %d", mat_float.channels());
  showmat2channels(mat_float, 2);

  cv::Mat mat_float1(10, 10, CV_32FC2);
  for (int j = 0; j < mat_float1.rows; j++)
    for (int i = 0; i < mat_float1.cols; i++)
    {
      mat_float1.at<cv::Vec2f>(j, i)[0] = i + j * mat_float1.cols;
      mat_float1.at<cv::Vec2f>(j, i)[1] = -i;
    }
  debug("channels: %d", mat_float1.channels());
  showmat2channels(mat_float1, 2);

  cv::Mat res;
  res = eco::complexDotDivision(mat_float, mat_float1);
  showmat2channels(res, 2);
}

TEST(ffttoolsTest, complexMatrixMultiplication)
{
  cv::Mat mat_float(10, 10, CV_32FC2);
  for (int j = 0; j < mat_float.rows; j++)
    for (int i = 0; i < mat_float.cols; i++)
    {
      mat_float.at<cv::Vec2f>(j, i)[0] = i + j * mat_float.cols;
      mat_float.at<cv::Vec2f>(j, i)[1] = i + j * mat_float.cols;
    }
  debug("channels: %d", mat_float.channels());
  showmat2channels(mat_float, 2);

  cv::Mat mat_float1(10, 10, CV_32FC2);
  for (int j = 0; j < mat_float1.rows; j++)
    for (int i = 0; i < mat_float1.cols; i++)
    {
      mat_float1.at<cv::Vec2f>(j, i)[0] = i + j * mat_float1.cols;
      mat_float1.at<cv::Vec2f>(j, i)[1] = -i;
    }
  debug("channels: %d", mat_float1.channels());
  showmat2channels(mat_float1, 2);

  cv::Mat res;
  res = eco::complexMatrixMultiplication(mat_float, mat_float1);
  showmat2channels(res, 2);
}

TEST(ffttoolsTest, mat_sum_f)
{
  cv::Mat_<float> mat_float(10, 10, CV_32FC1);
  for (int j = 0; j < mat_float.rows; j++)
    for (int i = 0; i < mat_float.cols; i++)
      mat_float.at<float>(j, i) = i + j * mat_float.cols;

  EXPECT_EQ(4950, eco::mat_sum_f(mat_float));
}

TEST(ffttoolsTest, mat_sum_d)
{
  cv::Mat_<double> mat_double(10, 10, CV_64FC1);
  for (int j = 0; j < mat_double.rows; j++)
    for (int i = 0; i < mat_double.cols; i++)
      mat_double.at<double>(j, i) = i + j * mat_double.cols;

  EXPECT_EQ(4950, eco::mat_sum_d(mat_double));
}

TEST(ffttoolsTest, rot90)
{
  cv::Mat_<float> mat_float(10, 10, CV_32FC1);
  for (int j = 0; j < mat_float.rows; j++)
    for (int i = 0; i < mat_float.cols; i++)
      mat_float.at<float>(j, i) = i + j * mat_float.cols;

  showmat1channels(mat_float, 2);
  debug("==============");
  eco::rot90(mat_float, 1);
  showmat1channels(mat_float, 2);
  debug("==============");
  eco::rot90(mat_float, 2);
  showmat1channels(mat_float, 2);
  debug("==============");
  eco::rot90(mat_float, 3);
  showmat1channels(mat_float, 2);
}

TEST(ffttoolsTest, complexConvolution)
{
  cv::Mat mat_float(14, 14, CV_32FC2);
  for (int j = 0; j < mat_float.rows; j++)
    for (int i = 0; i < mat_float.cols; i++)
    {
      mat_float.at<cv::Vec2f>(j, i)[0] = i + j * mat_float.cols;
      mat_float.at<cv::Vec2f>(j, i)[1] = i + j * mat_float.cols;
    }
  debug("channels: %d", mat_float.channels());
  showmat2channels(mat_float, 2);

  cv::Mat mat_float1(9, 9, CV_32FC2);
  for (int j = 0; j < mat_float1.rows; j++)
    for (int i = 0; i < mat_float1.cols; i++)
    {
      mat_float1.at<cv::Vec2f>(j, i)[0] = i + j * mat_float1.cols;
      mat_float1.at<cv::Vec2f>(j, i)[1] = -i;
    }
  debug("channels: %d", mat_float1.channels());
  showmat2channels(mat_float1, 2);

  cv::Mat res;
  res = eco::complexConvolution(mat_float, mat_float1, 0);
  showmat2channels(res, 2);
  res = eco::complexConvolution(mat_float, mat_float1, 1);
  showmat2channels(res, 2);
}

TEST(debug, copyTo_clone_Difference)
{
  copyTo_clone_Difference();
}

} //namespace