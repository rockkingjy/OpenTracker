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

// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.

#include "../../eco/ffttools.hpp"
#include "gtest/gtest.h"
namespace {

// Step 2. Use the TEST macro to define your tests.

// Tests factorial of negative numbers.
TEST(ffttoolsTest, mat_sum) {
  cv::Mat_<float> mat_float(10,10, CV_32FC1);
		for (int j = 0; j < mat_float.rows; j++)
      for (int i = 0; i < mat_float.cols; i++)
			  mat_float.at<float>(j, i) = i + j * mat_float.cols;

  EXPECT_EQ(4950, eco::mat_sum(mat_float));

}  

TEST(ffttoolsTest, mat_sumd) {
  cv::Mat_<double> mat_double(10,10, CV_64FC1);
		for (int j = 0; j < mat_double.rows; j++)
      for (int i = 0; i < mat_double.cols; i++)
			  mat_double.at<double>(j, i) = i + j * mat_double.cols;

  EXPECT_EQ(4950, eco::mat_sumd(mat_double));

}  

} //namespace