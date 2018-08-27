/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.30
* Copyright 2014 Piotr Dollar & Ron Appel.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#ifndef GRADIENTMEX_HPP
#define GRADIENTMEX_HPP

#include "wrappers.hpp"
#include <math.h>
#include "string.h"
#include "sse.hpp"
#include <stdlib.h>

#include <assert.h>
#include <stdio.h>

// compute x and y gradients for just one column (uses sse)
void grad1(float *I, float *Gx, float *Gy, int h, int w, int x);

// compute x and y gradients at each location (uses sse)
void grad2(float *I, float *Gx, float *Gy, int h, int w, int d);

// build lookup table a[] s.t. a[x*n]~=acos(x) for x in [-1,1]
float *acosTable();

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full);

// normalize gradient magnitude at each location (uses sse)
void gradMagNorm(float *M, float *S, int h, int w, float norm);

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1,
                  int nb, int n, float norm, int nOrients, bool full, bool interpolate);

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist(float *M, float *O, float *H, int h, int w,
              int bin, int nOrients, int softBin, bool full);

/******************************************************************************/

// HOG helper: compute 2x2 block normalization values (padded by 1 pixel)
float *hogNormMatrix(float *H, int nOrients, int hb, int wb, int bin);

// HOG helper: compute HOG or FHOG channels
void hogChannels(float *H, const float *R, const float *N,
                 int hb, int wb, int nOrients, float clip, int type);

// compute HOG features
void hog(float *M, float *O, float *H, int h, int w, int binSize,
         int nOrients, int softBin, bool full, float clip);

// compute FHOG features
void fhog(float *M, float *O, float *H, int h, int w, int binSize,
          int nOrients, int softBin, float clip);

#endif