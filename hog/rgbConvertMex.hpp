/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.22
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#ifndef RGBCONVERTMEX_HPP
#define RGBCONVERTMEX_HPP

#include "wrappers.hpp"
#include <cmath>
#include <typeinfo>
#include "sse.hpp"

// Constants for rgb2luv conversion and lookup table for y-> l conversion
template<class oT> oT* rgb2luv_setup( oT z, oT *mr, oT *mg, oT *mb,
  oT &minu, oT &minv, oT &un, oT &vn );

// Convert from rgb to luv
template<class iT, class oT> void rgb2luv( iT *I, oT *J, int n, oT nrm );

// Convert from rgb to luv using sse
template<class iT> void rgb2luv_sse( iT *I, float *J, int n, float nrm );

// Convert from rgb to hsv
template<class iT, class oT> void rgb2hsv( iT *I, oT *J, int n, oT nrm );

// Convert from rgb to gray
template<class iT, class oT> void rgb2gray( iT *I, oT *J, int n, oT nrm );

// Convert from rgb (double) to gray (float)
template<> void rgb2gray( double *I, float *J, int n, float nrm );

// Copy and normalize only
template<class iT, class oT> void normalize( iT *I, oT *J, int n, oT nrm );

// Convert rgb to various colorspaces
template<class iT, class oT>
oT* rgbConvert( iT *I, int n, int d, int flag, oT nrm );

#endif
