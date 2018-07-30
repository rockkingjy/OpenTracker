/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#ifndef IMRESAMPLEMEX_HPP
#define IMRESAMPLEMEX_HPP

#include "wrappers.hpp"
#include "string.h"
#include <math.h>
#include <typeinfo>
#include "sse.hpp"
typedef unsigned char uchar;

// compute interpolation values for single column for resapling
template<class T> void resampleCoef( int ha, int hb, int &n, int *&yas,
  int *&ybs, T *&wts, int bd[2], int pad=0 );

// resample A using bilinear interpolation and and store result in B
template<class T>
void resample( T *A, T *B, int ha, int hb, int wa, int wb, int d, T r );

#endif
