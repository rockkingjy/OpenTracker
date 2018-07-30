/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#ifndef IMPADMEX_HPP
#define IMPADMEX_HPP

#include "wrappers.hpp"
#include "string.h"
typedef unsigned char uchar;

// pad A by [pt,pb,pl,pr] and store result in B
template<class T> void imPad( T *A, T *B, int h, int w, int d, int pt, int pb,
  int pl, int pr, int flag, T val );

#endif
