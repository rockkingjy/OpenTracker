#ifndef MAT_ELEMENT_HPP
#define MAT_ELEMENT_HPP

#include <iostream>
#include <math.h>
#include <vector>

#include <opencv2/features2d/features2d.hpp>

namespace eco{

inline float mat_cos1(float x)
{
	return (cos(x * 3.1415926));
}

inline float mat_sin1(float x)
{
	return (sin(x * 3.1415926));
}

inline float mat_cos2(float x)
{
	return (cos(2 * x * 3.1415926));
}

inline float mat_sin2(float x)
{
	return (sin(2 * x * 3.1415926));
}

inline float mat_cos4(float x)
{
	return (cos(4 * x * 3.1415926));
}

inline float mat_sin4(float x)
{
	return (sin(4 * x * 3.1415926));
}

}
#endif