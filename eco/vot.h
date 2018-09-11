/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4; tab-width: 4 -*- */
/*
 * This header file contains C functions that can be used to quickly integrate
 * VOT challenge support into your C or C++ tracker.
 *
 * Copyright (c) 2017, VOT Committee
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */

#ifndef _VOT_TOOLKIT_H
#define _VOT_TOOLKIT_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <trax.h>

#define VOT_READ_BUFFER 2024

// Define VOT_OPENCV after including OpenCV core header to enable better OpenCV support
#if defined(__OPENCV_CORE_HPP__) || defined(OPENCV_CORE_HPP)
#define VOT_OPENCV
#endif

#ifndef VOT_RECTANGLE
#define VOT_POLYGON
#endif

#ifdef VOT_POLYGON
typedef struct vot_region
{
    float *x;
    float *y;
    int count;
} vot_region;

void vot_region_release(vot_region **region)
{
    if (!(*region))
        return;

    if ((*region)->x)
    {
        free((*region)->x);
        (*region)->x = NULL;
    }
    if ((*region)->y)
    {
        free((*region)->y);
        (*region)->y = NULL;
    }

    free(*region);

    *region = NULL;
}

vot_region *vot_region_create(int n)
{
    vot_region *region = (vot_region *)malloc(sizeof(vot_region));
    region->x = (float *)malloc(sizeof(float) * n);
    region->y = (float *)malloc(sizeof(float) * n);
    memset(region->x, 0, sizeof(float) * n);
    memset(region->y, 0, sizeof(float) * n);
    region->count = n;
    return region;
}

vot_region *vot_region_copy(const vot_region *region)
{
    vot_region *copy = vot_region_create(region->count);
    int i;
    for (i = 0; i < region->count; i++)
    {
        copy->x[i] = region->x[i];
        copy->y[i] = region->y[i];
    }
    return copy;
}

#else
typedef struct vot_region
{
    float x;
    float y;
    float width;
    float height;
} vot_region;

void vot_region_release(vot_region **region)
{

    if (!(*region))
        return;

    free(*region);

    *region = NULL;
}

vot_region *vot_region_create()
{
    vot_region *region = (vot_region *)malloc(sizeof(vot_region));
    region->x = 0;
    region->y = 0;
    region->width = 0;
    region->height = 0;
    return region;
}

vot_region *vot_region_copy(const vot_region *region)
{
    vot_region *copy = vot_region_create();
    copy->x = region->x;
    copy->y = region->y;
    copy->width = region->width;
    copy->height = region->height;
    return copy;
}

#endif

#ifdef __cplusplus

#include <string>
#include <fstream>
#include <iostream>

using namespace std;

class VOT;

class VOTRegion
{
    friend class VOT;

  public:
    ~VOTRegion()
    {
        vot_region_release(&_region);
    }

    VOTRegion(const vot_region *region)
    {
        _region = vot_region_copy(region);
    }

#ifdef VOT_POLYGON
    VOTRegion(int count)
    {
        _region = vot_region_create(count);
    }

    void set(int i, float x, float y)
    {
        assert(i >= 0 && i < _region->count);
        _region->x[i] = x;
        _region->y[i] = y;
    }
    float get_x(int i) const
    {
        assert(i >= 0 && i < _region->count);
        return _region->x[i];
    }
    float get_y(int i) const
    {
        assert(i >= 0 && i < _region->count);
        return _region->y[i];
    }
    int count() const { return _region->count; }

#else

    VOTRegion()
    {
        _region = vot_region_create();
    }

    float get_x() const { return _region->x; }
    float get_y() const { return _region->y; }
    float get_width() const { return _region->width; }
    float get_height() const { return _region->height; }

    float set_x(float x) { return _region->x = x; }
    float set_y(float y) { return _region->y = y; }
    float set_width(float width) { return _region->width = width; }
    float set_height(float height) { return _region->height = height; }

#endif

    VOTRegion &operator=(const VOTRegion &source)
    {

        if (this == &source)
            return *this;

#ifdef VOT_POLYGON

        if (this->_region->count != source.count())
        {
            vot_region_release(&(this->_region));
            this->_region = vot_region_create(source.count());
        }

        for (int i = 0; i < source.count(); i++)
        {
            set(i, source.get_x(i), source.get_y(i));
        }

#else

        set_x(source.get_x());
        set_y(source.get_y());
        set_width(source.get_width());
        set_height(source.get_height());

#endif

        return *this;
    }

#ifdef VOT_OPENCV

    VOTRegion(const cv::Rect2f &rectangle)
    {
#ifdef VOT_POLYGON
        _region = vot_region_create(4);
#else
        _region = vot_region_create();
#endif
        set(rectangle);
    }

    void set(const cv::Rect2f &rectangle)
    {

#ifdef VOT_POLYGON

        if (_region->count != 4)
        {
            vot_region_release(&(this->_region));
            _region = vot_region_create(4);
        }

        set(0, rectangle.x, rectangle.y);
        set(1, rectangle.x + rectangle.width, rectangle.y);
        set(2, rectangle.x + rectangle.width, rectangle.y + rectangle.height);
        set(3, rectangle.x, rectangle.y + rectangle.height);

#else

        set_x(rectangle.x);
        set_y(rectangle.y);
        set_width(rectangle.width);
        set_height(rectangle.height);

#endif
    }

    void get(cv::Rect2f &rectangle) const
    {

#ifdef VOT_POLYGON

        float top = FLT_MAX;
        float bottom = FLT_MIN;
        float left = FLT_MAX;
        float right = FLT_MIN;

        for (int j = 0; j < _region->count; j++)
        {
            top = MIN(top, _region->y[j]);
            bottom = MAX(bottom, _region->y[j]);
            left = MIN(left, _region->x[j]);
            right = MAX(right, _region->x[j]);
        }

        rectangle.x = left;
        rectangle.y = top;
        rectangle.width = right - left;
        rectangle.height = bottom - top;
#else

        rectangle.x = get_x();
        rectangle.y = get_y();
        rectangle.width = get_width();
        rectangle.height = get_height();

#endif
    }

    void operator=(cv::Rect2f &rectangle)
    {
        this->get(rectangle);
    }

#endif

  protected:
    vot_region *_region;
};

#ifdef VOT_OPENCV

void operator<<(VOTRegion &source, const cv::Rect2f &rectangle)
{
    source.set(rectangle);
}

void operator>>(const VOTRegion &source, cv::Rect2f &rectangle)
{
    source.get(rectangle);
}

void operator<<(cv::Rect2f &rectangle, const VOTRegion &source)
{
    source.get(rectangle);
}

void operator>>(const cv::Rect2f &rectangle, VOTRegion &source)
{
    source.set(rectangle);
}

#endif

class VOT
{
  public:
    VOT()
    {
        _region = vot_initialize();
    }

    ~VOT()
    {
        vot_quit();
    }

    const VOTRegion region()
    {
        return VOTRegion(_region);
    }

    void report(const VOTRegion &region, float confidence = 1)
    {

        vot_report2(region._region, confidence);
    }

    const string frame()
    {

        const char *result = vot_frame();

        if (!result)
            return string();

        return string(result);
    }

    bool end()
    {
        return vot_end() != 0;
    }

  private:
    vot_region *vot_initialize();

    void vot_quit();

    const char *vot_frame();

    void vot_report(vot_region *region);

    void vot_report2(vot_region *region, float confidence);

    int vot_end();

    vot_region *_region;

#endif

    // Current position in the sequence
    int _vot_sequence_position;
    // Size of the sequence
    int _vot_sequence_size;
    // List of image file names
    char **_vot_sequence;
    // List of results
    vot_region **_vot_result;

    trax_handle *_trax_handle;
    char _trax_image_buffer[VOT_READ_BUFFER];

#ifdef VOT_POLYGON

    vot_region *_trax_to_region(const trax_region *_trax_region)
    {
        int i;
        int count = trax_region_get_polygon_count(_trax_region);
        vot_region *region = vot_region_create(count);
        for (i = 0; i < count; i++)
            trax_region_get_polygon_point(_trax_region, i, &(region->x[i]), &(region->y[i]));
        return region;
    }
    trax_region *_region_to_trax(const vot_region *region)
    {
        int i;
        trax_region *_trax_region = trax_region_create_polygon(region->count);
        assert(trax_region_get_type(_trax_region) == TRAX_REGION_POLYGON);
        for (i = 0; i < region->count; i++)
            trax_region_set_polygon_point(_trax_region, i, region->x[i], region->y[i]);
        return _trax_region;
    }
#else

vot_region *_trax_to_region(const trax_region *_trax_region)
{
    vot_region *region = vot_region_create();
    assert(trax_region_get_type(_trax_region) == TRAX_REGION_RECTANGLE);
    trax_region_get_rectangle(_trax_region, &(region->x), &(region->y), &(region->width), &(region->height));
    return region;
}
trax_region *_region_to_trax(const vot_region *region)
{
    return trax_region_create_rectangle(region->x, region->y, region->width, region->height);
}

#endif

#ifdef __cplusplus
};

#endif

#ifdef __cplusplus
#define VOT_PREFIX(FUN) VOT::FUN
#else
#define VOT_PREFIX(FUN) FUN
#endif

/**
 * Reads the input data and initializes all structures. Returns the initial
 * position of the object as specified in the input data. This function should
 * be called at the beginning of the program.
 */
vot_region *VOT_PREFIX(vot_initialize)()
{

    //int j;
    //FILE *inputfile;
    // FILE *imagesfile;

    _vot_sequence_position = 0;
    _vot_sequence_size = 0;

    //trax_configuration config;
    trax_image *_trax_image = NULL;
    trax_region *_trax_region = NULL;
    _trax_handle = NULL;
    int response;
#ifdef VOT_POLYGON
    int region_format = TRAX_REGION_POLYGON;
#else
    int region_format = TRAX_REGION_RECTANGLE;
#endif

    trax_metadata *metadata = trax_metadata_create(region_format, TRAX_IMAGE_PATH, NULL, NULL, NULL);

    _trax_handle = trax_server_setup(metadata, trax_no_log);

    trax_metadata_release(&metadata);

    response = trax_server_wait(_trax_handle, &_trax_image, &_trax_region, NULL);

    assert(response == TRAX_INITIALIZE);

    strcpy(_trax_image_buffer, trax_image_get_path(_trax_image));

    trax_server_reply(_trax_handle, _trax_region, NULL);

    vot_region *region = _trax_to_region(_trax_region);

    trax_region_release(&_trax_region);
    trax_image_release(&_trax_image);

    return region;
}

/**
 * Stores results to the result file and frees memory. This function should be
 * called at the end of the tracking program.
 */
void VOT_PREFIX(vot_quit)()
{

    if (_trax_handle)
    {
        trax_cleanup(&_trax_handle);
        return;
    }
}

/**
 * Returns the file name of the current frame. This function does not advance
 * the current position.
 */
const char *VOT_PREFIX(vot_frame)()
{

    if (_trax_handle)
    {
        int response;
        trax_image *_trax_image = NULL;
        trax_region *_trax_region = NULL;

        if (_vot_sequence_position == 0)
        {
            _vot_sequence_position++;
            return _trax_image_buffer;
        }

        response = trax_server_wait(_trax_handle, &_trax_image, &_trax_region, NULL);

        if (response != TRAX_FRAME)
        {
            vot_quit();
            exit(0);
        }

        strcpy(_trax_image_buffer, trax_image_get_path(_trax_image));
        trax_image_release(&_trax_image);

        return _trax_image_buffer;
    }
}

/**
 * Used to report position of the object. This function also advances the
 * current position.
 */
void VOT_PREFIX(vot_report)(vot_region *region)
{

    if (_trax_handle)
    {
        trax_region *_trax_region = _region_to_trax(region);
        trax_server_reply(_trax_handle, _trax_region, NULL);
        trax_region_release(&_trax_region);
        return;
    }
}

/**
 * Used to report position of the object. This function also advances the
 * current position.
 */
void VOT_PREFIX(vot_report2)(vot_region *region, float confidence)
{

    if (_trax_handle)
    {
        trax_region *_trax_region = _region_to_trax(region);
        trax_properties *_trax_properties = trax_properties_create();
        trax_properties_set_float(_trax_properties, "confidence", confidence);
        trax_server_reply(_trax_handle, _trax_region, _trax_properties);
        trax_region_release(&_trax_region);
        trax_properties_release(&_trax_properties);
        return;
    }
}

int VOT_PREFIX(vot_end)()
{

    return 0;
}

#endif
