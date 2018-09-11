/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 4; tab-width: 4 -*- */
/*
 * This is an example of a stationary tracker. It only reports the initial
 * position for all frames and is used for testing purposes.
 * The main function of this example is to show the developers how to modify
 * their trackers to work with the evaluation environment.
 *
 * Copyright (c) 2015, VOT Committee
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
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "kcftracker.hpp"

#define VOT_RECTANGLE
#include "vot.h"

int main(int argc, char **argv)
{
    VOT vot;
    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //LAB color space features
    kcf::KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    cv::Rect2f initialization;
    initialization << vot.region();
    cv::Mat image = cv::imread(vot.frame());
    tracker.init(image, initialization);

    while (!vot.end())
    {
        string imagepath = vot.frame();

        if (imagepath.empty())
            break;

        cv::Mat image = cv::imread(imagepath);

        float confidence = 1;
        cv::Rect2d bbox;
        confidence = tracker.update(image, bbox);
        cv::Rect2f rect = cv::Rect(bbox);
        vot.report(rect, confidence);
    }
}
