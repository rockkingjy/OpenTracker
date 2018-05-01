//
// Created by ise on 16-10-23.
//

#ifndef MYGOTURN_LOADER_BASE_H
#define MYGOTURN_LOADER_BASE_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include "../helper/helper.h"

using std::string;
using std::vector;

struct Video{
    string video_name_;
    vector<string> all_frames_;
};

class Loader{

public:
    Loader(const string& _videos_folder);
    ~Loader(){};


    void loadervideos(const string _videos_folder);
    void loadFirstBox(const Video& video,cv::Mat& image,cv::Rect& firstBox);

    static void mouseHandler(int event, int x, int y, int flag, void *userdata);


    const string videos_folder_;
    vector<Video> videos_;
};






#endif //MYGOTURN_LOADER_BASE_H
