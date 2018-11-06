
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "debug.hpp"
#include "kalmantracker.hpp"

using namespace std;
using namespace cv;
using namespace sort;

int KalmanTracker::count_ = 0;

int main(int argc, char **argv)
{
    // Database settings
    string databaseTypes[3] = {"Demo", "MOT16", "2D_MOT_2015"};
    string databaseType = databaseTypes[0];
    // Read from the images ====================================================
    int f, fileNow = 1, isLost;
    float x, y, w, h, confidence, confidenceThreshhold = 0;
    vector<Rect2f> bboxGroundtruth;

    std::string s;
    std::string path;
    ifstream *groundtruth;
    ostringstream osfile;

    if (databaseType == "Demo")
    {
        path = "../sequences/Crossing";
        // some of the dataset has '\t' as the delimiter, change it to ','.
        fstream gt(path + "/groundtruth_rect.txt");
        string tmp;
        size_t index = 1;
        while (gt >> tmp)
        {
            if(tmp.find(',')<10)
            {
                break;
            }
            if (index%4 == 0)
            {
            }
            else
            {
                gt << ",";
            }
            index++;
        }
        gt.close();
        // Read the groundtruth bbox
        groundtruth = new ifstream(path + "/groundtruth_rect.txt");
        f = 1;
        getline(*groundtruth, s, ',');
        x = atof(s.c_str());
        getline(*groundtruth, s, ',');
        y = atof(s.c_str());
        getline(*groundtruth, s, ',');
        w = atof(s.c_str());
        getline(*groundtruth, s);
        h = atof(s.c_str());
        f = 1;
        bboxGroundtruth.push_back(Rect2f(x, y, w, h));
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "MOT16")
    {
        string folderMOT = "MOT16-01";
        path = "/media/elab/sdd/data/MOT16/test/" + folderMOT;
        groundtruth = new ifstream(path + "/det/det.txt");
        getline(*groundtruth, s, ',');
        f = atof(s.c_str());
        while (f == fileNow)
        {
            getline(*groundtruth, s, ',');
            getline(*groundtruth, s, ',');
            x = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y = atof(s.c_str());
            getline(*groundtruth, s, ',');
            w = atof(s.c_str());
            getline(*groundtruth, s, ',');
            h = atof(s.c_str());
            getline(*groundtruth, s, ',');
            confidence = atof(s.c_str());
            //cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
            if (confidence > confidenceThreshhold)
            {
                bboxGroundtruth.push_back(Rect2f(x, y, w, h));
            }
            getline(*groundtruth, s);
            getline(*groundtruth, s, ',');
            f = atof(s.c_str());
        }
        // Read images in a folder
        osfile << path << "/img1/" << setw(6) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "2D_MOT_2015")
    {
        string folderMOT = "PETS09-S2L1";
        path = "/media/elab/sdd/mycodes/tracker/sort/mot_benchmark/train/" + folderMOT;
        //groundtruth = new ifstream(path + "/det/det.txt");
        groundtruth = new ifstream("/media/elab/sdd/mycodes/tracker/sort/data/PETS09-S2L1/det.txt");
        getline(*groundtruth, s, ',');
        f = atof(s.c_str());
        while (f == fileNow)
        {
            getline(*groundtruth, s, ',');
            getline(*groundtruth, s, ',');
            x = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y = atof(s.c_str());
            getline(*groundtruth, s, ',');
            w = atof(s.c_str());
            getline(*groundtruth, s, ',');
            h = atof(s.c_str());
            getline(*groundtruth, s, ',');
            confidence = atof(s.c_str());
            //cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
            if (confidence > confidenceThreshhold)
            {
                bboxGroundtruth.push_back(Rect2f(x, y, w, h));
            }
            getline(*groundtruth, s);
            getline(*groundtruth, s, ',');
            f = atof(s.c_str());
        }
        // Read images in a folder
        osfile << path << "/img1/" << setw(6) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }

    cv::Mat frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if (!frame.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cv::Mat frameDraw;
    frame.copyTo(frameDraw);
    // Draw gt;
    if (databaseType == "MOT16" || databaseType == "2D_MOT_2015")
    {
        for (int i = 0; i < bboxGroundtruth.size(); i++)
        {
            cout << bboxGroundtruth[i].x << " " << bboxGroundtruth[i].y << " " << bboxGroundtruth[i].width << " " << bboxGroundtruth[i].height << endl;
            rectangle(frameDraw, bboxGroundtruth[i], Scalar(0, 0, 0), 2, 1);
        }
    }

    imshow("OpenTracker", frameDraw);
    waitKey(0);

    // Create KalmanTracker
    cv::Rect2f kalmanbbox;
    KalmanTracker kalmantracker;
    kalmantracker.init(bboxGroundtruth[0]);

    while (frame.data)
    {
        // KalmanTracker
        kalmantracker.predict();
        kalmanbbox = kalmantracker.get_state();
        debug("bbox: %f, %f, %f, %f", kalmanbbox.x, kalmanbbox.y, kalmanbbox.width, kalmanbbox.height);
        rectangle(frameDraw, kalmanbbox, Scalar(0, 255, 0), 2, 1);

        // Show the image
        imshow("OpenTracker", frameDraw);

        int c = cvWaitKey(0);
        if (c != -1)
            c = c % 256;
        if (c == 27)
        {
            cvDestroyWindow("OpenTracker");
            return 0;
        }
        waitKey(1);
        // Read next image======================================================
        fileNow++;
        osfile.str("");
        if (databaseType == "Demo")
        {
            bboxGroundtruth.clear();
            getline(*groundtruth, s, ',');
            x = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y = atof(s.c_str());
            getline(*groundtruth, s, ',');
            w = atof(s.c_str());
            getline(*groundtruth, s);
            h = atof(s.c_str());
            //cout << f << " " << x << " " << y << " " << w << " " << h << " " << isLost << endl;
            // Read images in a folder
            f++;
            bboxGroundtruth.push_back(Rect2f(x, y, w, h));
            osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
            //cout << osfile.str() << endl;
        }
        else if (databaseType == "MOT16" || databaseType == "2D_MOT_2015")
        {
            bboxGroundtruth.clear();
            while (f == fileNow)
            {
                getline(*groundtruth, s, ',');
                getline(*groundtruth, s, ',');
                x = atof(s.c_str());
                getline(*groundtruth, s, ',');
                y = atof(s.c_str());
                getline(*groundtruth, s, ',');
                w = atof(s.c_str());
                getline(*groundtruth, s, ',');
                h = atof(s.c_str());
                getline(*groundtruth, s, ',');
                confidence = atof(s.c_str());
                //cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
                if (confidence > confidenceThreshhold)
                {
                    bboxGroundtruth.push_back(Rect2f(x, y, w, h));
                }
                getline(*groundtruth, s);
                getline(*groundtruth, s, ',');
                f = atof(s.c_str());
            }
            // Read images in a folder
            osfile << path << "/img1/" << setw(6) << setfill('0') << f << ".jpg";
            cout << osfile.str() << endl;
        }

        frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if (!frame.data)
        {
            cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        frame.copyTo(frameDraw);
        // Draw gt;
        if (databaseType == "Demo" || databaseType == "MOT16" || databaseType == "2D_MOT_2015")
        {
            for (int i = 0; i < bboxGroundtruth.size(); i++)
            {
                cout << bboxGroundtruth[i].x << " " << bboxGroundtruth[i].y << " " << bboxGroundtruth[i].width << " " << bboxGroundtruth[i].height << endl;
                rectangle(frameDraw, bboxGroundtruth[i], Scalar(0, 0, 0), 2, 1);
            }
        }

        // KalmanTracker update
        kalmantracker.update(bboxGroundtruth[0]);
    }
    return 0;
}