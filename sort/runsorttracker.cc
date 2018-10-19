
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
    string databaseType = databaseTypes[2];
    // Read from the images ====================================================
    int f, fileNow = 1, isLost;
    float x, y, w, h, confidence, confidenceThreshhold = 0;
    vector<Rect2f> bboxGroundtruth;

    std::string s;
    std::string path;
    ifstream *groundtruth;
    ostringstream osfile;

    if (databaseType == "MOT16")
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
    /*
    // Create KCFTracker:
    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //LAB color space features
    KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    Rect2d kcfbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
    kcftracker.init(frame, kcfbbox);
*/
    while (frame.data)
    {
        /*
        //KCF=========================
        double timerkcf = (double)getTickCount();
        bool okkcf = kcftracker.update(frame, kcfbbox);
        float fpskcf = getTickFrequency() / ((double)getTickCount() - timerkcf);
        if (okkcf)
        {
            rectangle(frameDraw, kcfbbox, Scalar(0, 255, 0), 2, 1);
        }
        else
        {
            putText(frameDraw, "Kcf tracking failure detected", cv::Point(10, 80), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(0, 255, 0), 2);
        }

        // Display FPS on frameDraw
        ostringstream os; 
        os << float(fpskcf); 
        putText(frameDraw, "FPS: " + os.str(), Point(100, 30), FONT_HERSHEY_SIMPLEX,
                0.75, Scalar(0, 225, 0), 2);
*/
        // Show the image
        imshow("OpenTracker", frameDraw);

        int c = cvWaitKey(1);
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
        if (databaseType == "MOT16" || databaseType == "2D_MOT_2015")
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
        if (databaseType == "MOT16" || databaseType == "2D_MOT_2015")
        {
            for (int i = 0; i < bboxGroundtruth.size(); i++)
            {
                cout << bboxGroundtruth[i].x << " " << bboxGroundtruth[i].y << " " << bboxGroundtruth[i].width << " " << bboxGroundtruth[i].height << endl;
                rectangle(frameDraw, bboxGroundtruth[i], Scalar(0, 0, 0), 2, 1);
            }
        }
    }
    return 0;
}