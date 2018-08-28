#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

#include "opentracker/eco.hpp"

using namespace std;
using namespace cv;
using namespace eco;

int main(int argc, char **argv)
{
    // Read from the images
    string databaseTypes[5] = {"Demo"};
    string databaseType = databaseTypes[0];
    int f;
    float x, y, w, h;
    std::string s;
    std::string path;
    ifstream *groundtruth;
    ostringstream osfile;
    path = "../sequences/Crossing";
    fstream gt(path + "/groundtruth_rect.txt");
    string tmp;
    size_t index = 1;
    while (gt >> tmp)
    {
        if (tmp.find(',') < 10)
        {
            break;
        }
        if (index % 4 == 0)
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
    cout << "Bounding box:" << x << " " << y << " " << w << " " << h << " " << endl;
    // Read images in the folder
    osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
    cout << osfile.str() << endl;
    // Ini bounding box and frame
    Rect2f bboxGroundtruth(x, y, w, h);
    cv::Mat frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat frameDraw;
    frame.copyTo(frameDraw);
    if (!frame.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    // Draw gt;
    rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);

    // Initialize ECO tracker========================================
    ECO ecotracker;
    Rect2f ecobbox(x, y, w, h);
    float threshhold = 0.1f;
    ecotracker.init(frame, ecobbox, threshhold);
    //===============================================================    

    while (frame.data)
    {
        frame.copyTo(frameDraw); 
        
        // Update eco tracker=======================================
        bool okeco = ecotracker.update(frame, ecobbox);
        //==========================================================

        if (okeco)
        {
            rectangle(frameDraw, ecobbox, Scalar(255, 0, 255), 2, 1); //blue
        }
        else
        {
            putText(frameDraw, "ECO tracking failure detected", cv::Point(100, 80), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(255, 0, 255), 2);
        }

        // Draw ground truth box
        rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);

        // ShowImage
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

        // Read next image
        f++;
        osfile.str("");
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
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;

        bboxGroundtruth.x = x;
        bboxGroundtruth.y = y;
        bboxGroundtruth.width = w;
        bboxGroundtruth.height = h;
        frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if (!frame.data)
        {
            break;
        }
    }
    
    // If use multi_thread for train the tracker, add this to 
    // forbid thread running after programme finished error=====
#ifdef USE_MULTI_THREAD
    void *status;
    int rc = pthread_join(ecotracker.thread_train_, &status);
    if (rc)
    {
        cout << "Error:unable to join," << rc << std::endl;
        exit(-1);
    }
#endif
    //==========================================================

    return 0;
}
