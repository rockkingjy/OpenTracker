
#include "kcf/kcftracker.hpp"
//#include "goturn/network/regressor.h"
//#include "goturn/tracker/tracker.h"

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

// Convert to string
#define SSTR(x) static_cast<std::ostringstream &>(           \
                    (std::ostringstream() << std::dec << x)) \
                    .str()

int main(int argc, char **argv)
//int trackercompare()
{
    string databaseTypes[4] = {"VOT-2017", "TB-2015", "TLP", "UAV123"};
    string databaseType = databaseTypes[0];
    // Create KCFTracker:
    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //LAB color space features
    KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);

    // Create DSSTTracker:
    DSST = true;
    KCFTracker dssttracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);

    // Create Opencv tracker:
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN"};
    string trackerType = trackerTypes[2];
    Ptr<cv::Tracker> opencvtracker;

    if (trackerType == "BOOSTING")
        opencvtracker = cv::TrackerBoosting::create();
    if (trackerType == "MIL")
        opencvtracker = cv::TrackerMIL::create();
    if (trackerType == "KCF")
        opencvtracker = cv::TrackerKCF::create();
    if (trackerType == "TLD")
        opencvtracker = cv::TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
        opencvtracker = cv::TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
        opencvtracker = cv::TrackerGOTURN::create();
    /*
// Create GOTURN tracker:
    const string model_file = "goturn/nets/deploy.prototxt";
    const string pretrain_file = "goturn/nets/goturun_tracker.caffemodel";
    int gpu_id = 0;

    Regressor regressor(model_file,pretrain_file,gpu_id, false);
    goturn::Tracker goturntracker(false);
*/
    // Read from the images ====================================================

    int f, x, y, w, h, isLost;
    float x1, y1, x2, y2, x3, y3, x4, y4; //gt for vot
    std::string s;
    std::string path;
    ifstream *groundtruth;
    ostringstream osfile;
    if (databaseType == "TLP")
    {
        path = "/media/elab/sdd/data/TLP/Bike"; //Alladin";//IceSkating";//Sam";
        // Read the groundtruth bbox
        groundtruth = new ifstream(path + "/groundtruth_rect.txt");
        getline(*groundtruth, s, ',');
        f = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        x = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        y = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        w = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        h = atoi(s.c_str());
        getline(*groundtruth, s);
        isLost = atoi(s.c_str());
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << isLost << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(5) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "TB-2015")
    {
        path = "/media/elab/sdd/data/TB-2015/Coke";///Bird1";//BlurFace"; 
        // Read the groundtruth bbox
        groundtruth = new ifstream(path + "/groundtruth_rect.txt");
        f = 1;
        getline(*groundtruth, s, ',');
        x = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        y = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        w = atoi(s.c_str());
        getline(*groundtruth, s);
        h = atoi(s.c_str());
        cout << f << " " << x << " " << y << " " << w << " " << h << " "<< endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "UAV123")
    {
        string folderUAV = "bike2"; //"bike1"; //
        path = "/media/elab/sdd/data/UAV123/data_seq/UAV123/" + folderUAV;
        // Read the groundtruth bbox
        groundtruth = new ifstream("/media/elab/sdd/data/UAV123/anno/UAV123/" + folderUAV + ".txt");
        f = 1;
        getline(*groundtruth, s, ',');
        x = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        y = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        w = atoi(s.c_str());
        getline(*groundtruth, s);
        h = atoi(s.c_str());
        cout << x << " " << y << " " << w << " " << h << endl;
        // Read images in a folder
        osfile << path << "/" << setw(6) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "VOT-2017")
    {
        string folderVOT = "girl"; //"iceskater1";//"drone1"; //"iceskater2";//"helicopter";//"matrix";//"leaves";//"sheep";//"racing";//"girl";//"road"; //"uav2";//
        path = "/media/elab/sdd/data/VOT/vot2017/" + folderVOT;
        // Read the groundtruth bbox
        groundtruth = new ifstream("/media/elab/sdd/data/VOT/vot2017/" + folderVOT + "/groundtruth.txt");
        f = 1;
        getline(*groundtruth, s, ',');
        x1 = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        y1 = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        x2 = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        y2 = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        x3 = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        y3 = atoi(s.c_str());
        getline(*groundtruth, s, ',');
        x4 = atoi(s.c_str());
        getline(*groundtruth, s);
        y4 = atoi(s.c_str());
        x = (int)std::min(x1, x4);
        y = (int)std::min(y1, y2);
        w = (int)std::max(x2, x3) - x;
        h = (int)std::max(y3, y4) - y;
        cout << x << " " << y << " " << w << " " << h << endl;
        // Read images in a folder
        osfile << path << "/" << setw(8) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }

    Rect2d bboxGroundtruth(x, y, w, h);

    cv::Mat frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if (!frame.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    // Draw gt;
    if (databaseType == "TLP")
    {
        rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    }
    else if (databaseType == "TB-2015")
    {
        rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    }
    else if (databaseType == "UAV123")
    {
        rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    }
    else if (databaseType == "VOT-2017")
    {
        line(frame, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 0), 2, 1);
        line(frame, cv::Point(x2, y2), cv::Point(x3, y3), Scalar(0, 0, 0), 2, 1);
        line(frame, cv::Point(x3, y3), cv::Point(x4, y4), Scalar(0, 0, 0), 2, 1);
        line(frame, cv::Point(x4, y4), cv::Point(x1, y1), Scalar(0, 0, 0), 2, 1);
    }
    imshow("Tracking", frame);
    // Init the trackers=================================================
    Rect2d kcfbbox(x, y, w, h);
    kcftracker.init(frame, kcfbbox);

    Rect2d dsstbbox(x, y, w, h);
    dssttracker.init(frame, dsstbbox);

    Rect2d opencvbbox(x, y, w, h);
    opencvtracker->init(frame, opencvbbox);
    /*
    cv::Rect goturnbbox(x,y,w,h);
    BoundingBox bbox_gt;
    BoundingBox bbox_estimate_uncentered;
    bbox_gt.getRect(goturnbbox);
    goturntracker.Init(frame,bbox_gt,&regressor);
*/

    while (frame.data)
    {
        //KCF========================
        double timerkcf = (double)getTickCount();
        bool okkcf = kcftracker.update(frame, kcfbbox);
        float fpskcf = getTickFrequency() / ((double)getTickCount() - timerkcf);
        if (okkcf)
        {
            rectangle(frame, kcfbbox, Scalar(225, 0, 0), 2, 1); //blue
        }
        else
        {
            putText(frame, "Kcf tracking failure detected", cv::Point(100, 80), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(255, 0, 0), 2);
        }

        // Update the GOTURN tracking result--------------------------
        /*        goturntracker.Track(frame, &regressor, &bbox_estimate_uncentered);
        bbox_estimate_uncentered.putRect(goturnbbox);
        // draw goturn bbox
        rectangle(frame, goturnbbox, Scalar(0, 0, 255), 2, 1); //red
*/
        //DSST=============================
        double timerdsst = (double)getTickCount();
        bool okdsst = dssttracker.update(frame, dsstbbox);
        float fpsdsst = getTickFrequency() / ((double)getTickCount() - timerdsst);
        if (okdsst)
        {
            rectangle(frame, dsstbbox, Scalar(0, 0, 255), 2, 1); //blue
        }
        else
        {
            putText(frame, "DSST tracking failure detected", cv::Point(100, 80), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(0, 0, 255), 2);
        }
        //Opencv========================================
        double timercv = (double)getTickCount();
        bool okopencv = opencvtracker->update(frame, opencvbbox);
        float fpscv = getTickFrequency() / ((double)getTickCount() - timercv);
        if (okopencv)
        {
            rectangle(frame, opencvbbox, Scalar(0, 225, 0), 2, 1); //green
        }
        else
        {
            putText(frame, "Opencv tracking failure detected", cv::Point(100, 110), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(0, 225, 0), 2);
        }

        // Draw ground truth box===========================================
        if (databaseType == "TLP")
        {
            rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        }
        else if (databaseType == "TB-2015")
        {
            rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        }
        else if (databaseType == "UAV123")
        {
            rectangle(frame, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        }
        else if (databaseType == "VOT-2017")
        {
            line(frame, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 0), 2, 1);
            line(frame, cv::Point(x2, y2), cv::Point(x3, y3), Scalar(0, 0, 0), 2, 1);
            line(frame, cv::Point(x3, y3), cv::Point(x4, y4), Scalar(0, 0, 0), 2, 1);
            line(frame, cv::Point(x4, y4), cv::Point(x1, y1), Scalar(0, 0, 0), 2, 1);
        }

        // Display FPS on frame
        putText(frame, "FPS in total: " + SSTR(long(fpskcf)), Point(100, 50), FONT_HERSHEY_SIMPLEX,
                0.75, Scalar(0, 0, 0), 2);
        // Display tracker type on frame
        putText(frame, "Black:GT; Blue: KCF; Red: DSST; Green: opencv " + trackerType + ";",
                Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 2);
        //=============================================================================
        // Display frame.
        //cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
        imshow("Tracking", frame);

        int c = cvWaitKey(1);
        if (c != -1)
            c = c % 256;
        if (c == 27)
        {
            cvDestroyWindow("Tracking");
            return 0;
        }
        waitKey(1);
        // Read next image
        f++;
        osfile.str("");

        if (databaseType == "TLP")
        {
            // Read the groundtruth bbox
            getline(*groundtruth, s, ',');
            //f = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            x = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            y = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            w = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            h = atoi(s.c_str());
            getline(*groundtruth, s);
            isLost = atoi(s.c_str());
            //cout << "gt:" << f << " " << x << " " << y << " " << w << " " << h << " " << isLost << endl;
            osfile << path << "/img/" << setw(5) << setfill('0') << f << ".jpg";
            //cout << osfile.str() << endl;
        }
        else if (databaseType == "TB-2015")
        {
            getline(*groundtruth, s, ',');
            x = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            y = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            w = atoi(s.c_str());
            getline(*groundtruth, s);
            h = atoi(s.c_str());
            //cout << f << " " << x << " " << y << " " << w << " " << h << " " << isLost << endl;
            // Read images in a folder
            osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
            //cout << osfile.str() << endl;
        }
        else if (databaseType == "UAV123")
        {
            // Read the groundtruth bbox
            getline(*groundtruth, s, ',');
            x = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            y = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            w = atoi(s.c_str());
            getline(*groundtruth, s);
            h = atoi(s.c_str());
            //cout << "gt:" << x << " " << y << " " << w << " " << h << endl;
            // Read images in a folder
            osfile << path << "/" << setw(6) << setfill('0') << f << ".jpg";
            //cout << osfile.str() << endl;
        }
        else if (databaseType == "VOT-2017")
        {
            // Read the groundtruth bbox
            getline(*groundtruth, s, ',');
            x1 = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            y1 = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            x2 = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            y2 = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            x3 = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            y3 = atoi(s.c_str());
            getline(*groundtruth, s, ',');
            x4 = atoi(s.c_str());
            getline(*groundtruth, s);
            y4 = atoi(s.c_str());
            x = std::min(x1, x4);
            y = std::min(y1, y2);
            w = std::max(x2, x3) - x;
            h = std::max(y3, y4) - y;
            //cout << x << " " << y << " " << w << " " << h << endl;
            // Read images in a folder
            osfile << path << "/" << setw(8) << setfill('0') << f << ".jpg";
            //cout << osfile.str() << endl;
        }

        bboxGroundtruth.x = x;
        bboxGroundtruth.y = y;
        bboxGroundtruth.width = w;
        bboxGroundtruth.height = h;
        frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    }
    cvDestroyWindow("Tracking");
    return 0;


    /*
// Read from the video ====================================================
    // Read video
    VideoCapture video("videos/chaplin.mp4");
     
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;      
    }
     
    // Read first frame
    Mat frame;
    bool ok = video.read(frame);
     
    // Define initial boundibg box
    Rect2d bbox(287, 23, 86, 320);
     
    // Uncomment the line below to select a different bounding box
    bbox = selectROI(frame, false);
 
    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("Tracking", frame);
   
    while(video.read(frame))
    {     
        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
 
        // Display frame.
        imshow("Tracking", frame);
         
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
 
    }

*/
}
