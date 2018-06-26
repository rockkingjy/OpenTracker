
#include "kcf/kcftracker.hpp"
#include "goturn/network/regressor.h"
#include "goturn/tracker/tracker.h"
#include "eco/eco.h"

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
	::google::InitGoogleLogging(argv[0]);
    // Database settings
    string databaseTypes[4] = {"VOT-2017", "TB-2015", "TLP", "UAV123"};
    string databaseType = databaseTypes[0];

    // Read from the images ====================================================
    int f, isLost;
    float x, y, w, h;
    float x1, y1, x2, y2, x3, y3, x4, y4; //gt for vot
    std::string s;
    std::string path;
    ifstream *groundtruth;
    ostringstream osfile;
    if (databaseType == "VOT-2017")
    {
        string folderVOT = "graduate";//"glove";//"drone1"; //"iceskater1";//"girl"; //"road";//"iceskater1";//"helicopter";//"matrix";//"leaves";//"sheep";//"racing";//"girl";//"road"; //"uav2";//
        path = "/media/elab/sdd/data/VOT/vot2017/" + folderVOT;
        // Read the groundtruth bbox
        groundtruth = new ifstream("/media/elab/sdd/data/VOT/vot2017/" + folderVOT + "/groundtruth.txt");
        f = 1;
        getline(*groundtruth, s, ',');
        x1 = atof(s.c_str());
        getline(*groundtruth, s, ',');
        y1 = atof(s.c_str());
        getline(*groundtruth, s, ',');
        x2 = atof(s.c_str());
        getline(*groundtruth, s, ',');
        y2 = atof(s.c_str());
        getline(*groundtruth, s, ',');
        x3 = atof(s.c_str());
        getline(*groundtruth, s, ',');
        y3 = atof(s.c_str());
        getline(*groundtruth, s, ',');
        x4 = atof(s.c_str());
        getline(*groundtruth, s);
        y4 = atof(s.c_str());
        x = std::min(x1, x4);
        y = std::min(y1, y2);
        w = std::max(x2, x3) - x;
        h = std::max(y3, y4) - y;
        cout << x << " " << y << " " << w << " " << h << endl;
        // Read images in a folder
        osfile << path << "/" << setw(8) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "TB-2015")
    {
        path = "/media/elab/sdd/data/TB-2015/Coke";///Bird1";//BlurFace"; 
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
        cout << f << " " << x << " " << y << " " << w << " " << h << " "<< endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "TLP")
    {
        path = "/media/elab/sdd/data/TLP/Sam";//IceSkating";//Drone3";//
        // Read the groundtruth bbox
        groundtruth = new ifstream(path + "/groundtruth_rect.txt");
        getline(*groundtruth, s, ',');
        f = atof(s.c_str());
        getline(*groundtruth, s, ',');
        x = atof(s.c_str());
        getline(*groundtruth, s, ',');
        y = atof(s.c_str());
        getline(*groundtruth, s, ',');
        w = atof(s.c_str());
        getline(*groundtruth, s, ',');
        h = atof(s.c_str());
        getline(*groundtruth, s);
        isLost = atof(s.c_str());
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << isLost << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(5) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "UAV123")
    {
        string folderUAV = "wakeboard1";//"person23";//"bike1"; //"person16"; //"person21";//
        path = "/media/elab/sdd/data/UAV123/data_seq/UAV123/" + folderUAV;
        // Read the groundtruth bbox
        groundtruth = new ifstream("/media/elab/sdd/data/UAV123/anno/UAV123/" + folderUAV + ".txt");
        f = 1;
        getline(*groundtruth, s, ',');
        x = atof(s.c_str());
        getline(*groundtruth, s, ',');
        y = atof(s.c_str());
        getline(*groundtruth, s, ',');
        w = atof(s.c_str());
        getline(*groundtruth, s);
        h = atof(s.c_str());
        cout << x << " " << y << " " << w << " " << h << endl;
        // Read images in a folder
        osfile << path << "/" << setw(6) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }

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
    if (databaseType == "TLP")
    {
        rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    }
    else if (databaseType == "TB-2015")
    {
        rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    }
    else if (databaseType == "UAV123")
    {
        rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    }
    else if (databaseType == "VOT-2017")
    {
        line(frameDraw, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 0), 2, 1);
        line(frameDraw, cv::Point(x2, y2), cv::Point(x3, y3), Scalar(0, 0, 0), 2, 1);
        line(frameDraw, cv::Point(x3, y3), cv::Point(x4, y4), Scalar(0, 0, 0), 2, 1);
        line(frameDraw, cv::Point(x4, y4), cv::Point(x1, y1), Scalar(0, 0, 0), 2, 1);
    }
    // Init the trackers=================================================
    
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
    
    Rect2d opencvbbox((int)x, (int)y, (int)w, (int)h);
    opencvtracker->init(frame, opencvbbox);
    // Create KCFTracker:
    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //LAB color space features
    KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    Rect2d kcfbbox((int)x, (int)y, (int)w, (int)h);
    kcftracker.init(frame, kcfbbox);
    // Create DSSTTracker:
    DSST = true;
    KCFTracker dssttracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    Rect2d dsstbbox((int)x, (int)y, (int)w, (int)h);
    dssttracker.init(frame, dsstbbox);    
    // Create GOTURN tracker:
    const string model_file = "goturn/nets/deploy.prototxt";
    const string pretrain_file = "goturn/nets/goturun_tracker.caffemodel";
    int gpu_id = 0;
    Regressor regressor(model_file,pretrain_file,gpu_id, false);
    goturn::Tracker goturntracker(false);
    cv::Rect goturnbbox(x,y,w,h);
    BoundingBox bbox_gt;
    BoundingBox bbox_estimate_uncentered;
    bbox_gt.getRect(goturnbbox);
    goturntracker.Init(frame,bbox_gt,&regressor);
    
    // Create ECO trakcer;
    eco::ECO ecotracker;
    Rect2f ecobbox(x, y, w, h);
    ecotracker.init(frame, ecobbox);

    //imshow("Tracking", frame);
    //waitKey(0);
    while (frame.data)
    {   
        frame.copyTo(frameDraw);
        
        //Opencv=====================
        double timercv = (double)getTickCount();
        bool okopencv = opencvtracker->update(frame, opencvbbox);
        float fpscv = getTickFrequency() / ((double)getTickCount() - timercv);
        if (okopencv)
        {
            rectangle(frameDraw, opencvbbox, Scalar(255, 0, 0), 2, 1); 
        }
        else
        {
            putText(frameDraw, "Opencv tracking failure detected", cv::Point(10, 50), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(255, 0, 0), 2);
        }
        
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
        
        //DSST========================
        double timerdsst = (double)getTickCount();
        bool okdsst = dssttracker.update(frame, dsstbbox);
        float fpsdsst = getTickFrequency() / ((double)getTickCount() - timerdsst);
        if (okdsst)
        {
            rectangle(frameDraw, dsstbbox, Scalar(0, 0, 255), 2, 1); 
        }
        else
        {
            putText(frameDraw, "DSST tracking failure detected", cv::Point(10, 100), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(0, 0, 255), 2);
        }
        
        //GOTURN=====================
        double timergoturn = (double)getTickCount();
        goturntracker.Track(frame, &regressor, &bbox_estimate_uncentered);
        bbox_estimate_uncentered.putRect(goturnbbox);
        float fpsgoturn = getTickFrequency() / ((double)getTickCount() - timergoturn);
        rectangle(frameDraw, goturnbbox, Scalar(255, 255, 0), 2, 1); 
        
        //ECO========================
        double timeeco = (double)getTickCount();
        bool okeco = ecotracker.update(frame, ecobbox);
        float fpseco = getTickFrequency() / ((double)getTickCount() - timeeco);
        if (okeco)
        {
            rectangle(frameDraw, ecobbox, Scalar(255, 0, 255), 2, 1); 
        }
        else
        {
            putText(frameDraw, "ECO tracking failure detected", cv::Point(10, 30), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(255, 0, 255), 2);
        } 
        // Draw ground truth box===========================================
        if (databaseType == "TLP")
        {
            rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        }
        else if (databaseType == "TB-2015")
        {
            rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        }
        else if (databaseType == "UAV123")
        {
            rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        }
        else if (databaseType == "VOT-2017")
        {
            line(frameDraw, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 0), 2, 1);
            line(frameDraw, cv::Point(x2, y2), cv::Point(x3, y3), Scalar(0, 0, 0), 2, 1);
            line(frameDraw, cv::Point(x3, y3), cv::Point(x4, y4), Scalar(0, 0, 0), 2, 1);
            line(frameDraw, cv::Point(x4, y4), cv::Point(x1, y1), Scalar(0, 0, 0), 2, 1);
        }

        // Display FPS on frameDraw
        putText(frameDraw, "FPS: " + SSTR(long(fpseco)), Point(10, 30), FONT_HERSHEY_SIMPLEX,
                0.75, Scalar(255, 0, 255), 2);
        
        // Draw the label of trackers
        putText(frameDraw, "Opencv ", cv::Point(frameDraw.cols - 180, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 0), 2);
        line(frameDraw, cv::Point(frameDraw.cols - 100, 50), cv::Point(frameDraw.cols - 10, 50), Scalar(255, 0, 0), 2, 1);
        putText(frameDraw, "KCF ", cv::Point(frameDraw.cols - 180, 75), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);
        line(frameDraw, cv::Point(frameDraw.cols - 100, 75), cv::Point(frameDraw.cols - 10, 75), Scalar(0, 255, 0), 2, 1);
        putText(frameDraw, "DSST ", cv::Point(frameDraw.cols - 180, 100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        line(frameDraw, cv::Point(frameDraw.cols - 100, 100), cv::Point(frameDraw.cols - 10, 100), Scalar(0, 0, 255), 2, 1);
        putText(frameDraw, "GOTURN ", cv::Point(frameDraw.cols - 180, 125), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 0), 2);
        line(frameDraw, cv::Point(frameDraw.cols - 100, 125), cv::Point(frameDraw.cols - 10, 125), Scalar(255, 255, 0), 2, 1);
        putText(frameDraw, "ECO ", cv::Point(frameDraw.cols - 180, 150), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 255), 2);
        line(frameDraw, cv::Point(frameDraw.cols - 100, 150), cv::Point(frameDraw.cols - 10, 150), Scalar(255, 0, 255), 2, 1);

        //=============================================================================
        // Display frameDraw.
        //cvNamedWindow("Tracking", CV_WINDOW_NORMAL);
        imshow("Tracking", frameDraw);

        int c = cvWaitKey(1);
        if (c != -1)
            c = c % 256;
        if (c == 27)
        {
            cvDestroyWindow("Tracking");
            return 0;
        }
        waitKey(1);
        // Read next image======================================================
        f++;
        osfile.str("");

        if (databaseType == "TLP")
        {
            // Read the groundtruth bbox
            getline(*groundtruth, s, ',');
            //f = atof(s.c_str());
            getline(*groundtruth, s, ',');
            x = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y = atof(s.c_str());
            getline(*groundtruth, s, ',');
            w = atof(s.c_str());
            getline(*groundtruth, s, ',');
            h = atof(s.c_str());
            getline(*groundtruth, s);
            isLost = atof(s.c_str());
            //cout << "gt:" << f << " " << x << " " << y << " " << w << " " << h << " " << isLost << endl;
            osfile << path << "/img/" << setw(5) << setfill('0') << f << ".jpg";
            //cout << osfile.str() << endl;
        }
        else if (databaseType == "TB-2015")
        {
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
            //cout << osfile.str() << endl;
        }
        else if (databaseType == "UAV123")
        {
            // Read the groundtruth bbox
            getline(*groundtruth, s, ',');
            x = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y = atof(s.c_str());
            getline(*groundtruth, s, ',');
            w = atof(s.c_str());
            getline(*groundtruth, s);
            h = atof(s.c_str());
            //cout << "gt:" << x << " " << y << " " << w << " " << h << endl;
            // Read images in a folder
            osfile << path << "/" << setw(6) << setfill('0') << f << ".jpg";
            //cout << osfile.str() << endl;
        }
        else if (databaseType == "VOT-2017")
        {
            // Read the groundtruth bbox
            getline(*groundtruth, s, ',');
            x1 = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y1 = atof(s.c_str());
            getline(*groundtruth, s, ',');
            x2 = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y2 = atof(s.c_str());
            getline(*groundtruth, s, ',');
            x3 = atof(s.c_str());
            getline(*groundtruth, s, ',');
            y3 = atof(s.c_str());
            getline(*groundtruth, s, ',');
            x4 = atof(s.c_str());
            getline(*groundtruth, s);
            y4 = atof(s.c_str());
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
