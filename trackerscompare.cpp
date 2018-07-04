
#include "kcf/kcftracker.hpp"
#include "goturn/network/regressor.h"
#include "goturn/tracker/tracker.h"
#include "eco/eco.hpp"
#include "inputs/readdatasets.hpp"
#include "inputs/openpose.hpp"

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
{ /*
    // Read using openpose============================================
    cv::Rect2f bboxGroundtruth;
    cv::Mat frame, frameDraw;

    OpenPose openpose;
    openpose.IniRead(bboxGroundtruth);
    VideoCapture cap(0); // open the default camera
    if (!cap.isOpened()) // check if we succeeded
        return -1;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    cap >> frame;
    //frame.copyTo(frameDraw);
    //rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    //imshow("Tracking", frameDraw);
    */
   
    // Read from the datasets==========================================
    ::google::InitGoogleLogging(argv[0]);
    cv::Rect2f bboxGroundtruth;
    cv::Mat frame, frameDraw;
    ReadDatasets readdatasets;
    readdatasets.IniRead(bboxGroundtruth, frame);
    frame.copyTo(frameDraw);
    readdatasets.DrawGroundTruth(bboxGroundtruth, frameDraw);
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
    Rect2d opencvbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
    opencvtracker->init(frame, opencvbbox);

    // Create KCFTracker:
    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //LAB color space features
    KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    Rect2d kcfbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
    kcftracker.init(frame, kcfbbox);

    // Create DSSTTracker:
    DSST = true;
    KCFTracker dssttracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);
    Rect2d dsstbbox((int)bboxGroundtruth.x, (int)bboxGroundtruth.y, (int)bboxGroundtruth.width, (int)bboxGroundtruth.height);
    dssttracker.init(frame, dsstbbox);

    // Create GOTURN tracker:
    const string model_file = "goturn/nets/deploy.prototxt";
    const string pretrain_file = "goturn/nets/goturun_tracker.caffemodel";
    int gpu_id = 0;
    Regressor regressor(model_file, pretrain_file, gpu_id, false);
    goturn::Tracker goturntracker(false);
    cv::Rect goturnbbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
    BoundingBox bbox_gt;
    BoundingBox bbox_estimate_uncentered;
    bbox_gt.getRect(goturnbbox);
    goturntracker.Init(frame, bbox_gt, &regressor);

    // Create ECO trakcer;
    eco::ECO ecotracker;
    Rect2f ecobbox(bboxGroundtruth.x, bboxGroundtruth.y, bboxGroundtruth.width, bboxGroundtruth.height);
    ecotracker.init(frame, ecobbox);

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
        readdatasets.DrawGroundTruth(bboxGroundtruth, frameDraw);

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

        // Display frameDraw.=========================================================
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

        // Read the next frame
        readdatasets.ReadNextFrame(bboxGroundtruth, frame);
        //cap >> frame;
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
