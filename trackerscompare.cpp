
#include "kcf/kcftracker.hpp"
#include "goturn/network/regressor.h"
#include "goturn/tracker/tracker.h"

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
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char **argv)
//int trackercompare()
{
// Create KCFTracker: 
    bool HOG = true, FIXEDWINDOW = true, MULTISCALE = true, LAB = true, DSST = false; //LAB color space features
	KCFTracker kcftracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);

// Create DSSTTracker: 
    DSST = true; 
	KCFTracker dssttracker(HOG, FIXEDWINDOW, MULTISCALE, LAB, DSST);

// Create Opencv tracker:
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
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
    string path = "/media/elab/sdd/data/TLP/Bike";//Alladin";//IceSkating";//Sam";
	// Read the groundtruth bbox
	ifstream groundtruth(path + "/groundtruth_rect.txt");
	int f,x,y,w,h,isLost;
	std::string s;
	getline(groundtruth, s, ',');	
	f = atoi(s.c_str());
	getline(groundtruth, s, ',');
	x = atoi(s.c_str());
	getline(groundtruth, s, ',');	
	y = atoi(s.c_str());
	getline(groundtruth, s, ',');
	w = atoi(s.c_str());
	getline(groundtruth, s, ',');	
	h = atoi(s.c_str());
	getline(groundtruth, s);
	isLost = atoi(s.c_str());
	cout << f <<" " << x <<" " << y <<" " << w <<" " << h <<" " << isLost << endl;
    Rect2d bboxGroundtruth(x,y,w,h);
	
	// Read images in a folder
	ostringstream osfile;
	osfile << path << "/img/" << setw(5) << setfill('0') << f <<".jpg";
	cout << osfile.str() << endl;
    cv::Mat frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if(! frame.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	// Init the trackers=================================================
    Rect2d kcfbbox(x,y,w,h);
    kcftracker.init(frame, kcfbbox);

    Rect2d dsstbbox(x,y,w,h);
    dssttracker.init(frame, dsstbbox);

    Rect2d opencvbbox(x,y,w,h);
    opencvtracker->init(frame, opencvbbox);
/*
    cv::Rect goturnbbox(x,y,w,h);
    BoundingBox bbox_gt;
    BoundingBox bbox_estimate_uncentered;
    bbox_gt.getRect(goturnbbox);
    goturntracker.Init(frame,bbox_gt,&regressor);
*/

    while(frame.data) {
        // Draw ground truth box
		rectangle(frame, bboxGroundtruth, Scalar( 0, 0, 0 ), 2, 1 );

        // Start timer
        double timer = (double)getTickCount();
         
        // Update the KCF tracking result-----------------------------
        bool okkcf = kcftracker.update(frame, kcfbbox);
        
        bool okdsst = dssttracker.update(frame, dsstbbox);
//        cout << kcfbbox.x << ", " << kcfbbox.y << ", " <<kcfbbox.width << ", " << kcfbbox.height << std::endl;
        // Calculate Frames per second (FPS)-------------------------------
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
        // Update the Opencv tracking result----------------------------
        bool okopencv = opencvtracker->update(frame, opencvbbox);
        // Update the GOTURN tracking result--------------------------
/*        goturntracker.Track(frame, &regressor, &bbox_estimate_uncentered);
        bbox_estimate_uncentered.putRect(goturnbbox);
*/
        //============================================================
        // draw kcf bbox
        if (okkcf) {
            rectangle(frame, kcfbbox, Scalar( 225, 0, 0 ), 2, 1); //blue
        } else {
            putText(frame, "Kcf tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX,
                     0.75, Scalar(255,0,0),2);
        }
        if (okdsst) {
            rectangle(frame, dsstbbox, Scalar( 0, 0, 255 ), 2, 1); //blue
        } else {
            putText(frame, "DSST tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX,
                     0.75, Scalar(0,0,255),2);
        }
        // draw opencv bbox
        if (okopencv) {
            rectangle(frame, opencvbbox, Scalar( 0, 225, 0 ), 2, 1); //green
        } else {
            putText(frame, "Opencv tracking failure detected", Point(100,110), FONT_HERSHEY_SIMPLEX,
                     0.75, Scalar(0,225,0),2);
        }
        // draw goturn bbox
        // rectangle(frame, goturnbbox, Scalar(0, 0, 255), 2, 1); //red

        // Display FPS on frame
        putText(frame, "FPS in total: " + SSTR(long(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX,
                     0.75, Scalar(0,0,0), 2);
        // Display tracker type on frame
        putText(frame, "Black:GT; Blue: KCF; Red: DSST; Green: opencv " + trackerType + ";",
                 Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),2);
        //=============================================================================
        // Display frame.        
        //cvNamedWindow("Tracking", CV_WINDOW_NORMAL); 
        imshow("Tracking", frame);

        int c = cvWaitKey(1);
        if (c != -1) c = c%256;
        if (c == 27) {
            cvDestroyWindow("Tracking");
            return 0;
        } 
        waitKey(1);
		// Read next image
		f++;
		osfile.str("");
		osfile << path << "/img/" << setw(5) << setfill('0') << f <<".jpg";
		cout << osfile.str() << endl;
    	frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
		// Read next bbox
		getline(groundtruth, s, ',');	
		f = atoi(s.c_str());
		getline(groundtruth, s, ',');
		x = atoi(s.c_str());
		getline(groundtruth, s, ',');	
		y = atoi(s.c_str());
		getline(groundtruth, s, ',');
		w = atoi(s.c_str());
		getline(groundtruth, s, ',');	
		h = atoi(s.c_str());
		getline(groundtruth, s);
		isLost = atoi(s.c_str());
		//cout << f <<" " << x <<" " << y <<" " << w <<" " << h <<" " << isLost << endl;
		bboxGroundtruth.x = x;
		bboxGroundtruth.y = y;
		bboxGroundtruth.width = w;
		bboxGroundtruth.height = h;

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
