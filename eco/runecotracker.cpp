#include <iostream>
#include <fstream>
#include <string>
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/caffe.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "head.h" //****** caffe C++ problem solution *******   !!!just be careful
#include "eco.h"

//#define USE_VIDEO

using namespace std;
using namespace caffe;
using namespace cv;
using namespace eco;
// Convert to string
#define SSTR(x) static_cast<std::ostringstream &>(           \
                    (std::ostringstream() << std::dec << x)) \
                    .str()
/*
static string WIN_NAME = "ECO-Tracker";

bool gotBB = false;
bool drawing_box = false;
cv::Rect box;

void mouseHandler(int event, int x, int y, int flags, void *param){
	switch (event){
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box){
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = cv::Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0){
			box.x += box.width;
			box.width *= -1;
		}
		if (box.height < 0){
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	}
}

void drawBox(cv::Mat& image, cv::Rect box, cv::Scalar color, int thick){
	rectangle(image, cvPoint(box.x, box.y), cvPoint(box.x + box.width, box.y + box.height), color, thick);
}
*/

int main()
{
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
    if (databaseType == "TLP")
    {
        path = "/media/elab/sdd/data/TLP/Bike"; //Alladin";//IceSkating";//Sam";
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
    else if (databaseType == "TB-2015")
    {
        path = "/media/elab/sdd/data/TB-2015/Coke"; ///Bird1";//BlurFace";
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
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "UAV123")
    {
        string folderUAV = "uav8"; //"bike1"; //
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
    else if (databaseType == "VOT-2017")
    {
        string folderVOT = "girl"; //"iceskater1";//"drone1"; //"iceskater2";//"helicopter";//"matrix";//"leaves";//"sheep";//"racing";//"girl";//"road"; //"uav2";//
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

    Rect2f bboxGroundtruth(x, y, w, h);

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

    //imshow("Tracking", frame);

    ECO ecotracker;
    Rect2f ecobbox(x, y, w, h);
    ecotracker.init(frame, ecobbox);

    while (frame.data)
    {
        double timereco = (double)getTickCount();
        bool okeco = ecotracker.update(frame, ecobbox);
        float fpseco = getTickFrequency() / ((double)getTickCount() - timereco);
        if (okeco)
        {
            rectangle(frame, ecobbox, Scalar(0, 225, 0), 2, 1); //blue
        }
        else
        {
            putText(frame, "ECO tracking failure detected", cv::Point(100, 80), FONT_HERSHEY_SIMPLEX,
                    0.75, Scalar(0, 225, 0), 2);
        }

        // Draw ground truth box
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
        putText(frame, "FPS: " + SSTR(float(fpseco)), Point(100, 50), FONT_HERSHEY_SIMPLEX,
                0.75, Scalar(0, 225, 0), 2);

        //imshow("Tracking", frame);

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
        cout << f << " FPS:" << fpseco << endl;
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
    /*
#ifdef  USE_VIDEO
	// **********Frame readed****************************************
	cv::Mat frame;
	cv::Rect result;

	// ***********reading frome video**********************************
	cv::namedWindow(WIN_NAME);
	cv::VideoCapture capture;
	capture.open("/home/elab/Videos/tracking_bike.mp4");
	if (!capture.isOpened())
	{
		std::cout << "capture device failed to open!" << std::endl;
		return -1;
	}

	// **********Register mouse callback to draw the bounding box******
	cvNamedWindow(WIN_NAME.c_str(), CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(WIN_NAME.c_str(), mouseHandler, NULL);
	
	capture >> frame;
	cv::Mat temp;
	frame.copyTo(temp);
	while (!gotBB)
	{
		drawBox(frame, box, cv::Scalar(0, 0, 255), 1);
		cv::imshow(WIN_NAME, frame);
		temp.copyTo(frame);
		if (cvWaitKey(20) == 27)
			return 1;
	}
	// ************** Remove callback  *********************************
	cvSetMouseCallback(WIN_NAME.c_str(), NULL, NULL);
	printf("bbox:%d, %d, %d, %d\n", box.x, box.y, box.width, box.height);

	ECO Eco(1, proto, model, mean_file,mean_yml);

	Eco.init(frame, box);

	int idx = 0;
	while (idx++<100)
	{
		capture >> frame;
		if (frame.empty())
			return -1;
		Eco.process_frame(frame);

		//rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
		//imshow(WIN_NAME, frame);
		//waitKey(1);
	}
#else
	DIR *dir=nullptr;
	struct dirent *entry=nullptr;
	string path = "F:\\code_tfy\\Matlab\\ECO\\ECO-Caffe\\3 - test\\sequences\\MountainBike\\img";
	if ((dir = opendir(path.c_str())) == NULL)
	{
		assert("Error opening \n ");
		return 1;
	}

	ECO eco_tracker(0,proto, model, mean_file);
	size_t  id = 0;
	cv::Mat frame;
	while ((entry = readdir(dir)) != NULL)
	{
		string img_name = entry->d_name;
		if (*(img_name.end() - 1) == 'g')
		{
			frame = cv::imread(path + "\\" + img_name);
			if (id++ == 0)
			{
				cvNamedWindow(WIN_NAME.c_str(), CV_WINDOW_AUTOSIZE);
				cvSetMouseCallback(WIN_NAME.c_str(), mouseHandler, NULL);

				//cv::resize(frame, frame, cv::Size(frame.cols / 2, frame.rows / 2));
				cv::Mat temp;
				frame.copyTo(temp);
				while (!gotBB)
				{
					drawBox(frame, box, cv::Scalar(0, 0, 255), 1);
					imshow(WIN_NAME, frame);
					temp.copyTo(frame);
					if (cvWaitKey(20) == 27)
						return 1;
				}
				//Remove callback  
				cvSetMouseCallback(WIN_NAME.c_str(), NULL, NULL);

				//Convert im0 to grayscale
				cv::Mat im0_gray;
				if (frame.channels() > 1) {
					cvtColor(frame, im0_gray, CV_BGR2GRAY);
				}

				//box.x = 400 - 1; box.y = 48 - 1; box.width = 87; box.height = 319;
				//Initialize LS tracker
				//box.x =182; box.y = 154; box.width = 100; box.height = 78;
				// car 219 246 288 175
				//box.x = 218; box.y = 245; box.width = 288; box.height = 175;
				eco_tracker.init(frame, box); // police 192 154 100 78

			}
			else
			{
				if (img_name == "0013.jpg")
					std::cout << std::endl;

				//cv::resize(frame, frame, cv::Size(frame.cols /2, frame.rows / 2));
				if (frame.empty())
					return -1;
				cv::Mat im_gray;
				//cvtColor(frame, im_gray, CV_BGR2GRAY);
				eco_tracker.process_frame(frame);
			 
				//cv::rectangle(frame, showRect, cv::Scalar(0, 255, 0));
				//cv::imshow(WIN_NAME, frame);
				//cv::waitKey(1);
			}

		}

	}


	closedir(dir);

#endif 
*/
    return 0;
}