
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "kcftracker.hpp"

using namespace std;
using namespace cv;
using namespace kcf;

int main(int argc, char **argv)
{
    // Database settings
    string databaseTypes[5] = {"Demo","VOT-2017", "TB-2015", "TLP", "UAV123"};
    string databaseType = databaseTypes[0];
    // Read from the images ====================================================
    int f, isLost;
    float x, y, w, h;
    float x1, y1, x2, y2, x3, y3, x4, y4; //gt for vot
    std::string s;
    std::string path;
    ifstream *groundtruth;
    ostringstream osfile;
    if (databaseType == "Demo")
    {
        path = "../sequences/Crossing";
        // some of the dataset has '\t' as the delimiter, so first change it to ','.
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
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "VOT-2017")
    {
        string folderVOT = "girl"; //"iceskater1";//"road";//"drone1";//"iceskater1";//"girl"; //"road";//"bag";////"helicopter";
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
        path = "/media/elab/sdd/data/TB-2015/Crossing"; //Coke"; ///Bird1";//BlurFace";
        // some of the dataset has '\t' as the delimiter, so first change it to ','.
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
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "TLP")
    {
        path = "/media/elab/sdd/data/TLP/Drone1";//Sam";//Drone2"; //Bike";//Alladin";//IceSkating";//
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
        string folderUAV = "bike1"; //"person23";//"building1";//"wakeboard2"; //"person3";//
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

    //imshow("OpenTracker", frameDraw);
    //waitKey(0);
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

    while (frame.data)
    {
        frame.copyTo(frameDraw); // only copy can do the real copy, just equal not.
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
        // Draw ground truth box
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
        ostringstream os; 
        os << float(fpskcf); 
        putText(frameDraw, "FPS: " + os.str(), Point(100, 30), FONT_HERSHEY_SIMPLEX,
                0.75, Scalar(0, 225, 0), 2);

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
        f++;
        osfile.str("");
        cout << "Frame:" << f << " FPS:" << fpskcf << endl;
        if (databaseType == "Demo")
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
        else if (databaseType == "TLP")
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
/*
        if(f%10==0)
        {
            ecotracker.reset(frame, bboxGroundtruth);
        }
*/      
    }
    return 0;
}