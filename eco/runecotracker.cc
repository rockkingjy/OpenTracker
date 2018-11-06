#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "eco.hpp"
#include "parameters.hpp"
#include "metrics.hpp"
#include "debug.hpp"

using namespace std;
using namespace cv;
using namespace eco;

int main(int argc, char **argv)
{
    // Database settings
    string databaseTypes[5] = {"Demo","VOT-2017", "TB-2015", "TLP", "UAV123"};
    string databaseType = databaseTypes[0];//4];
    // Read from the images ====================================================
    std::vector<float> CenterError;
    std::vector<float> Iou;
    std::vector<float> FpsEco;
    float SuccessRate = 0.0f;
    float AvgPrecision = 0.0f;
    float AvgIou = 0.0f;
    float AvgFps = 0.0f;
    Metrics metrics;

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
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "VOT-2017")
    {
        string folderVOT = "girl";//"glove";//"ants3";//"drone1";//"iceskater1";//"road";//"bag";//"helicopter";
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
    if (databaseType == "Demo")
    {
        rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
    }
    else if (databaseType == "TLP")
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

    double timereco = (double)getTickCount();
    ECO ecotracker;
    Rect2f ecobbox(x, y, w, h);
    eco::EcoParameters parameters;

    parameters.useCnFeature = false;
    parameters.cn_features.fparams.tablename = "/usr/local/include/opentracker/eco/look_tables/CNnorm.txt";
    /* VOT2016_HC_settings 
    parameters.useDeepFeature = false;
    parameters.useHogFeature = true;
    parameters.useColorspaceFeature = false;
    parameters.useCnFeature = true;
    parameters.useIcFeature = true;
    parameters.learning_rate = 0.01;
    parameters.projection_reg = 5e-7;
    parameters.init_CG_iter = 10 * 20;
    parameters.CG_forgetting_rate = 60;
    parameters.precond_reg_param = 0.2;
    parameters.reg_window_edge = 4e-3;
    parameters.use_scale_filter = false;
    */
    /* VOT2016_DEEP_settings
    parameters.useDeepFeature = true;
    parameters.useHogFeature = true;
    parameters.useColorspaceFeature = false;
    parameters.useCnFeature = true;
    parameters.useIcFeature = true;
    parameters.hog_features.fparams.cell_size = 4;
    parameters.output_sigma_factor = 1.0f / 12.0f;
    parameters.learning_rate = 0.012;
    parameters.nSamples = 50;
    parameters.skip_after_frame = 1;
    parameters.projection_reg = 2e-7;
    parameters.init_CG_iter = 10 * 20;
    parameters.CG_forgetting_rate = 75;
    parameters.precond_data_param = 0.7;
    parameters.precond_reg_param = 0.1;
    parameters.precond_proj_param = 30;
    parameters.reg_sparsity_threshold = 0.12;
    parameters.reg_window_edge = 4e-3;
    */
    /* SRDCF_settings - not implemented yet
    parameters.useDeepFeature = false;
    parameters.useHogFeature = true;
    parameters.useColorspaceFeature = false;
    parameters.useCnFeature = true;
    parameters.useIcFeature = true;
    parameters.hog_features.fparams.cell_size = 4;
    parameters.learning_rate = 0.010;
    parameters.nSamples = 300;
    parameters.train_gap = 0;
    parameters.skip_after_frame = 0;
    parameters.use_detection_sample = false;
    parameters.use_projection_matrix = false;
    parameters.use_sample_merge = false;
    parameters.init_CG_iter = 50;
    parameters.interpolation_centering = false;
    */
    ecotracker.init(frame, ecobbox, parameters);
    float fpsecoini = getTickFrequency() / ((double)getTickCount() - timereco);

    while (frame.data)
    {
        frame.copyTo(frameDraw); // only copy can do the real copy, just equal not.
        timereco = (double)getTickCount();
        bool okeco = ecotracker.update(frame, ecobbox);
        float fpseco = getTickFrequency() / ((double)getTickCount() - timereco);
        if (okeco)
        {
            rectangle(frameDraw, ecobbox, Scalar(255, 0, 255), 2, 1); //blue
        }
        else
        {
            putText(frameDraw, "ECO tracking failure detected", cv::Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 255), 2);
            //waitKey(0);
        }
/*
        // Draw ground truth box
        if (databaseType == "Demo")
        {
            rectangle(frameDraw, bboxGroundtruth, Scalar(0, 0, 0), 2, 1);
        }
        else if (databaseType == "TLP")
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
*/
        // Display FPS on frameDraw
        ostringstream os; 
        os << float(fpseco); 
        putText(frameDraw, "FPS: " + os.str(), Point(100, 30), FONT_HERSHEY_SIMPLEX,
                0.75, Scalar(255, 0, 255), 2);

        if (parameters.debug == 0)
        {
            imshow("OpenTracker", frameDraw);
        }

        int c = cvWaitKey(1);
        if (c != -1)
            c = c % 256;
        if (c == 27)
        {
            cvDestroyWindow("OpenTracker");
            exit(1);
        }
        waitKey(1);
        // Read next image======================================================
        cout << "Frame:" << f << " FPS:" << fpseco << endl;
        f++;
        osfile.str("");
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
        if(!frame.data)
        {
            break;
        }
        // Calculate the metrics;
        float centererror = metrics.center_error(ecobbox, bboxGroundtruth);
        float iou = metrics.iou(ecobbox, bboxGroundtruth);
        CenterError.push_back(centererror);
        Iou.push_back(iou);
        FpsEco.push_back(fpseco);

        cout << "iou:" << iou << std::endl;

        if(centererror <= 20)
        {
            AvgPrecision++;
        }
        if(iou >= 0.5)
        {
            SuccessRate++;
        }
/*
        if(f%10==0)
        {
            ecotracker.init(frame, bboxGroundtruth, parameters);
        }
*/
    }
#ifdef USE_MULTI_THREAD
    void *status;
    if (pthread_join(ecotracker.thread_train_, &status))
    {
         cout << "Error:unable to join!"  << std::endl;
         exit(-1);
    }
#endif
    AvgPrecision /= (float)(f - 2);
    SuccessRate /= (float)(f - 2);
    AvgIou = std::accumulate(Iou.begin(), Iou.end(), 0.0f) / Iou.size();
    AvgFps = std::accumulate(FpsEco.begin(), FpsEco.end(), 0.0f) / FpsEco.size();
    cout << "Frames:" << f - 2
         << " AvgPrecision:" << AvgPrecision
         << " AvgIou:" << AvgIou 
         << " SuccessRate:" << SuccessRate
         << " IniFps:" << fpsecoini
         << " AvgFps:" << AvgFps << std::endl;

    delete groundtruth;
    return 0;
}
