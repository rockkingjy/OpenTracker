#include "readdatasets.hpp"

ReadDatasets::ReadDatasets(){};
ReadDatasets::~ReadDatasets(){};
void ReadDatasets::IniRead(cv::Rect2f &bboxGroundtruth, cv::Mat &frame)
{
    // Read from the images ====================================================
    if (databaseType == "Demo")
    {
        path = "sequences/Crossing";
        // some of the dataset has '\t' as the delimiter, so first change it to ','.
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
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "VOT-2017")
    {
        string folderVOT = "graduate"; //"glove";//"drone1"; //"iceskater1";//"girl"; //"road";//"iceskater1";//"helicopter";//"matrix";//"leaves";//"sheep";//"racing";//"girl";//"road"; //"uav2";//
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
        cout << f << " " << x << " " << y << " " << w << " " << h << " " << endl;
        // Read images in a folder
        osfile << path << "/img/" << setw(4) << setfill('0') << f << ".jpg";
        cout << osfile.str() << endl;
    }
    else if (databaseType == "TLP")
    {
        path = "/media/elab/sdd/data/TLP/Sam"; //IceSkating";//Drone3";//
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
        string folderUAV = "bike1"; //"person16"; //"person21";//"wakeboard1";// "person22";//
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

    bboxGroundtruth.x = x;
    bboxGroundtruth.y = y;
    bboxGroundtruth.width = w;
    bboxGroundtruth.height = h;

    frame = cv::imread(osfile.str().c_str(), CV_LOAD_IMAGE_UNCHANGED);
    if (!frame.data)
    {
        cout << "Could not open or find the image" << std::endl;
        //return -1;
    }
}
void ReadDatasets::DrawGroundTruth(cv::Rect2f &bboxGroundtruth, cv::Mat &frameDraw)
{
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
}
void ReadDatasets::ReadNextFrame(Rect2f &bboxGroundtruth, cv::Mat &frame)
{
    // Read next image======================================================
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
}