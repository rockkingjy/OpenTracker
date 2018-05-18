//
// Created by ise on 16-10-23.
//
#include "loader_base.h"

namespace bfs = boost::filesystem;

bool getBox = false;
bool drawing_box = false;

Loader::Loader(const string& _videos_folder)
        :videos_folder_(_videos_folder) {
    loadervideos(_videos_folder);
}

void Loader::loadervideos(const string _videos_folder){
    if(!bfs::is_directory(_videos_folder)){
        printf("ERROR - %s is not a valid directory!\n",_videos_folder.c_str());
        return;
    }

    vector<string> videos;
    find_subfolders(_videos_folder,&videos);
    printf("find %zu videos ... \n" ,videos.size());

    for(unsigned int i = 0; i < videos.size(); ++i){
        const string video_name = videos[i];
        const string video_path = _videos_folder+"/"+video_name;
        Video video;
        video.video_name_ = video_path;
        const boost::regex image_filer(".*\\.jpg");
        find_matching_files(video_path,image_filer,&video.all_frames_);
        printf("%d: %s  %zu frames \n",i,video_name.c_str(),video.all_frames_.size());
        videos_.push_back(video);
    }
}


void Loader::loadFirstBox(const Video& video,cv::Mat& image,cv::Rect& firstBox)
{
    const string& image_file = video.video_name_ + "/" + video.all_frames_[0];

    image = cv::imread(image_file);

    //groundtruth exist, so read first groundtruth from annotation
    if(boost::filesystem::is_regular_file(video.video_name_+"/groundtruth.txt")){

        const string groundtruthPath = video.video_name_+"/groundtruth.txt";
        FILE*  groundtruth = fopen(groundtruthPath.c_str(),"r");

        double ax,ay,bx,by,cx,cy,dx,dy;
        fscanf(groundtruth,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",&ax, &ay, &bx, &by, &cx, &cy, &dx, &dy);

        firstBox.x = (int)round(std::min(ax,std::min(bx,std::min(cx,dx))));
        firstBox.y = (int)round(std::min(ay,std::min(by,std::min(cy,dy))));
        firstBox.width = (int)round(std::max(ax,std::max(bx,std::max(cx,dx)))) - firstBox.x;
        firstBox.height = (int)round(std::max(ay,std::max(by,std::max(cy,dy)))) - firstBox.y;

        //printf("x:%d y:%d w:%d h:%d",firstBox.x,firstBox.y,firstBox.width,firstBox.height);
        cv::Mat firstImage;
        image.copyTo(firstImage);
        cv::rectangle(firstImage,firstBox,CV_RGB(255,0,0),3);
        cv::imshow("first image",firstImage);
        cv::waitKey(0);
    }
    else{


        cv::Rect box;
        const string windowName = "first image";
        cv::namedWindow(windowName,CV_WINDOW_AUTOSIZE);
        cv::setMouseCallback(windowName,mouseHandler,&box);
        cv::Mat firstImage;

        while(!getBox){
            image.copyTo(firstImage);
            cv::rectangle(firstImage,box,CV_RGB(255,0,0));
            cv::imshow(windowName,firstImage);
            cv::waitKey(30);
        }
        cv::setMouseCallback(windowName,NULL,NULL);
        firstBox = box;
        printf("x1: %i  y1: %i "
                       "width: %i  height: %i \n",
              box.x,box.y,box.width,box.height);
        cv::waitKey(0);
        getBox = false;
    }
}

 void Loader::mouseHandler(int event, int x, int y, int flag, void *userdata ){

    cv::Rect* box = (cv::Rect*)userdata;
    switch (event)
    {
        case CV_EVENT_MOUSEMOVE:
            if(drawing_box){
                box->width = x - box->x;
                box->height = y - box->y;
            }
            break;
        case CV_EVENT_LBUTTONDOWN:
            drawing_box = true;
            box->x = x;
            box->y = y;
            break;
        case CV_EVENT_LBUTTONUP:
            drawing_box = false;
            getBox = true;
            break;
        default:
            break;
    }
}
