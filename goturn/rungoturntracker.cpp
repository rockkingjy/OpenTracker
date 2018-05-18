#include <opencv2/opencv.hpp>
#include <string>
#include "network/regressor.h"
#include "tracker/tracker.h"
#include "helper/bounding_box.h"
#include "loader/loader_base.h"

using std::string;
using namespace goturn;

int main(int argc, char *argv[]){

    //::google::InitGoogleLogging(argv[0]);

    const string model_file = "nets/deploy.prototxt";
    const string pretrain_file = "nets/goturun_tracker.caffemodel";
    const string video_folder = "/media/elab/sdd/data/VOT/vot2014";
    int gpu_id = 1;

    Regressor regressor(model_file,pretrain_file,gpu_id, false);

    Tracker tracker(false);
    Loader loaderVideo(video_folder);

    unsigned int video_num;
    printf("Please select video: ");
    scanf("%d",&video_num);

    for( ; video_num < loaderVideo.videos_.size(); ++video_num ){
        Video video = loaderVideo.videos_[video_num];
        printf(" video: %d: %s \n",video_num,video.video_name_.c_str());

        cv::Rect goturnBox;
        cv::Mat image_curr;
        loaderVideo.loadFirstBox(video,image_curr,goturnBox);

        BoundingBox bbox_gt;
        bbox_gt.getRect(goturnBox);

        tracker.Init(image_curr,bbox_gt,&regressor);

        std::string windowName(video.video_name_, video.video_name_.find_last_of('/')+1);

        for(unsigned int frame_num = 1; frame_num < video.all_frames_.size(); ++frame_num ){

            const string& image_file = video.video_name_ + "/" + video.all_frames_[frame_num];
            image_curr = cv::imread(image_file);

            // Track and estimate the target's bounding box location in the current image.
            // Important: this method cannot receive bbox_gt (the ground-truth bounding box) as an input.
            BoundingBox bbox_estimate_uncentered;
            tracker.Track(image_curr, &regressor, &bbox_estimate_uncentered);
            bbox_estimate_uncentered.putRect(goturnBox);

            cv::Mat display;
            image_curr.copyTo(display);
            cv::rectangle(display,goturnBox,CV_RGB(255,0,0), 2);
            cv::imshow(windowName,display);

            if(cv::waitKey(1) == 27)
                return 0;
        }
        cv::destroyAllWindows();
    }

    return 0;

}

