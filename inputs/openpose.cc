#include "openpose.hpp"

DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                               " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                               " low priority messages and 4 for important ones.");
DEFINE_bool(disable_multi_thread, false, "It would slightly reduce the frame rate in order to highly reduce the lag. Mainly useful"
                                         " for 1) Cases where it is needed a low latency (e.g. webcam in real-time scenarios with"
                                         " low-range GPU devices); and 2) Debugging OpenPose when it is crashing to locate the"
                                         " error.");
DEFINE_int32(profile_speed, 1000, "If PROFILER_ENABLED was set in CMake or Makefile.config files, OpenPose will show some"
                                  " runtime statistics at this frame number.");
// Producer
DEFINE_int32(camera, -1, "The camera index for cv::VideoCapture. Integer in the range [0, 9]. Select a negative"
                         " number (by default), to auto-detect and open the first available camera.");
DEFINE_string(camera_resolution, "-1x-1", "Set the camera resolution (either `--camera` or `--flir_camera`). `-1x-1` will use the"
                                          " default 1280x720 for `--camera`, or the maximum flir camera resolution available for"
                                          " `--flir_camera`");
DEFINE_double(camera_fps, 30.0, "Frame rate for the webcam (also used when saving video). Set this value to the minimum"
                                " value between the OpenPose displayed speed and the webcam real frame rate.");
DEFINE_string(video, "", "Use a video file instead of the camera. Use `examples/media/video.avi` for our default"
                         " example video.");
DEFINE_string(image_dir, "", "Process a directory of images. Use `examples/media/` for our default example folder with 20"
                             " images. Read all standard formats (jpg, png, bmp, etc.).");
DEFINE_bool(flir_camera, false, "Whether to use FLIR (Point-Grey) stereo camera.");
DEFINE_int32(flir_camera_index, -1, "Select -1 (default) to run on all detected flir cameras at once. Otherwise, select the flir"
                                    " camera index to run, where 0 corresponds to the detected flir camera with the lowest"
                                    " serial number, and `n` to the `n`-th lowest serial number camera.");
DEFINE_string(ip_camera, "", "String with the IP camera URL. It supports protocols like RTSP and HTTP.");
DEFINE_uint64(frame_first, 0, "Start on desired frame number. Indexes are 0-based, i.e. the first frame has index 0.");
DEFINE_uint64(frame_last, -1, "Finish on desired frame number. Select -1 to disable. Indexes are 0-based, e.g. if set to"
                              " 10, it will process 11 frames (0-10).");
DEFINE_bool(frame_flip, false, "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
DEFINE_int32(frame_rotate, 0, "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
DEFINE_bool(frames_repeat, false, "Repeat frames when finished.");
DEFINE_bool(process_real_time, false, "Enable to keep the original source frame rate (e.g. for video). If the processing time is"
                                      " too long, it will skip frames. If it is too fast, it will slow it down.");
DEFINE_string(camera_parameter_folder, "models/cameraParameters/flir/", "String with the folder where the camera parameters are located.");
DEFINE_bool(frame_keep_distortion, false, "If false (default), it will undistortionate the image based on the"
                                          " `camera_parameter_folder` camera parameters; if true, it will not undistortionate, i.e.,"
                                          " it will leave it as it is.");
// OpenPose
DEFINE_string(model_folder, "/media/elab/sdd/mycodes/openpose/models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(output_resolution, "-1x-1", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                          " input image resolution.");
DEFINE_int32(num_gpu, -1, "The number of GPU devices to use. If negative, it will use all the available GPUs in your"
                          " machine.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_int32(keypoint_scale, 0, "Scaling of the (x,y) coordinates of the final pose data array, i.e. the scale of the (x,y)"
                                " coordinates that will be saved with the `write_json` & `write_keypoint` flags."
                                " Select `0` to scale it to the original source resolution; `1`to scale it to the net output"
                                " size (set with `net_resolution`); `2` to scale it to the final output size (set with"
                                " `resolution`); `3` to scale it in the range [0,1], where (0,0) would be the top-left"
                                " corner of the image, and (1,1) the bottom-right one; and 4 for range [-1,1], where"
                                " (-1,-1) would be the top-left corner of the image, and (1,1) the bottom-right one. Non"
                                " related with `scale_number` and `scale_gap`.");
DEFINE_int32(number_people_max, -1, "This parameter will limit the maximum number of people detected, by keeping the people with"
                                    " top scores. The score is based in person area over the image, body part score, as well as"
                                    " joint score (between each pair of connected body parts). Useful if you know the exact"
                                    " number of people in the scene, so it can remove false positives (if all the people have"
                                    " been detected. However, it might also include false negatives by removing very small or"
                                    " highly occluded people. -1 will keep them all.");
// OpenPose Body Pose
DEFINE_bool(body_disable, false, "Disable body keypoint detection. Option only possible for faster (but less accurate) face"
                                 " keypoint detection.");
DEFINE_string(model_pose, "BODY_25", "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                     "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(net_resolution, "-1x368", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                              " If you want to change the initial scale, you actually want to multiply the"
                              " `net_resolution` by your desired initial scale.");
// OpenPose Body Pose Heatmaps and Part Candidates
DEFINE_bool(heatmaps_add_parts, false, "If true, it will fill op::Datum::poseHeatMaps array with the body part heatmaps, and"
                                       " analogously face & hand heatmaps to op::Datum::faceHeatMaps & op::Datum::handHeatMaps."
                                       " If more than one `add_heatmaps_X` flag is enabled, it will place then in sequential"
                                       " memory order: body parts + bkg + PAFs. It will follow the order on"
                                       " POSE_BODY_PART_MAPPING in `src/openpose/pose/poseParameters.cpp`. Program speed will"
                                       " considerably decrease. Not required for OpenPose, enable it only if you intend to"
                                       " explicitly use this information later.");
DEFINE_bool(heatmaps_add_bkg, false, "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
                                     " background.");
DEFINE_bool(heatmaps_add_PAFs, false, "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
DEFINE_int32(heatmaps_scale, 2, "Set 0 to scale op::Datum::poseHeatMaps in the range [-1,1], 1 for [0,1]; 2 for integer"
                                " rounded [0,255]; and 3 for no scaling.");
DEFINE_bool(part_candidates, false, "Also enable `write_json` in order to save this information. If true, it will fill the"
                                    " op::Datum::poseCandidates array with the body part candidates. Candidates refer to all"
                                    " the detected body parts, before being assembled into people. Note that the number of"
                                    " candidates is equal or higher than the number of final body parts (i.e. after being"
                                    " assembled into people). The empty body parts are filled with 0s. Program speed will"
                                    " slightly decrease. Not required for OpenPose, enable it only if you intend to explicitly"
                                    " use this information.");
// OpenPose Face
DEFINE_bool(face, false, "Enables face keypoint detection. It will share some parameters from the body pose, e.g."
                         " `model_folder`. Note that this will considerable slow down the performance and increse"
                         " the required GPU memory. In addition, the greater number of people on the image, the"
                         " slower OpenPose will be.");
DEFINE_string(face_net_resolution, "368x368", "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the face keypoint"
                                              " detector. 320x320 usually works fine while giving a substantial speed up when multiple"
                                              " faces on the image.");
// OpenPose Hand
DEFINE_bool(hand, false, "Enables hand keypoint detection. It will share some parameters from the body pose, e.g."
                         " `model_folder`. Analogously to `--face`, it will also slow down the performance, increase"
                         " the required GPU memory and its speed depends on the number of people.");
DEFINE_string(hand_net_resolution, "368x368", "Multiples of 16 and squared. Analogous to `net_resolution` but applied to the hand keypoint"
                                              " detector.");
DEFINE_int32(hand_scale_number, 1, "Analogous to `scale_number` but applied to the hand keypoint detector. Our best results"
                                   " were found with `hand_scale_number` = 6 and `hand_scale_range` = 0.4.");
DEFINE_double(hand_scale_range, 0.4, "Analogous purpose than `scale_gap` but applied to the hand keypoint detector. Total range"
                                     " between smallest and biggest scale. The scales will be centered in ratio 1. E.g. if"
                                     " scaleRange = 0.4 and scalesNumber = 2, then there will be 2 scales, 0.8 and 1.2.");
DEFINE_bool(hand_tracking, false, "Adding hand tracking might improve hand keypoints detection for webcam (if the frame rate"
                                  " is high enough, i.e. >7 FPS per GPU) and video. This is not person ID tracking, it"
                                  " simply looks for hands in positions at which hands were located in previous frames, but"
                                  " it does not guarantee the same person ID among frames.");
// OpenPose 3-D Reconstruction
DEFINE_bool(3d, false, "Running OpenPose 3-D reconstruction demo: 1) Reading from a stereo camera system."
                       " 2) Performing 3-D reconstruction from the multiple views. 3) Displaying 3-D reconstruction"
                       " results. Note that it will only display 1 person. If multiple people is present, it will"
                       " fail.");
DEFINE_int32(3d_min_views, -1, "Minimum number of views required to reconstruct each keypoint. By default (-1), it will"
                               " require all the cameras to see the keypoint in order to reconstruct it.");
DEFINE_int32(3d_views, 1, "Complementary option to `--image_dir` or `--video`. OpenPose will read as many images per"
                          " iteration, allowing tasks such as stereo camera processing (`--3d`). Note that"
                          " `--camera_parameters_folder` must be set. OpenPose must find as many `xml` files in the"
                          " parameter folder as this number indicates.");
// Extra algorithms
DEFINE_bool(identification, false, "Experimental, not available yet. Whether to enable people identification across frames.");
DEFINE_int32(tracking, -1, "Experimental, not available yet. Whether to enable people tracking across frames. The"
                           " value indicates the number of frames where tracking is run between each OpenPose keypoint"
                           " detection. Select -1 (default) to disable it or 0 to run simultaneously OpenPose keypoint"
                           " detector and tracking for potentially higher accurary than only OpenPose.");
DEFINE_int32(ik_threads, 0, "Experimental, not available yet. Whether to enable inverse kinematics (IK) from 3-D"
                            " keypoints to obtain 3-D joint angles. By default (0 threads), it is disabled. Increasing"
                            " the number of threads will increase the speed but also the global system latency.");
// OpenPose Rendering
DEFINE_int32(part_to_show, 0, "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
                              " part heat map, 19 for the background heat map, 20 for all the body part heat maps"
                              " together, 21 for all the PAFs, 22-40 for each body part pair PAF.");
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                     " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                     " `alpha_pose`, and `alpha_pose`.");
// OpenPose Rendering Pose
DEFINE_double(render_threshold, 0.05, "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                      " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                      " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                      " more false positives (i.e. wrong detections).");
DEFINE_int32(render_pose, -1, "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering"
                              " (slower but greater functionality, e.g. `alpha_X` flags). If -1, it will pick CPU if"
                              " CPU_ONLY is enabled, or GPU if CUDA is enabled. If rendering is enabled, it will render"
                              " both `outputData` and `cvOutputData` with the original image and desired body part to be"
                              " shown (i.e. keypoints, heat maps or PAFs).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                               " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap, 0.7, "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                  " heatmap, 0 will only show the frame. Only valid for GPU rendering.");
// OpenPose Rendering Face
DEFINE_double(face_render_threshold, 0.4, "Analogous to `render_threshold`, but applied to the face keypoints.");
DEFINE_int32(face_render, -1, "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same"
                              " configuration that `render_pose` is using.");
DEFINE_double(face_alpha_pose, 0.6, "Analogous to `alpha_pose` but applied to face.");
DEFINE_double(face_alpha_heatmap, 0.7, "Analogous to `alpha_heatmap` but applied to face.");
// OpenPose Rendering Hand
DEFINE_double(hand_render_threshold, 0.2, "Analogous to `render_threshold`, but applied to the hand keypoints.");
DEFINE_int32(hand_render, -1, "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same"
                              " configuration that `render_pose` is using.");
DEFINE_double(hand_alpha_pose, 0.6, "Analogous to `alpha_pose` but applied to hand.");
DEFINE_double(hand_alpha_heatmap, 0.7, "Analogous to `alpha_heatmap` but applied to hand.");
// Display
DEFINE_bool(fullscreen, false, "Run in full-screen mode (press f during runtime to toggle).");
DEFINE_bool(no_gui_verbose, false, "Do not write text on output images on GUI (e.g. number of current frame and people). It"
                                   " does not affect the pose rendering.");
DEFINE_int32(display, -1, "Display mode: -1 for automatic selection; 0 for no display (useful if there is no X server"
                          " and/or to slightly speed up the processing if visual output is not required); 2 for 2-D"
                          " display; 3 for 3-D display (if `--3d` enabled); and 1 for both 2-D and 3-D display.");
// Result Saving
DEFINE_string(write_images, "", "Directory to write rendered frames in `write_images_format` image format.");
DEFINE_string(write_images_format, "png", "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV"
                                          " function cv::imwrite for all compatible extensions.");
DEFINE_string(write_video, "", "Full file path to write rendered frames in motion JPEG video format. It might fail if the"
                               " final path does not finish in `.avi`. It internally uses cv::VideoWriter. Flag"
                               " `camera_fps` controls FPS.");
DEFINE_string(write_json, "", "Directory to write OpenPose output in JSON format. It includes body, hand, and face pose"
                              " keypoints (2-D and 3-D), as well as pose candidates (if `--part_candidates` enabled).");
DEFINE_string(write_coco_json, "", "Full file path to write people pose data with JSON COCO validation format.");
DEFINE_string(write_coco_foot_json, "", "Full file path to write people foot pose data with JSON COCO validation format.");
DEFINE_string(write_heatmaps, "", "Directory to write body pose heatmaps in PNG format. At least 1 `add_heatmaps_X` flag"
                                  " must be enabled.");
DEFINE_string(write_heatmaps_format, "png", "File extension and format for `write_heatmaps`, analogous to `write_images_format`."
                                            " For lossless compression, recommended `png` for integer `heatmaps_scale` and `float` for"
                                            " floating values.");
DEFINE_string(write_keypoint, "", "(Deprecated, use `write_json`) Directory to write the people pose keypoint data. Set format"
                                  " with `write_keypoint_format`.");
DEFINE_string(write_keypoint_format, "yml", "(Deprecated, use `write_json`) File extension and format for `write_keypoint`: json, xml,"
                                            " yaml & yml. Json not available for OpenCV < 3.0, use `write_json` instead.");
// Result Saving - Extra Algorithms
DEFINE_string(write_video_adam, "", "Experimental, not available yet. E.g.: `~/Desktop/adamResult.avi`. Flag `camera_fps`"
                                    " controls FPS.");
DEFINE_string(write_bvh, "", "Experimental, not available yet. E.g.: `~/Desktop/mocapResult.bvh`.");
// UDP communication
DEFINE_string(udp_host, "127.0.0.1", "IP for UDP communication.");
DEFINE_string(udp_port, "8051", "Port number for UDP communication.");


WUserOutput::WUserOutput(cv::Rect2f* bboxGroundtruth)
{
    gt = bboxGroundtruth;
}


void WUserOutput::workConsumer(const std::shared_ptr<std::vector<UserDatum>> &datumsPtr)
{
    try
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            const auto &poseKeypoints = datumsPtr->at(0).poseKeypoints;
            int LWristx = 10000000;
            int LWristy = 10000000;

            int RWristx = 10000000;
            int RWristy = 10000000;

            int Nosex = 0;
            int Nosey = 0;

            //int numberPeopleDetected = poseKeypoints.getSize(0);
            //int numberBodyParts = poseKeypoints.getSize(1);

            int x = 0;
            int y = 0;

            //double score = 0;
            //int baseIndex = 0;

            for (auto person = 0; person < poseKeypoints.getSize(0); person++)
            {
                LWristx = 10000000;
                LWristy = 10000000;

                RWristx = 10000000;
                RWristy = 10000000;

                Nosex = 0;
                Nosey = 0;

                xmin = 10000000;
                xmax = 0;
                ymin = 10000000;
                ymax = 0;
                //&& detect==0
                //			op::log("/n:");
                for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
                {
                    x = poseKeypoints[{person, bodyPart, 0}];
                    y = poseKeypoints[{person, bodyPart, 1}];

                    if (xmin > x && x > 0)
                    {
                        xmin = x;
                    }

                    if (xmax < x)
                    {
                        xmax = x;
                    }

                    if (ymin > y && y > 0)
                    {
                        ymin = y;
                    }

                    if (ymax < y)
                    {
                        ymax = y;
                    }

                    //score = poseKeypoints[{person, bodyPart, 2}];
                    //		op::log("bodyPart"+ std::to_string(bodyPart) + " x " + std::to_string(x) + " y " + std::to_string(y) )

                    if ((bodyPart == 3 || bodyPart == 4) && x > 0 && y > 0)
                    {
                        LWristx = x;
                        LWristy = y;
                        //					op::log(" position of Nose and Wrist");
                        //					op::log( "Nose   " + std::to_string(Nosex ) + " " + std::to_string( Nosey  ));
                        //					op::log( "Rwrist " + std::to_string(RWristx)+ " " + std::to_string( RWristy));
                        //					op::log( "Lwrist " + std::to_string(LWristx)+ " " + std::to_string( LWristy));
                    }
                    else if ((bodyPart == 6 || bodyPart == 7) && x > 0 && y > 0)
                    {
                        RWristx = x;
                        RWristy = y;
                        //					op::log(" position of Nose and Wrist");
                        //					op::log( "Nose   " + std::to_string(Nosex ) + " " + std::to_string( Nosey  ));
                        //					op::log( "Rwrist " + std::to_string(RWristx)+ " " + std::to_string( RWristy));
                        //					op::log( "Lwrist " + std::to_string(LWristx)+ " " + std::to_string( LWristy));
                    }
                    else if ((bodyPart == 15 || bodyPart == 16 || bodyPart == 17 || bodyPart == 18) && x > 0 && y > 0)
                    {
                        Nosex = x;
                        Nosey = y;
                        //					op::log(" position of Nose and Wrist");
                        //					op::log( "Nose   " + std::to_string(Nosex ) + " " + std::to_string( Nosey  ));
                        //					op::log( "Rwrist " + std::to_string(RWristx)+ " " + std::to_string( RWristy));
                        //					op::log( "Lwrist " + std::to_string(LWristx)+ " " + std::to_string( LWristy));
                    }

                    if (Nosey > LWristy && Nosey > RWristy && Nosex > 0 && Nosey > 0 && LWristx > 0 && RWristx > 0 && LWristy > 0 && RWristy > 0 && LWristy < 10000 && RWristy < 10000 && LWristx < 10000 && RWristx < 10000 && Nosex < 10000 && Nosey < 10000)
                    {
                        op::log("Person " + std::to_string(person));
                        op::log("last position of Nose and Wrist");
                        op::log("Nose   " + std::to_string(Nosex) + " " + std::to_string(Nosey));
                        op::log("Rwrist " + std::to_string(RWristx) + " " + std::to_string(RWristy));
                        op::log("Lwrist " + std::to_string(LWristx) + " " + std::to_string(LWristy));
                        std::cout << "xmin xmax ymin ymax:\n";
                        std::cout << xmin << " " << xmax << " " << ymin << " " << ymax << "\n";
                        gt->x = xmin;
                        gt->y = ymin;
                        gt->width = xmax - xmin;
                        gt->height = ymax - ymin;
                        //f = new cv::Mat(datumsPtr->at(0).cvOutputData.size(),datumsPtr->at(0).cvOutputData.type());
                        //cv::imwrite("ini.jpg", *f);
                        //cv::destroyAllWindows();
                        this->stop();
                        /*
                        cv::Point nose = cv::Point(Nosex, Nosey);
                        cv::circle(datumsPtr->at(0).cvOutputData, nose, 5, cv::Scalar(0, 255, 0));
                        cv::Point rwrist = cv::Point(RWristx, RWristy);
                        cv::circle(datumsPtr->at(0).cvOutputData, rwrist, 5, cv::Scalar(255, 255, 0));
                        cv::Point lwrist = cv::Point(LWristx, LWristy);
                        cv::circle(datumsPtr->at(0).cvOutputData, lwrist, 5, cv::Scalar(255, 255, 0));
                        rectangle(datumsPtr->at(0).cvOutputData, gt, cv::Scalar(0, 0, 0), 2, 1);
                        gt.x = 0;
                        gt.y = 0;
                        gt.width = 0;
                        gt.height = 0;
                        */
                    }
                }
            }
            op::log(" ");
            // Alternative: just getting std::string equivalent
            //op::log("Face keypoints: " + datumsPtr->at(0).faceKeypoints.toString());
            //op::log("Left hand keypoints: " + datumsPtr->at(0).handKeypoints[0].toString());
            //op::log("Right hand keypoints: " + datumsPtr->at(0).handKeypoints[1].toString());
            // Heatmaps
            f = datumsPtr->at(0).cvOutputData;
            const auto &poseHeatMaps = datumsPtr->at(0).poseHeatMaps;
            if (!poseHeatMaps.empty())
            {
                op::log("Pose heatmaps size: [" + std::to_string(poseHeatMaps.getSize(0)) + ", " + std::to_string(poseHeatMaps.getSize(1)) + ", " + std::to_string(poseHeatMaps.getSize(2)) + "]");
                const auto &faceHeatMaps = datumsPtr->at(0).faceHeatMaps;
                op::log("Face heatmaps size: [" + std::to_string(faceHeatMaps.getSize(0)) + ", " + std::to_string(faceHeatMaps.getSize(1)) + ", " + std::to_string(faceHeatMaps.getSize(2)) + ", " + std::to_string(faceHeatMaps.getSize(3)) + "]");
                const auto &handHeatMaps = datumsPtr->at(0).handHeatMaps;
                op::log("Left hand heatmaps size: [" + std::to_string(handHeatMaps[0].getSize(0)) + ", " + std::to_string(handHeatMaps[0].getSize(1)) + ", " + std::to_string(handHeatMaps[0].getSize(2)) + ", " + std::to_string(handHeatMaps[0].getSize(3)) + "]");
                op::log("Right hand heatmaps size: [" + std::to_string(handHeatMaps[1].getSize(0)) + ", " + std::to_string(handHeatMaps[1].getSize(1)) + ", " + std::to_string(handHeatMaps[1].getSize(2)) + ", " + std::to_string(handHeatMaps[1].getSize(3)) + "]");
            }

            // Display rendered output image
            cv::imshow("User worker GUI", datumsPtr->at(0).cvOutputData);
            // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
            const char key = (char)cv::waitKey(1);
            if (key == 27)
                this->stop();
        }
    }
    catch (const std::exception &e)
    {
        this->stop();
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int openPoseDemo(cv::Rect2f* bboxGroundtruth)
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        //const auto timerBegin = std::chrono::high_resolution_clock::now();

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);

        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // // For debugging
        // // Print all logging messages
        // op::ConfigureLog::setPriorityThreshold(op::Priority::None);
        // // Print out speed values faster
        // op::Profiler::setDefaultX(100);

        // Applying user defined configuration - Google flags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
        // producerType
        const auto producerSharedPtr = op::flagsToProducer(FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera,
                                                           FLAGS_flir_camera, FLAGS_camera_resolution, FLAGS_camera_fps,
                                                           FLAGS_camera_parameter_folder, !FLAGS_frame_keep_distortion,
                                                           (unsigned int)FLAGS_3d_views, FLAGS_flir_camera_index);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.",
                    op::Priority::Max);
        // keypointScale
        const auto keypointScale = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScale = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;
        // Logging
        op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

        // OpenPose wrapper
        op::log("Configuring OpenPose wrapper...", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
        // op::Wrapper<std::vector<op::Datum>> opWrapper;
        op::Wrapper<std::vector<UserDatum>> opWrapper;

        // Initializing the user custom classes
        // GUI (Display)
        auto wUserOutput = std::make_shared<WUserOutput>(bboxGroundtruth);

        // Add custom processing
        const auto workerOutputOnNewThread = true;
        opWrapper.setWorkerOutput(wUserOutput, workerOutputOnNewThread);

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            !FLAGS_body_disable, netInputSize, outputSize, keypointScale, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScale, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, enableGoogleLogging};
        // Face configuration (use op::WrapperStructFace{} to disable it)

        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceNetInputSize, op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        // Hand configuration (use op::WrapperStructHand{} to disable it)

        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range, FLAGS_hand_tracking,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)

        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        // Producer (use default to disable any input)

        const op::WrapperStructInput wrapperStructInput{
            producerSharedPtr, FLAGS_frame_first, FLAGS_frame_last, FLAGS_process_real_time, FLAGS_frame_flip,
            FLAGS_frame_rotate, FLAGS_frames_repeat};
        // Consumer (comment or use default argument to disable any output)
        // const op::WrapperStructOutput wrapperStructOutput{op::flagsToDisplayMode(FLAGS_display, FLAGS_3d),
        //                                                   !FLAGS_no_gui_verbose, FLAGS_fullscreen, FLAGS_write_keypoint,
        const auto displayMode = op::DisplayMode::NoDisplay;
        const bool guiVerbose = false;
        const bool fullScreen = false;
        const op::WrapperStructOutput wrapperStructOutput{
            displayMode, guiVerbose, fullScreen, FLAGS_write_keypoint,
            op::stringToDataFormat(FLAGS_write_keypoint_format), FLAGS_write_json, FLAGS_write_coco_json,
            FLAGS_write_coco_foot_json, FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video,
            FLAGS_camera_fps, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_adam,
            FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
        // Configure wrapper
        opWrapper.configure(wrapperStructPose, wrapperStructFace, wrapperStructHand, wrapperStructExtra,
                            wrapperStructInput, wrapperStructOutput);
        // Set to single-thread running (to debug and/or reduce latency)
        //if (FLAGS_disable_multi_thread)
        opWrapper.disableMultiThreading();

        // Start processing
        // Two different ways of running the program on multithread environment
        op::log("Starting thread(s)...", op::Priority::High);
        // Option a) Recommended - Also using the main thread (this thread) for processing (it saves 1 thread)
        // Start, run & stop threads - it blocks this thread until all others have finished

        opWrapper.exec();

        return 0;
    }
    catch (const std::exception &e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return -1;
    }
}


OpenPose::OpenPose(){};
OpenPose::~OpenPose(){};
void OpenPose::IniRead(cv::Rect2f &bboxGroundtruth)
{
    cv::Rect2f *gt ;//= NULL;
    //gflags::ParseCommandLineFlags(&argc, &argv, true);
    openPoseDemo(gt);
    printf("%f %f %f %f\n", gt->x, gt->y, gt->width, gt->height);
    bboxGroundtruth.x = gt->x;
    bboxGroundtruth.y = gt->y;
    bboxGroundtruth.width = gt->width;
    bboxGroundtruth.height = gt->height;
}


/*
int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseDemo
    return openPoseDemo();
}
*/