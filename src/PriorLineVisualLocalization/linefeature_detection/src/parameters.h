

#ifndef INC_2D_3D_POSE_TRACKING_MASTER_PARAMETERS_H
#define INC_2D_3D_POSE_TRACKING_MASTER_PARAMETERS_H

#include <ros/ros.h>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <string.h>


using namespace std;
//using namespace camodocal;
using namespace Eigen;

extern int ROW;
extern int COL;
const int NUM_OF_CAM = 1;

extern std::string IMAGE_TOPIC;
extern std::string CAM_NAME;

extern int MAX_CNT;
extern int MIN_DIST;
extern int FREQ;
extern int EQUALIZE;
extern bool PUB_THIS_FRAME;

//extern cv::Mat undist_map1, undist_map2, K_cam;

//extern camodocal::CameraPtr camera;


//void readIntrinsicParam(const std::string &calib_file);
void readParameters(ros::NodeHandle &n);


#endif //INC_2D_3D_POSE_TRACKING_MASTER_PARAMETERS_H
