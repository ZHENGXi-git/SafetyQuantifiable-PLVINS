
#include "parameters.h"

using namespace std;

std::string IMAGE_TOPIC;
std::string CAM_NAME;

int ROW;
int COL;
int MAX_CNT;
int MIN_DIST;
int FREQ;
int EQUALIZE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);  
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    fsSettings["image_topic"] >> IMAGE_TOPIC; 
    ROW = fsSettings["height"];
    COL = fsSettings["width"];
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    FREQ = fsSettings["freq"];
    EQUALIZE = fsSettings["equalize"];

    CAM_NAME = config_file;

    PUB_THIS_FRAME = true;

    if(0 == FREQ)
    {
        FREQ = 10;
    }

    fsSettings.release();
}
















