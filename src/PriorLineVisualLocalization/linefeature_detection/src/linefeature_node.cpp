

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <mutex>
#include <queue>
#include <thread>
#include <line_msgs/lines2d.h>

#include "linefeature.h"

queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_LineFeature;

std::mutex m_buf;

LineFeatureDetection LineDetector;

//std::string IMAGE_TOPIC;
//std::string CAM_NAME;

bool process_finished = true;
bool first_image_flag = true;
int pub_count = 1;

double first_image_time;


void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img_buf.push(img_msg);
    m_buf.unlock();
}


void line_process()
{
    int flag = 1;
    int image_index = 1;
    while(true)
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;

        m_buf.lock();

        if (!img_buf.empty() )
        {
           // ROS_INFO("-------------img_buf is not empty--------------");
            image_msg = img_buf.front();
            img_buf.pop();
            pub_count++;
            if (pub_count <= FREQ && !img_buf.empty())
            {
                img_buf.pop();
            }
        }
        m_buf.unlock();
        pub_count = 1;

        if(image_msg != NULL)
        {
            if (first_image_flag)
            {
                first_image_flag = false;
                first_image_time = image_msg->header.stamp.toSec();
                ROS_INFO("the first image time stamp is %f", first_image_time);
            }

            process_finished = false;

            cv_bridge::CvImageConstPtr ptr;
            if (image_msg->encoding == "8UC1") //gray img
            {
                sensor_msgs::Image img;
                img.header = image_msg->header;
                img.height = image_msg->height;
                img.width = image_msg->width;
                img.is_bigendian = image_msg->is_bigendian;
                img.step = image_msg->step;
                img.data = image_msg->data;
                img.encoding = "mono8";
                ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
            }
            else //color img
            {
                ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
            }
            cv::Mat _img = ptr->image;

            //--------------------------------line detection--------------------------------//
            cv::Mat img, img_detect;

            img = _img.clone();
            if(EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(_img, img);
            }

            if (LineDetector.undisKeyLine == 0)
            {
                cv::undistort(img, img_detect, LineDetector.Intrinsic_cam, LineDetector.distortion);
            }
            else
            {
                img_detect = img.clone();
            }

            TicToc t_line_detection;

            int L_threshold = int(LineDetector.length_threshold);
            float distance_threshold = 2.4142;
            double canny_th1 = 50.0;
            double canny_th2 = 50.0;
            int canny_aperture_size = 5;
            bool do_merge = true;

            Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(L_threshold, distance_threshold,
                                                                                           canny_th1, canny_th2, canny_aperture_size,
                                                                                           do_merge);
            vector<Vec4f> fld_lines;
            fld_lines.clear();

            fld->detect(img_detect, fld_lines);

            image_index++;

            //ROS_INFO("the average time of line detection is %fms", t_line_detection.toc());

            if (flag == 1)
            {
                sensor_msgs::PointCloudPtr feature_lines(new sensor_msgs::PointCloud);
                sensor_msgs::ChannelFloat32 start_x;
                sensor_msgs::ChannelFloat32 start_y;
                sensor_msgs::ChannelFloat32 end_x;
                sensor_msgs::ChannelFloat32 end_y;

            //    ROS_INFO("publish the line massage!!!");
                feature_lines->header = image_msg->header; 
                feature_lines->header.frame_id = "world";

                Vec4f line;
                geometry_msgs::Point32 p;
                for (int i = 0; i < fld_lines.size(); ++i)  
                {
                    line = fld_lines[i];
                    p.x = 0;
                    p.y = 0;
                    p.z = 1;
                    feature_lines->points.push_back(p);
                    start_x.values.push_back(line[0]);
                    start_y.values.push_back(line[1]);
                    end_x.values.push_back(line[2]);
                    end_y.values.push_back(line[3]);
                }
                feature_lines->channels.push_back(start_x);
                feature_lines->channels.push_back(start_y);
                feature_lines->channels.push_back(end_x);
                feature_lines->channels.push_back(end_y);

                ROS_DEBUG("publish %f, at %f", feature_lines->header.stamp.toSec(), ros::Time::now().toSec());
                pub_LineFeature.publish(feature_lines);

            }
            process_finished = true;

        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "linefeature_detection");

    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);  

   // ROS_INFO("the line feature detection node start");

    ROS_INFO_STREAM("cam name : " << CAM_NAME);

    LineDetector.readIntrinsicParameter(CAM_NAME);

    ROS_INFO("start 2D line detection");

    pub_LineFeature = n.advertise<sensor_msgs::PointCloud>("Lines2d", 1000);

    ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 1000, img_callback);

    std::thread linefeature_process;
    linefeature_process = std::thread(line_process);

    ros::spin();
    return 0;

}