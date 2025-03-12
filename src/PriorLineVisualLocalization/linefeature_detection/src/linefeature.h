

#ifndef INC_2D_3D_POSE_TRACKING_MASTER_LINEFEATURE_H
#define INC_2D_3D_POSE_TRACKING_MASTER_LINEFEATURE_H

#include <iostream>
#include <queue>

#include "parameters.h"
#include "tic_toc.h"

#include <opencv4/opencv2/line_descriptor.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>
#include <opencv2/calib3d.hpp>


using namespace cv::line_descriptor;
using namespace std;
using namespace cv;

struct Line
{
    Point2f StartPt;
    Point2f EndPt;
    float lineWidth;
    Point2f Vp;

    Point2f Center;
    Point2f unitDir; 
    float length;
    float theta;

    float para_a;
    float para_b;
    float para_c;

    float image_dx;
    float image_dy;
    float line_grad_avg;

    float xMin;
    float xMax;
    float yMin;
    float yMax;
    unsigned short id;
    int colorIdx;
};

class LineFeatureDetection
{
public:
    LineFeatureDetection();

    void readIntrinsicParameter(const string &calib_file);
    vector<Line> undistortedLinePoints(std::vector<KeyLine> lines);


    cv::Mat Intrinsic_cam;
    cv::Mat distortion;
    cv::Mat map1, map2;    // use for undistortion remap

    Eigen::Matrix3d K_cam;

    int width, height;
    int frame_cnt;

    double m_k1, m_k2, m_p1, m_p2;
    double m_fx, m_fy, m_cx, m_cy;

    double sum_time;
    double length_threshold;

    int undisKeyLine;

};

#endif //INC_2D_3D_POSE_TRACKING_MASTER_LINEFEATURE_H
