
#include "linefeature.h"

LineFeatureDetection::LineFeatureDetection()
{
    sum_time = 0.0;
    frame_cnt = 0;
}

void LineFeatureDetection::readIntrinsicParameter(const string &calib_file)
{

    cv::FileStorage fsSettings(calib_file, cv::FileStorage::READ);

    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    m_fx = static_cast<double>(fsSettings["fx"]);
    m_fy = fsSettings["fy"];
    m_cx = fsSettings["cx"];
    m_cy = fsSettings["cy"];

    Intrinsic_cam = (cv::Mat_<double>(3, 3) << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1);
    cv::cv2eigen(Intrinsic_cam, K_cam);

    m_k1 = fsSettings["k1"];
    m_k2 = fsSettings["k2"];
    m_p1 = fsSettings["p1"];
    m_p2 = fsSettings["p2"];

    distortion = (cv::Mat_<double>(1, 4) << m_k1, m_k2, m_p1, m_p2);

    width = static_cast<int>(fsSettings["width"]);
    height = static_cast<int>(fsSettings["height"]);

    undisKeyLine = fsSettings["undisKeyLine"];
    length_threshold = fsSettings["length_threshold"];

    cv::Size imageSize(width, height);
    double alpha = 0.0;
    cv::Mat newCameraMatrix = getOptimalNewCameraMatrix(Intrinsic_cam, distortion, imageSize,
                                                        alpha, imageSize, 0);
    initUndistortRectifyMap(Intrinsic_cam, distortion, cv::Mat(), newCameraMatrix,
                            imageSize, CV_32FC1, map1, map2);
    fsSettings.release();
}


vector<Line> LineFeatureDetection::undistortedLinePoints(vector<KeyLine> lines)
{
    vector<Line> un_lines;
    float fx = K_cam(0, 0);
    float fy = K_cam(1, 1);
    float cx = K_cam(0, 2);
    float cy = K_cam(1, 2);
    Point2f startPoint, endPoint;
    for (int i = 0; i < lines.size(); ++i)
    {
        startPoint = lines[i].getStartPoint();
        endPoint = lines[i].getEndPoint();
        un_lines[i].StartPt.x = (startPoint.x - cx)/fx;
        un_lines[i].StartPt.y = (startPoint.y - cy)/fy;
        un_lines[i].EndPt.x = (endPoint.x - cx)/fx;
        un_lines[i].EndPt.y = (endPoint.y - cy)/fy;
    }
    return un_lines;
}