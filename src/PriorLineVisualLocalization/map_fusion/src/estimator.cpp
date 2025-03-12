#include "estimator.h"
#include <omp.h>
#include <stdio.h>
std::string TRACK_RESULT_PATH;
std::string OUTPUT_PATH;
std::string PL_PATH;


using namespace cv;


void estimator::setParameters(const string &calib_file, vector<Vector6d> &_lines3d, vector<double> &alarmThreshold)
{
	frame_count=-1;
	index=0;
	solver_flag=INITIAL;
	lines3d_map= _lines3d;
    ChiTestThreshold = alarmThreshold;
    undisKeyLine = false;

	cv::FileStorage fsSettings(calib_file, cv::FileStorage::READ);
//	cv::FileNode ns = fsSettings["projection_parameters"];
    double m_fx = fsSettings["fx"];
    double m_fy = fsSettings["fy"];
	double m_cx = fsSettings["cx"];
	double m_cy = fsSettings["cy"];
	cv_KMatrix = (cv::Mat_<double>(3, 3) << m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1);


//	ns = fsSettings["distortion_parameters"];
	double m_k1 = fsSettings["k1"];
	double m_k2 = fsSettings["k2"];
	double m_p1 = fsSettings["p1"];
	double m_p2 = fsSettings["p2"];
	cv_dist = (cv::Mat_<double>(1, 4) << m_k1, m_k2, m_p1, m_p2);

	width = static_cast<int>(fsSettings["width"]);
	height = static_cast<int>(fsSettings["height"]);

    undisKeyLine = fsSettings["undisKeyLine"];

	new_Matrix = getOptimalNewCameraMatrix(cv_KMatrix, cv_dist, cv::Size(width, height), 0, cv::Size(width, height), 0);
    initUndistortRectifyMap(cv_KMatrix, cv_dist, cv::Mat(), new_Matrix,
                            cv::Size(width, height), CV_32FC1, map1, map2);
	
	cv::cv2eigen(cv_KMatrix, K);
	//camera frame to body frame
    cv::Mat cv_b2c_R, cv_b2c_T;
    fsSettings["extrinsicRotation"] >> cv_b2c_R;
    fsSettings["extrinsicTranslation"] >> cv_b2c_T;
    cv::cv2eigen(cv_b2c_R, b2c_R);
    cv::cv2eigen(cv_b2c_T, b2c_T);

  //  ROS_INFO_STREAM("Extrinsic_R : " << std::endl << b2c_R);
   // ROS_INFO_STREAM("Extrinsic_T : " << std::endl << b2c_T.transpose());

	m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);

	//optimization parameters
	iterations = static_cast<int>(fsSettings["iterations"]);
	per_inliers = static_cast<int>(fsSettings["per_inliers"]);
	threshold = static_cast<double>(fsSettings["threshold"]);
	lambda = static_cast<double>(fsSettings["lambda"]);
	save = static_cast<int>(fsSettings["savefile"]);
     
    PL_Calculation = static_cast<int>(fsSettings["PL_Calculation"]);
    
    overlap_th = static_cast<double>(fsSettings["overlap_th"]);
    dist_th = static_cast<double>(fsSettings["dist_th"]);
    degree_th = static_cast<double>(fsSettings["degree_th"]);
    angle_th = static_cast<double>(fsSettings["angle_th"]);
    outlier_th = static_cast<double>(fsSettings["outlier_th"]);

	ROS_INFO("Finishing setting params for sliding window...");
}
// load transformation of 3D map to global body, body to camera tranform.
void estimator::loadExtrinsictf(Vector3d &_w2gb_T, Matrix3d &_w2gb_R)
{
	w2gb_T = _w2gb_T;
	w2gb_R = _w2gb_R;

	ROS_INFO("Finishing load extrinsic...");
}

// show matched line pairs
void estimator::VisualizeMatchedLines(const vector<pairsmatch> &matches, const vector<line2d> &line2d_,
                                      const vector<line2d> &undis_line2d_, const vector<line3d> &line3d_, cv::Mat &image,
                                      const Eigen::Matrix3d &K_, const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    cv::Mat img, img3d;

    if (image.channels() != 3)
    {
        cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
        cv::cvtColor(image, img3d, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img = image;
        img3d = image;
    }

    cv::Mat img_show1, img_show2;

    img_show1 = img;
    img_show2 = img;

    cv::Point startPointM;
    cv::Point endPointM;
    cv::Point startPointE;
    cv::Point endPointE;
    line2d lineM;
    line3d lineE_W;
    line3d lineE;
    line2d linePro;

    for (size_t i = 0; i < line2d_.size(); ++i)  
    {
        lineM = line2d_[i];
        startPointM = cv::Point(int(lineM.ptstart[0]), int(lineM.ptstart[1]));
        endPointM = cv::Point(int(lineM.ptend[0]), int(lineM.ptend[1]));

        lineM = undis_line2d_[i];
        startPointM = cv::Point(int(lineM.ptstart[0]), int(lineM.ptstart[1]));
        endPointM = cv::Point(int(lineM.ptend[0]), int(lineM.ptend[1]));
    }

    for (size_t i = 0; i < line3d_.size(); ++i)
    {
        lineE_W = line3d_[i];
        lineE = lineE_W.transform3D(R, t);
        linePro = lineE.project3D(K_);
        startPointE = cv::Point(int(linePro.ptstart[0]), int(linePro.ptstart[1]));
        endPointE = cv::Point(int(linePro.ptend[0]), int(linePro.ptend[1]));
        cv::line(img_show2, startPointE, endPointE, cv::Scalar(0, 255, 0), 2, 8);  
    }

    for (size_t i = 0; i < matches.size(); ++i)
    {
        lineM = matches[i].line2dt;
        lineE_W = matches[i].line3dt;

        lineE = lineE_W.transform3D(R, t);
        linePro = lineE.project3D(K_);
        startPointM = cv::Point(int(lineM.ptstart[0]), lineM.ptstart[1]);
        endPointM = cv::Point(int(lineM.ptend[0]), lineM.ptend[1]);
        startPointE = cv::Point(int(linePro.ptstart[0]), linePro.ptstart[1]);
        endPointE = cv::Point(int(linePro.ptend[0]), linePro.ptend[1]);
        cv::line(img_show1, startPointM, endPointM, cv::Scalar(0, 0, 255), 2, 8);  
        cv::line(img_show1, startPointE, endPointE, cv::Scalar(0, 255, 0), 2, 8);  
    }

    cv::imshow("show matched line", img_show1);
    cv::imshow("show all detected lines", img_show2);
    waitKey(1);
}

void estimator::VisualizeDistortLine(const vector<line2d> &line2d_dist, const vector<line3d> &line3d_, cv::Mat &image,
                          const Eigen::Matrix3d &K_, const cv::Mat &cv_dist, const Eigen::Matrix3d &R,
                          const Eigen::Vector3d &t)
{
    cv::Mat img3d;
    if (image.channels() != 3)
    {
        cv::cvtColor(image, img3d, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img3d = image;
    }

    cv::Point startPointM;
    cv::Point endPointM;
    cv::Point startPointE;
    cv::Point endPointE;
    line2d lineM;
    line3d lineE_W;
    line3d lineE;
    line2d linePro;

    for (size_t i = 0; i < line2d_dist.size(); ++i) 
    {
        lineM = line2d_dist[i];
        startPointM = cv::Point(int(lineM.ptstart[0]), int(lineM.ptstart[1]));
        endPointM = cv::Point(int(lineM.ptend[0]), int(lineM.ptend[1]));
        cv::line(img3d, startPointM, endPointM, cv::Scalar(0, 0, 255), 1, 8);  
    }

    Eigen::Vector3d start, end;
    Eigen::Vector3d start_c, end_c;
    double x_s, y_s, x_e, y_e;
    double x_s_d, y_s_d, x_e_d, y_e_d;
    double u_s, v_s, u_e, v_e;
    double r_2_s, r_2_e, r_4_s, r_4_e;
    double k1, k2, p1, p2;
    double fx, fy, cx, cy;
    k1 = cv_dist.at<double>(0, 0);
    k2 = cv_dist.at<double>(0, 1);
    p1 = cv_dist.at<double>(0, 2);
    p2 = cv_dist.at<double>(0, 3);
    fx = K_(0, 0);
    cx = K_(0, 2);
    fy = K_(1, 1);
    cy = K_(1, 2);

    for (size_t i = 0; i < line3d_.size(); ++i)
    {
        lineE_W = line3d_[i];
        start = lineE_W.ptstart;
        end = lineE_W.ptend;
        start_c = R * start + t;
        end_c = R * end + t;
        // the distortion model
        x_s = start_c[0] / start_c[2];
        y_s = start_c[1] / start_c[2];
        x_e = end_c[0] / end_c[2];
        y_e = end_c[1] / end_c[2];
        r_2_s = Sqrt2(x_s) + Sqrt2(y_s);
        r_2_e = Sqrt2(x_e) + Sqrt2(y_e);
        r_4_s = Sqrt2(r_2_s);
        r_4_e = Sqrt2(r_2_e);
        x_s_d = x_s * (1 + k1 * r_2_s + k2 * r_4_s) + (2 * p1 * x_s * y_s + p2 * (r_2_s + 2 * Sqrt2(x_s)));
        y_s_d = y_s * (1 + k1 * r_2_s + k2 * r_4_s) + (p1 * (r_2_s + 2 * Sqrt2(y_s)) + 2 * p2 * x_s * y_s);
        x_e_d = x_e * (1 + k1 * r_2_e + k2 * r_4_e) + (2 * p1 * x_e * y_e + p2 * (r_2_e + 2 * Sqrt2(x_e)));
        y_e_d = y_e * (1 + k1 * r_2_e + k2 * r_4_e) + (p1 * (r_2_e + 2 * Sqrt2(y_e)) + 2 * p2 * x_e * y_e);
        u_s = fx * x_s_d + cx;
        v_s = fy * y_s_d + cy;
        u_e = fx * x_e_d + cx;
        v_e = fy * y_e_d + cy;

        startPointE = cv::Point(int(u_s), int(v_s));
        endPointE = cv::Point(int(u_e), int(v_e));
        cv::line(img3d, startPointE, endPointE, cv::Scalar(0, 255, 0), 1, 8); 
        cv::imshow("show distort lines", img3d);
        waitKey(1);
    }
}

// create estimator online
void estimator::processImage(double _time_stamp, Vector3d &_vio_T, Matrix3d &_vio_R, cv::Mat &_image, vector<line2d> &_lines2d,
                             Matrix3d &GT_R, Vector3d &GT_T)
{

	if (frame_count < WINDOW_SIZE)
		frame_count++;
	time_stamp[frame_count] = _time_stamp;
	vio_T[frame_count]=_vio_T;
	vio_R[frame_count] = _vio_R;
	image[frame_count]= _image.clone();
	lines2d[frame_count]= _lines2d;
    undist_lines2d[frame_count] = _lines2d;

    if (undisKeyLine == 1)
    {
        undist_lines2d[frame_count].clear();
        undist_lines2d[frame_count] = undistortionLine(_lines2d);
    }

    for (int i = 0; i < 0; ++i)
    {
        ROS_INFO("the processImage line2d is %f, %f", _lines2d[i].ptstart[0], _lines2d[i].ptstart[1]);
        ROS_INFO("the processImage undis line2d is %f, %f", undist_lines2d[frame_count][i].ptstart[0],
                 undist_lines2d[frame_count][i].ptstart[1]);
    }

	if (frame_count > 0)
    {  
        delta_R[frame_count - 1] = vio_R[frame_count - 1] * vio_R[frame_count].transpose();
        delta_T[frame_count - 1] = vio_T[frame_count - 1] - delta_R[frame_count - 1] * vio_T[frame_count];  // Pn = vio_Tn
    }
	delta_R[frame_count] << 1,0,0,
						    0,1,0,
						    0,0,1;
	delta_T[frame_count] << 0.0,0.0,0.0;

	if (solver_flag == INITIAL)
	{
		//find local 3d lines
		T_w[frame_count] = _vio_T;   // VINS-result
		R_w[frame_count] = _vio_R;
        // projection 3D line into camera frame then find the available 3d lines
        // return the 3d lines on the boby origin frame (the start and end points)
		lines3d[frame_count] = UpdateLinesInFOV(_vio_T, _vio_R);

		Matrix3d tempRot = b2c_R.transpose() * R_w[frame_count].transpose();           // from the body origin frame to camera frame
		Vector3d tempTrans = -tempRot * T_w[frame_count] - b2c_R.transpose() * b2c_T;

        matches2d3d[frame_count] = LineCorrespondence(lines3d[frame_count], undist_lines2d[frame_count],
                                                      K, tempRot, tempTrans, overlap_th, dist_th,
                                                      degree_th, angle_th, per_inliers, outlier_th);

		fuse_pose();
		if (frame_count == WINDOW_SIZE-1 || WINDOW_SIZE==0)
		{
			solver_flag = NON_LINEAR; 
		}
	} 
	else
	{
		//predict current frame in the window_size
        // R_w, T_w: state the boby pose on starting point frame
		R_w[frame_count] = delta_R[frame_count - 1].transpose()*R_w[frame_count - 1];
        T_w[frame_count] = delta_R[frame_count - 1].transpose()*(T_w[frame_count-1]-delta_T[frame_count - 1]);
        
        TicToc t_line_pair;

        lines3d[frame_count] = UpdateLinesInFOV(T_w[frame_count], R_w[frame_count]);

        // temRot, temTrans: transform to camera frame
		Matrix3d tempRot = b2c_R.transpose() * R_w[frame_count].transpose();
		Vector3d tempTrans = -tempRot * T_w[frame_count] - b2c_R.transpose() * b2c_T;

        matches2d3d[frame_count] = LineCorrespondence(lines3d[frame_count], undist_lines2d[frame_count],
                                                      K, tempRot, tempTrans, overlap_th, dist_th,
                                                      degree_th, angle_th, per_inliers, outlier_th);
    
       // ROS_INFO("the average line pair time cost: %fms", t_line_pair.toc());                                    

        TicToc t_solver;

         UpdatedOptimization(); // proposed

	   	//JointOptimization();  // benchmark

		slideWindow();

	}
	index++;

	if (index < 2)   
	{
		ROS_INFO("Starting time: %f", _time_stamp);
		Matrix3d tempRot = b2c_R.transpose() * R_w[frame_count].transpose();
		Vector3d tempTrans = -tempRot * T_w[frame_count] - b2c_R.transpose() * b2c_T;
		vector<pairsmatch> match1;

        match1 = LineCorrespondence(lines3d[frame_count], undist_lines2d[frame_count],
                                    K, tempRot, tempTrans, overlap_th, dist_th,
                                    degree_th, angle_th, per_inliers, outlier_th);

		savelines_2d3d(true);
		savematches(match1,index,delta_R[frame_count], delta_T[frame_count], false);
	}
}

// project the 3D start and end points of line into cam image
vector<line3d> estimator::UpdateLinesInFOV(Vector3d &_vio_T, Matrix3d &_vio_R)   
{
	vector<line3d> tmp_lines3d;
    tmp_lines3d.clear();

	// update transformations:
    // the output of vins: _vio_R/t is the body pose on the start point
    // b2c_R/t is the intrinsic matrix between imu and camera
    // w2gb_R/t is the start point pose on the world frame
    // R, t is the absolute pose (world frame) of camera on the current time. also is the extrinsic matrix of camera
    // transform from world frame to camera frame
    // _vio_T = _vio_T - b0_T;

	Eigen::Matrix3d R = b2c_R.transpose() * _vio_R.transpose() * w2gb_R;
	Eigen::Vector3d T = b2c_R.transpose() * (_vio_R.transpose() * (w2gb_T - _vio_T) - b2c_T);

	for (size_t i = 0; i < lines3d_map.size(); i++)
	{
		bool start_flag = false, end_flag = false;
		Eigen::Vector3d start_pt = Eigen::Vector3d(lines3d_map[i][0], lines3d_map[i][1], lines3d_map[i][2]);
		Eigen::Vector3d end_pt = Eigen::Vector3d(lines3d_map[i][3], lines3d_map[i][4], lines3d_map[i][5]);

		Eigen::Vector3d tf_start_pt = R * start_pt + T;    // projective from world frame into camera frame
		Eigen::Vector3d tf_end_pt = R * end_pt + T;

        double xx, yy, xx_, yy_;
        if(tf_start_pt[2] > 0)
        {
            xx = K(0,0) * tf_start_pt[0] / tf_start_pt[2] + K(0,2);    // image frame
            yy = K(1,1) * tf_start_pt[1] / tf_start_pt[2] + K(1,2);
            if (xx > 0 && xx < (width - 1) && yy > 0 && yy < (height - 1))     // start_point in the image plane
                start_flag = true;
        }
		if (tf_end_pt[2] > 0)
		{
            xx_ = K(0,0) * tf_end_pt[0] / tf_end_pt[2] + K(0,2);    // image frame
            yy_ = K(1,1) * tf_end_pt[1] / tf_end_pt[2] + K(1,2);
			if (xx_ > 0 && xx_ < (width - 1) && yy_ > 0 && yy_ < (height - 1))
					end_flag = true;
		}

		if (start_flag && end_flag) // both end points are in FOV
		{
			//map to vio frame, point cloud
			Vector3d pt1 = w2gb_R * start_pt + w2gb_T;     // trans from world to body origin
			Vector3d pt2 = w2gb_R * end_pt + w2gb_T;
			line3d l3d(pt1, pt2);     //
			tmp_lines3d.push_back(l3d);
		}
		else if (start_flag) //only start point is in FOV start_flag
		{
			Eigen::Vector3d dirvec = tf_end_pt - tf_start_pt;
			double t = 0.05;
			bool inFOV = true;
            bool findPointE = false;
			while (inFOV)
			{
				Eigen::Vector3d temp_tf_end_pt = tf_start_pt + t * dirvec;
				if (temp_tf_end_pt[2] > 0)
				{
					double xx = K(0, 0) * temp_tf_end_pt[0] / temp_tf_end_pt[2] + K(0, 2);
					double yy = K(1, 1) * temp_tf_end_pt[1] / temp_tf_end_pt[2] + K(1, 2);
					if (xx > 0 && xx < (width - 1) && yy > 0 && yy < (height - 1))
                    {
                        findPointE = true;
                        t += 0.05;
                    }
					else
						inFOV = false;
				}
				else
					inFOV = false;
			}                                   // step = 0.1 to find the longest 3d lines
            if(findPointE)
            {
                end_pt = start_pt + (t - 0.05) * (end_pt - start_pt);
                Vector3d pt1 = w2gb_R * start_pt + w2gb_T;
                Vector3d pt2 = w2gb_R * end_pt + w2gb_T;
                line3d l3d(pt1, pt2);
                tmp_lines3d.push_back(l3d);
            }
		}
		else if (end_flag) // only end point is in FOV end_flag
		{
			Eigen::Vector3d dirvec = tf_start_pt- tf_end_pt;
			double t = 0.1;
			bool inFOV = true;
            bool findPointS = false;
			while (inFOV)
			{
				Eigen::Vector3d temp_tf_start_pt = tf_end_pt + t * dirvec;
				if (temp_tf_start_pt[2] > 0)
				{
					double xx = K(0, 0) * temp_tf_start_pt[0] / temp_tf_start_pt[2] + K(0, 2);
					double yy = K(1, 1) * temp_tf_start_pt[1] / temp_tf_start_pt[2] + K(1, 2);
					if (xx > 0 && xx < (width - 1) && yy > 0 && yy < (height - 1))
                    {
                        findPointS = true;
                        t += 0.1;
                    }
					else
						inFOV = false;
				}
				else
					inFOV = false;
			}
            if(findPointS)
            {
                start_pt = end_pt + (t - 0.1)*(start_pt - end_pt);
                Vector3d pt1 = w2gb_R * start_pt + w2gb_T;
                Vector3d pt2 = w2gb_R * end_pt + w2gb_T;
                line3d l3d(pt1, pt2);
                tmp_lines3d.push_back(l3d);
            }
		}
		//for the cases of two end points are not in FOV, but a part of line is visible, not considered
	}

	return tmp_lines3d;
}

void estimator::showUndistortion(const string &name)
{
	int ROW = image[frame_count].rows;
	int COL = image[frame_count].cols;
	int FOCAL_LENGTH = 460;
	cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
	vector<Eigen::Vector2d> distortedp, undistortedp;
	for (int i = 0; i < COL; i++)
		for (int j = 0; j < ROW; j++)
		{
			Eigen::Vector2d a(i, j);
			Eigen::Vector3d b;
			m_camera->liftProjective(a, b);
			distortedp.push_back(a);
			undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
		}
	for (int i = 0; i < int(undistortedp.size()); i++)
	{ 
		cv::Mat pp(3, 1, CV_32FC1);
		pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
		pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
		pp.at<float>(2, 0) = 1.0;
		if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600
           && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
		{
			undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300)
			        = image[frame_count].at<uchar>(distortedp[i].y(), distortedp[i].x());
		}
	}
	cv::imwrite(name, undistortedImg);
	cv::waitKey(2);
}

vector<line2d> estimator::undistortionLine(vector<line2d> &disLines)
{
    vector<line2d> undistortedLines;
    line2d line, undis_line;
    Mat terminalPoint(2, 2, CV_32F);
    for (size_t i = 0; i < disLines.size(); ++i) //
    {
        line = disLines[i];
        terminalPoint.at<float>(0, 0) = line.ptstart[0];
        terminalPoint.at<float>(0, 1) = line.ptstart[1];
        terminalPoint.at<float>(1, 0) = line.ptend[0];
        terminalPoint.at<float>(1, 1) = line.ptend[1];
        cv::undistortPoints(terminalPoint, terminalPoint, cv_KMatrix, cv_dist, cv::Mat(), cv_KMatrix);
        undis_line.ptstart[0] = terminalPoint.at<float>(0, 0);
        undis_line.ptstart[1] = terminalPoint.at<float>(0, 1);
        undis_line.ptend[0] = terminalPoint.at<float>(1, 0);
        undis_line.ptend[1] = terminalPoint.at<float>(1, 1);
        undistortedLines.push_back(undis_line);
    }
    ROS_INFO("undistortion the key line");
    return undistortedLines;
}

vector<line2d> estimator::undistortedPoints(vector<line2d> &_lines2d)
{
	vector<line2d> _undist_lines2d;
	for (unsigned int i = 0; i < _lines2d.size(); i++)  
	{
		line2d kl = _lines2d[i];
		Eigen::Vector3d b;
        b.x() = 0;
        b.y() = 0;
        b.z() = 0;
		m_camera->liftProjective(kl.ptstart, b);  

		Eigen::Vector3d pt_start = K * b;

		Eigen::Vector3d d;
		m_camera->liftProjective(kl.ptend, d);
		Eigen::Vector3d pt_end = K * d;
		Vector4d l2d(pt_start.x() / pt_start.z(), pt_start.y() / pt_start.z(),
                     pt_end.x() / pt_end.z(), pt_end.y() / pt_end.z());

        line2d undist_line2d(l2d);
		_undist_lines2d.push_back(undist_line2d);
	}
	return _undist_lines2d;
}

void estimator:: UpdatedOptimization()
{
    Matrix3d delta_R_n[frame_count + 1];
    Vector3d delta_T_n[frame_count + 1];
    delta_R_n[frame_count] = delta_R[frame_count]; 
    delta_T_n[frame_count] = delta_T[frame_count];

    savelines_2d3d(save);  

    for (int nframe = frame_count - 1; nframe >= 0; nframe--)
    {
        delta_R_n[nframe] = delta_R[nframe] * delta_R_n[nframe + 1];
        delta_T_n[nframe] = delta_R[nframe] * delta_T_n[nframe + 1] + delta_T[nframe];
    }
    // obtain 2d-3d correspondences
    int Num_matches = 0;
    for (int nframe = frame_count; nframe >= 0; nframe--)
    {
        Num_matches += matches2d3d[nframe].size();
    }
    if (Num_matches < frame_count * per_inliers)  
    {
        FrameFlag = 0;
        fuse_pose();
        return;
    }

    // add the chi-squared test for outliers rejection
    bool TestPass = false;

    int OutliersCount = 0;
    double TD = 0.0;   

    Eigen::Quaterniond updateQuat(R_w[frame_count]);
    double states[1][7];
    states[0][0] = T_w[frame_count].x();
    states[0][1] = T_w[frame_count].y();
    states[0][2] = T_w[frame_count].z();
    states[0][3] = updateQuat.w();
    states[0][4] = updateQuat.x();
    states[0][5] = updateQuat.y();
    states[0][6] = updateQuat.z();

    while (!TestPass)
    {
        int sizeMatchFrame = matches2d3d[frame_count].size();
        if (sizeMatchFrame < per_inliers)
        {
            FrameFlag = 0;
            fuse_pose();
            return;
        }

        //ceres optimization using 2d-3d correspondences
        ceres::Problem problem;

       // ceres::LossFunction *loss_function(new ceres::HuberLoss(3.0));
        ceres::LossFunction *loss_function(new ceres::CauchyLoss(1.0));

        std::vector<ceres::CostFunction*> CostFs;

        for (int nframe = frame_count; nframe >= 0; nframe--)
        {
            for (unsigned int i = 0; i < matches2d3d[nframe].size(); i++)
            {
                Vector3d line_param(matches2d3d[nframe][i].line2dt.A,
                                    matches2d3d[nframe][i].line2dt.B, matches2d3d[nframe][i].line2dt.C);
                const Eigen::Vector3d pts_start = matches2d3d[nframe][i].line3dt.ptstart;
                const Eigen::Vector3d pts_end = matches2d3d[nframe][i].line3dt.ptend;
                LineProjectionFactor *f = new LineProjectionFactor(pts_start, pts_end, line_param,
                                                                   K, b2c_R, b2c_T, delta_R_n[nframe], delta_T_n[nframe]);
                int size_pose = 7;
                ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
                problem.AddParameterBlock(states[0], size_pose, local_parameterization);

                problem.AddResidualBlock(f, loss_function, states[0]);

                if(nframe == frame_count)
                {
                    CostFs.push_back(f);
                }
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;  // ceres::DENSE_SCHUR;
        //options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 100;
        options.num_threads = 12;
        // options.logging_type = SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        int NumNewestMatches = matches2d3d[frame_count].size();
        Eigen::MatrixXd FullJacobian, FullResidual;
        FullJacobian.resize(CostFs.size() * 2, 7); 
        FullResidual.resize(CostFs.size(), 2);

        if (CostFs.size() == 0)
        {
            FrameFlag = 0;
            fuse_pose();
            return;
        }

        for (size_t i = 0; i < CostFs.size(); ++i)
        {
            auto Factor_i = CostFs[i];
            double **parameters = new double *[1];
            parameters[0] = states[0];

            std::vector<int> BlockSizes = Factor_i->parameter_block_sizes();
            double **JacobianAnalytic = new double *[BlockSizes.size()];

            Eigen::Matrix<double, 2, 7, Eigen::RowMajor> Jacobian;
            JacobianAnalytic[0] = Jacobian.data();

            double ResidualAnalytic[2];

            Factor_i->Evaluate(parameters, ResidualAnalytic, JacobianAnalytic);
            FullJacobian.row(2 * i + 0) = Jacobian.row(0);
            FullJacobian.block<1, 7>(2 * i + 0, 0) = Jacobian.row(0);
            FullJacobian.block<1, 7>(2 * i + 1, 0) = Jacobian.row(1);

            FullResidual(1 * i + 0, 0) = sqrt(ResidualAnalytic[0]);
            FullResidual(1 * i + 0, 1) = sqrt(ResidualAnalytic[1]);

        }

        int Dim = matches2d3d[frame_count].size() * 2 * 3 - 6; 

        TD = ChiTestThreshold[Dim - 1];

        double Epsilon = 0.0;
        Eigen::MatrixXd FullResidual_ = FullResidual.transpose();
        Eigen::Map<Eigen::VectorXd> SumResidual(FullResidual_.data(), FullResidual_.size());
        Epsilon = SumResidual.transpose() * SumResidual;
        FeatureWeight = 172;
        if (Epsilon / FeatureWeight > TD)
        {
            double maxResidual;
            double maxRow, maxCol;
            maxResidual = SumResidual.maxCoeff(&maxRow, &maxCol);
            int index = int(maxRow / 2);
            Outliers.push_back(matches2d3d[frame_count][index]); 
            matches2d3d[frame_count].erase(matches2d3d[frame_count].begin() + index);
            OutliersCount++;

        }
        else     
        {   // Chi-Squared test passed

            TestPass = true;

            Eigen::Matrix3d RotOld = R_w[frame_count];
            Eigen::Vector3d TransOld = T_w[frame_count];

            // state update
            Eigen::Vector3d t(states[0][0], states[0][1], states[0][2]);
            Eigen::Quaterniond q(states[0][3], states[0][4], states[0][5], states[0][6]);
            R_w[frame_count] = q.normalized().toRotationMatrix();
            T_w[frame_count] = t;

            // ProtectionLevel Calculation
            if (PL_Calculation)
            {
                int x_direction = 0;
                int y_direction = 1;
                int z_direction = 2;
                int roll_direction = 3;
                int pitch_direction = 4;
                int yaw_direction = 5;
                XPL = ProtectionLevel(FullJacobian, x_direction, TD);
                YPL = ProtectionLevel(FullJacobian, y_direction, TD);
                ZPL = ProtectionLevel(FullJacobian, z_direction, TD);
                rollPL = ProtectionLevel(FullJacobian, roll_direction, TD);
                pitchPL = ProtectionLevel(FullJacobian, pitch_direction, TD);
                yawPL = ProtectionLevel(FullJacobian, yaw_direction, TD);

                save_PL();
            }


            Eigen::Matrix3d Rot = b2c_R.transpose() * RotOld.transpose();
            Eigen::Vector3d Trans = -Rot* TransOld - b2c_R.transpose()* b2c_T;

            Eigen::Matrix3d RotNew = b2c_R.transpose() * R_w[frame_count].transpose();
            Eigen::Vector3d TransNew = -Rot* T_w[frame_count] - b2c_R.transpose()* b2c_T;

            FrameFlag = 1;

        }
    }

    fuse_pose();
}

vector<double> estimator::ProtectionLevel(const Eigen::MatrixXd &Jacobian_, int direction, double TD)
{
    double PL, Epsilon_B, Epsilon_N;
    vector<double> ProtectionValue;
    ProtectionValue.clear();
    Eigen::MatrixXd H, inv_H, HH, S, D, A, P, Q;
    int K = 3;
    int rowJ = Jacobian_.rows();
    Eigen::MatrixXd Jacobian = Jacobian_.leftCols(6);
    Eigen::MatrixXd Weight = Eigen::MatrixXd::Identity(rowJ, rowJ) / FeatureWeight;

    H = Jacobian.transpose() * Weight * Jacobian;
    inv_H = H.inverse();
    HH = Jacobian * inv_H * Jacobian.transpose() * Weight;
    int length_HH = HH.rows();
    S = Weight * (Eigen::MatrixXd::Identity(length_HH, length_HH) - HH);
    A = Eigen::MatrixXd::Zero(1, 6);
    A(0, direction) = 1;
    D = Weight * Jacobian * inv_H * A.transpose() * A * inv_H * Jacobian.transpose() * Weight;
    int rowD = D.rows();
    double maxEig = 0.0;
    for (int i = 0; i < rowJ/2; ++i)
    {
        P = Eigen::MatrixXd::Zero(rowD, 2);
        P(2 * i, 0) = 1;
        P(2 * i + 1, 1) = 1;
        Q = P.transpose() * D * P * (P.transpose() * S * P).inverse();
        Eigen::VectorXcd Eig = Q.eigenvalues();
        Eigen::VectorXd EigReal = Eig.real();
        double eigrow, eigcol;
        double maxeig = EigReal.maxCoeff(&eigrow, &eigcol);

        if (maxEig < maxeig)
        {
            maxEig = maxeig;
        }
    }
    Epsilon_B = sqrt(maxEig * TD);
    Epsilon_N = K * sqrt(inv_H(direction, direction));  // ThreeSigma
    PL = Epsilon_B + Epsilon_N;

    ProtectionValue.push_back(PL);
    ProtectionValue.push_back(Epsilon_B);
    ProtectionValue.push_back(Epsilon_N);
    ProtectionValue.push_back(maxEig);
    return ProtectionValue;
}


void estimator::JointOptimization()
{

	Matrix3d delta_R_n[frame_count + 1];
	Vector3d delta_T_n[frame_count + 1];
	delta_R_n[frame_count] = delta_R[frame_count];  // delta_R/T between frames from vins result
	delta_T_n[frame_count] = delta_T[frame_count];
	
	savelines_2d3d(save);

    for (int nframe = frame_count - 1; nframe >= 0; nframe--)
    {
    //    ROS_INFO("obtain %d", nframe);
        delta_R_n[nframe] = delta_R[nframe] * delta_R_n[nframe + 1];
        delta_T_n[nframe] = delta_R[nframe] * delta_T_n[nframe + 1] + delta_T[nframe];

    }

    // obtain 2d-3d correspondences
    int Num_matches = 0;
    for (int nframe = frame_count; nframe >= 0; nframe--)
    {
        Num_matches += matches2d3d[nframe].size();
    }


    if (save)
    {
        savematches(matches2d3d[frame_count], frame_count, delta_R_n[frame_count], delta_T_n[frame_count], false);
    }

    if (Num_matches < frame_count * per_inliers)//current frame feature is not stable, skip, use the vio pose
    {
        ROS_WARN("feature matching is not enough");

        fuse_pose();
        return;
    }

    //ceres optimization using 2d-3d correspondences
    ceres::Problem problem;
    Eigen::Quaterniond updateQuat(R_w[frame_count]);
    std::vector<double> ceres_rotation = std::vector<double>({updateQuat.w(), updateQuat.x(), updateQuat.y(), updateQuat.z()});
    std::vector<double> ceres_translation = std::vector<double>({T_w[frame_count].x(), T_w[frame_count].y(), T_w[frame_count].z()});
    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;

    ceres::LossFunction *loss_func(new ceres::HuberLoss(3.0));

    double states[1][7];
    states[0][0] = T_w[frame_count].x();
    states[0][1] = T_w[frame_count].y();
    states[0][2] = T_w[frame_count].z();
    states[0][3] = updateQuat.w();
    states[0][4] = updateQuat.x();
    states[0][5] = updateQuat.y();
    states[0][6] = updateQuat.z();


    for (int nframe = frame_count; nframe >= 0; nframe--)
    {

        for (size_t i = 0; i < matches2d3d[nframe].size(); i++)
        {
            Vector3d param2d(matches2d3d[nframe][i].line2dt.A / matches2d3d[nframe][i].line2dt.A2B2,
                             matches2d3d[nframe][i].line2dt.B / matches2d3d[nframe][i].line2dt.A2B2, matches2d3d[nframe][i].line2dt.C / matches2d3d[nframe][i].line2dt.A2B2);
            ceres::CostFunction *cost_function =
                    RegistrationError::Create(param2d, matches2d3d[nframe][i].line3dt.ptstart, matches2d3d[nframe][i].line3dt.ptend, K,
                                              b2c_R, b2c_T, delta_R_n[nframe], delta_T_n[nframe]);

            int size_pose = 7;
            ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
            problem.AddParameterBlock(states[0], size_pose, local_parameterization);
            problem.AddResidualBlock(
                    cost_function,
                    loss_func,
                    states[0]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    options.num_threads = 12;
    // options.logging_type = SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << "\n";
    Eigen::Vector3d t(states[0][0], states[0][1], states[0][2]);
    Eigen::Quaterniond q(states[0][3], states[0][4], states[0][5], states[0][6]);
    R_w[frame_count] = q.normalized().toRotationMatrix();    // state update
    T_w[frame_count] = t;

    if (save )
    {
        savematches(matches2d3d[frame_count], frame_count, delta_R_n[frame_count], delta_T_n[frame_count], true);
    }
 
    Matrix3d tempRot = b2c_R.transpose() * R_w[frame_count].transpose();
    Vector3d tempTrans = -tempRot * T_w[frame_count] - b2c_R.transpose() * b2c_T;

    matches2d3d[frame_count] = LineCorrespondence(lines3d[frame_count], undist_lines2d[frame_count],
                                                  K, tempRot, tempTrans, overlap_th, dist_th,
                                                  degree_th, angle_th, per_inliers, outlier_th);
	fuse_pose();
}

void estimator::save_PL()
{
    ofstream foutPL(PL_PATH, ios::app);
    foutPL.setf(ios::fixed, ios::floatfield);
    foutPL.precision(0);
    foutPL << time_stamp[frame_count] * 1e9 << ",";
    foutPL.precision(5);
    foutPL << XPL[0] << "," << YPL[0] << "," << ZPL[0] << "," << rollPL[0] << "," << pitchPL[0] << "," << yawPL[0] << endl;
    foutPL.close();
}

void estimator::fuse_pose()
{
	Eigen::Matrix3d deR = R_w[frame_count - 1] * R_w[frame_count].transpose();
	Eigen::Vector3d deT = T_w[frame_count - 1]- deR * T_w[frame_count];
	
	if ((deT-delta_T[frame_count - 1]).norm()>0.8)  
	{
		ROS_WARN("correspondence error...");
		R_w[frame_count] = delta_R[frame_count - 1].transpose() * R_w[frame_count - 1];
        T_w[frame_count] = delta_R[frame_count - 1].transpose() * (T_w[frame_count-1]-delta_T[frame_count - 1]);
	}

	ofstream foutC(TRACK_RESULT_PATH, ios::app);
	foutC.setf(ios::fixed, ios::floatfield);
	foutC.precision(0);
	foutC << time_stamp[frame_count] * 1e9 << ",";
	foutC.precision(5);
	Eigen::Quaterniond tmp_Q(R_w[frame_count]);

    Eigen::Matrix3d tmp_R = R_w[frame_count];

    Eigen::Vector3d ypr = Utility::R2ypr(tmp_R);
	foutC << T_w[frame_count].x() << ","
		  << T_w[frame_count].y() << ","
		  << T_w[frame_count].z() << ","
		  << tmp_Q.w() << ","
		  << tmp_Q.x() << ","
		  << tmp_Q.y() << ","
		  << tmp_Q.z() << ","
          << ypr(2) / 180.0 * M_PI << ","      // alpha
          << ypr(1) / 180.0 * M_PI << ","      // beta
          << ypr(0) / 180.0 * M_PI << ","      // gamma
          << FrameFlag << endl;                  
	foutC.close();
}

void estimator::slideWindow()
{
	for (size_t i = 0; i < WINDOW_SIZE; i++)
	{
		time_stamp[i] = time_stamp[i+1];
		vio_T[i] = vio_T[i+1];
		vio_R[i] = vio_R[i+1];
		delta_R[i]= delta_R[i+1];
		delta_T[i]= delta_T[i+1];
		T_w[i] = T_w[i+1];
		R_w[i] =R_w[i+1];
		undist_lines2d[i] = undist_lines2d[i+1];
		lines3d[i] = lines3d[i+1];
		matches2d3d[i] = matches2d3d[i+1];
	}
}

void estimator::savematches(const vector<pairsmatch> &matches, int &frame,
							 Matrix3d &delta_R_i, Vector3d &delta_t_i, const bool &optimized)
{
	ROS_INFO("Number of matches: %d", int(matches.size()));
	char filename2d[100];
	if (optimized)
		sprintf(filename2d, "optmlines2d_%d_%d.txt", index, frame);
	else
	{
		sprintf(filename2d, "bfmlines2d_%d_%d.txt", index, frame);
	}
	ofstream out2d(filename2d);
	char filename3d[100];
	if (optimized)
		sprintf(filename3d, "optmlines3d_%d_%d.txt", index, frame);
	else
	{
		sprintf(filename3d, "bfmlines3d_%d_%d.txt", index, frame);
	}
	ofstream out3d(filename3d);

	Matrix3d R_w_n = delta_R_i * R_w[frame_count];
	Vector3d T_w_n = delta_R_i * T_w[frame_count] + delta_t_i;
	Matrix3d R = b2c_R.transpose() * R_w_n.transpose();
	Vector3d t = -R * T_w_n - b2c_R.transpose() * b2c_T;
	double sum_error=0;
	for (size_t i = 0; i < matches.size(); i++)
	{
		line3d l3d=matches[i].line3dt;
		pairsmatch match(matches[i].index, matches[i].line2dt, l3d.transform3D(R,t));
		auto start_pt2d = match.line2dt.ptstart;
		out2d << start_pt2d.x() << " " << start_pt2d.y() << " ";
		auto end_pt2d = match.line2dt.ptend;
		out2d << end_pt2d.x() << " " << end_pt2d.y() << " ";
		out2d << "\n";

		line2d p_l2d = match.line3dt.project3D(K);
		start_pt2d = p_l2d.ptstart;
		out3d << start_pt2d.x() << " " << start_pt2d.y() << " ";
		end_pt2d = p_l2d.ptend;
		out3d << end_pt2d.x() << " " << end_pt2d.y() << " ";
		out3d << match.distance.x() << " " << match.distance.y() << " " << match.distance.z();
		out3d << "\n";

		sum_error+=(match.calcEulerDist(K, 0.2)).x();
	}

	out2d.close();
	out3d.close();
	if(optimized)
	{
		ROS_INFO("after sum: %f, mean: %f", sum_error, sum_error / matches.size());
		ROS_INFO("------------------------------------------");
	}
	else
	{
		ROS_INFO("before sum: %f, mean: %f", sum_error, sum_error/matches.size());
	}
}

//save 2d and 3d lines for a local frame and camera pose, and image frames, the last frame in slidingwindow
void estimator::savelines_2d3d(const bool &save) 
{
	if (save)
	{
		//save camera poses
		Eigen::Matrix3d R = b2c_R.transpose() * R_w[frame_count].transpose();
		Eigen::Vector3d T = -R * T_w[frame_count] - b2c_R.transpose() * b2c_T;
		Eigen::Quaterniond quat(R);
		
		string pose_file = "estimator_poses.csv";
		ofstream outpose;
		if (index==0)
			outpose.open(pose_file);
		else
			outpose.open(pose_file, ios::app);
		if (outpose)
		{
			outpose << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << "," << T.x() << "," << T.y() << "," << T.z() << "\n";
		}
		outpose.close();
		//save 2d lines
		char filename2d[100];
		sprintf(filename2d, "lines2d_%d.txt", index);
		ofstream out2d(filename2d);
		for (size_t i = 0; i < undist_lines2d[frame_count].size(); i++)
		{
			auto start_pt2d = undist_lines2d[frame_count][i].ptstart;
			out2d << start_pt2d.x() << " " << start_pt2d.y() << " ";
			auto end_pt2d = undist_lines2d[frame_count][i].ptend;
			out2d << end_pt2d.x() << " " << end_pt2d.y();
			out2d << "\n";
		}
		out2d.close();

		//save 3d lines
		char filename3d[100];
		sprintf(filename3d, "lines3d_%d.txt", index);
		ofstream out3d(filename3d);
		for (size_t i = 0; i < lines3d[frame_count].size(); i++)
		{
			auto start_pt3d = lines3d[frame_count][i].ptstart;
			auto tf_start_pt = K * (R * start_pt3d + T);
			out3d << tf_start_pt.x() / tf_start_pt.z() << " " << tf_start_pt.y() / tf_start_pt.z() << " " << 1 << " ";
			auto end_pt3d = lines3d[frame_count][i].ptend;
			auto tf_end_pt = K * (R * end_pt3d + T);
			out3d << tf_end_pt.x() / tf_end_pt.z() << " " << tf_end_pt.y() / tf_end_pt.z() << " " << 1;
			out3d << "\n";
		}
		out3d.close();

		//save images
		char imgname[100];
		sprintf(imgname, "img_%d.jpg", index);
		cv::imwrite(imgname, image[frame_count]);
		if (index < 0)
			ROS_INFO("%d image time=%lf, lines3d size: %d, lines2d size: %d", index, time_stamp[frame_count], int(lines3d[frame_count].size()), int(lines2d[frame_count].size()));
	}
}
