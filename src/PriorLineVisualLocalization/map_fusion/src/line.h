#ifndef _LINE_H_
#define _LINE_H_
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <vector>
#include <ros/ros.h>
#include <math.h>

using namespace std;

#define PI 3.1415926
typedef Eigen::Matrix< double, 6, 1 > 	Vector6d;

class line2d
{
public:
    Eigen::Vector2d ptstart, ptend, ptmid, direction, subvec;
    double length;
    Eigen::Vector3d hptstart, hptend;
    double A, B, C, A2B2;
    bool horizon;
    public:
    line2d(){};
    ~line2d(){};
    line2d(const Eigen::Vector4d &vec);
    Eigen::Vector3d ComputeCircleNormal(const Eigen::Matrix3d &K);
    // get the leatest point on a finite line to the point
    Eigen::Vector2d point2flined(const Eigen::Vector2d & mdpt);
};

class line3d
{
public:
    Eigen::Vector3d ptstart, ptend, ptmid, direction;
    double length;
    Eigen::Vector4d hptstart, hptend;
    
    public:
    line3d(){};
    
    line3d(const Vector6d &vec);
    line3d(const Eigen::Vector3d &_ptstart, const Eigen::Vector3d &_ptend);
    line3d transform3D(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);
    line2d project3D(const Eigen::Matrix3d &K);

};

class pairsmatch
{
public:
    int index;
    line2d line2dt;
    line3d line3dt;
    line2d line3dto2d;
    Vector6d criteria;
    Eigen::Vector3d distance;

    double weight;  
    bool repetitive = false;

public:
    pairsmatch(int indx, line2d tmp2d, line3d tmp3d);
    // Follow Brown, Mark, David Windridge, and Jean-Yves Guillemaut.
    //"A family of globally optimal branch-and-bound algorithms for 2D–3D correspondence-free registration."
    // Pattern Recognition 93 (2019): 36-54.
    Eigen::Vector3d calcAngleDist(const Eigen::Matrix3d &K, const double & lambda);
    // sum of 3D end points projections to 2D line distance 
    Eigen::Vector3d calcEulerDist(const Eigen::Matrix3d &K, const double & theta);

    void setZero();

    bool isZero();

    void printmsg();
};

//reject outliers
bool rejectoutliers(std::vector<pairsmatch> &matches,
                    const Eigen::Matrix3d &K,
                    const Eigen::Matrix3d &R,
                    const Eigen::Vector3d &t,
                    double &lambda, double &outlier_threshold,
                    bool &UseAngleDist);

//update correspondence from the oringinal 2D and 3D lines
std::vector<pairsmatch> updatecorrespondence(std::vector<line3d> &lines_3d,
                                             std::vector<line2d> &lines_2d,
                                             const Eigen::Matrix3d &K,
                                             const Eigen::Matrix3d &R,
                                             const Eigen::Vector3d &t,
                                             double &theta_th, double &outlier_threshold);

std::vector<pairsmatch> CorrespondenceFilter(std::vector<pairsmatch> &InputMatches,
                                             double &overlap_th, double &dist_th, double &degree_th,
                                             double &angle_th, double &outlier_th, int &min_matches);

std::vector<pairsmatch> CorrespondenceFilter_original(std::vector<pairsmatch> &InputMatches,
                                                      double &overlap_th, double &dist_th, double &degree_th,
                                                      double &angle_th, double &outlier_th, int &per_inliers);

std::vector<pairsmatch> LineCorrespondence(std::vector<line3d> &lines_3d,
                                           std::vector<line2d> &lines_2d,
                                           const Eigen::Matrix3d &K,
                                           const Eigen::Matrix3d &R,
                                           const Eigen::Vector3d &t,
                                           double &overlap_th, double &dist_th,
                                           double &degree_th, double &angle_th,
                                           int &per_inliers, double &outlier_th);

std::vector<pairsmatch> LineCorrespondenceV2(std::vector<line3d> &lines_3d,
                                            std::vector<line2d> &lines_2d,
                                            const Eigen::Matrix3d &K,
                                            const Eigen::Matrix3d &R,
                                            const Eigen::Vector3d &t,
                                            double &overlap_th, double &dist_th,
                                            double &degree_th, double &angle_th,
                                            int &per_inliers, double &outlier_th);


Eigen::Vector4d CalAngleDist(line2d &lineM, line3d &lineE,
                            const Eigen::Matrix3d &K, double &lambda, double &degree_th);

Eigen::Vector2d CalEulerDist(line2d &lineM, line2d &linePro, double &overlap_th);

// compute the distribution of the matched 3D lines in 3D space and give the confidence, 
double computedistribution(const std::vector<pairsmatch> &matches);

bool Compare(pairsmatch pair1, pairsmatch pair2);

#endif