
#ifndef INC_2D_3D_POSE_TRACKING_MASTER_LINE_PROJECTION_FACTOR_H
#define INC_2D_3D_POSE_TRACKING_MASTER_LINE_PROJECTION_FACTOR_H

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "utility.h"
#include "tic_toc.h"

class LineProjectionFactor: public ceres::SizedCostFunction<2, 7> // 2: residual; 7: state
{
public:
    LineProjectionFactor(const Eigen::Vector3d &_pts_start, const Eigen::Vector3d &_pts_end, const Eigen::Vector3d &_line_param,
                         const Eigen::Matrix3d _K, const Eigen::Matrix3d _b_c_R, const Eigen::Vector3d _b_c_T,
                         const Eigen::Matrix3d _delta_R, const Eigen::Vector3d _delta_T);  // state, measurements and intermediate parameters

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    // the calculated parameters
    Eigen::Vector3d pts_start;
    Eigen::Vector3d pts_end;
    Eigen::Vector3d line_param;
    Eigen::Matrix3d K;
    Eigen::Matrix3d b_c_R;
    Eigen::Vector3d b_c_T;
    Eigen::Matrix3d delta_R;
    Eigen::Vector3d delta_T;

    static double sum_t;
};


#endif //INC_2D_3D_POSE_TRACKING_MASTER_LINE_PROJECTION_FACTOR_H
