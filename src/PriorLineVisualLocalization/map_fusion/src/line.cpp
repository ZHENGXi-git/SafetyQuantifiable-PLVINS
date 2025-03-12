#include "line.h"
#include <iostream>

using namespace std;
using namespace Eigen;
line2d::line2d(const Eigen::Vector4d &vec)
{
    ptstart = Vector2d(vec[0], vec[1]);
    ptend = Vector2d(vec[2], vec[3]);
    ptmid = (ptstart + ptend) / 2;
    subvec = ptend - ptstart;
    length = subvec.norm();
    direction = subvec / length;
    hptstart = Vector3d(ptstart[0],ptstart[1], 1);
    hptend = Vector3d(ptend[0],ptend[1], 1);
    A = ptend[1] - ptstart[1];
    B = ptstart[0] - ptend[0];
    C = ptend[0] * ptstart[1] - ptstart[0] * ptend[1];
    A2B2 = sqrt(A * A + B * B);
    double gradient = atan2(ptstart[1] - ptend[1], ptstart[0] - ptend[0]) / PI * 180;
    if ((0 <= abs(gradient) <= 50) || ( 130 <= abs(gradient) <= 180))
    {
        horizon = true;
    }
    if(40 <= abs(gradient) <= 140)
    {
        horizon = false;
    }
}

Eigen::Vector3d line2d::ComputeCircleNormal(const Eigen::Matrix3d &K)
{
    Vector3d p1_ = K.inverse() * hptstart;
    Vector3d p2_ = K.inverse() * hptend;
    MatrixXd A(3, 4);
    A << 0, 0, 0, 1,
        p1_.transpose(), 1,
        p2_.transpose(), 1;
    EigenSolver<MatrixXd> es(A.transpose() * A);
    Vector4d v = es.eigenvectors().col(2).real();
    if (es.eigenvalues()[2].real() > es.eigenvalues()[3].real())
        v = es.eigenvectors().col(3).real();
    Vector3d normal(v[0], v[1], v[2]);
    return normal.normalized();
}

Eigen::Vector2d line2d::point2flined(const Eigen::Vector2d &mdpt)    //  find mdpt's the closet point on line2d
{
    Vector2d tmpstart = mdpt - ptstart;
    double d1 = tmpstart.norm();
    Vector2d tmpend = mdpt - ptend;
    double d2 = tmpend.norm();
 
    // line2 Bx-Ay+C2=0  the vertical line of line2d
    double A2, B2, C2;
    A2 = B;
    B2 = -A;
    C2 = -(A2 * mdpt[0] + B2 * mdpt[1]);
    Matrix2d Cof;
    Cof << A, B, A2, B2;
    Vector2d intersection = Cof.inverse() * Vector2d(-C, -C2);  // the intersection between line2d and line2
    if((intersection.x()-ptstart.x())*(intersection.x()-ptend.x())>=0)
    {
        if (d1 < d2) return ptstart;
        else
          return ptend;
    }
    else
    {
        return intersection;
    }
}

line3d::line3d(const Vector6d &vec)
{
    ptstart = Vector3d(vec[0], vec[1], vec[2]);
    ptend = Vector3d(vec[3], vec[4], vec[5]);
    ptmid = (ptstart + ptend) / 2;
    Vector3d temp = ptend - ptstart;
    length = temp.norm();
    direction = temp / length;
}

line3d::line3d(const Eigen::Vector3d &_ptstart, const Eigen::Vector3d &_ptend)
{
    ptstart = _ptstart;
    ptend = _ptend;
    ptmid = (ptstart + ptend) / 2;
    Vector3d temp = ptend - ptstart;
    length = temp.norm();
    direction = temp / length;
}
line3d line3d::transform3D(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    line3d transformedLine3d;
    Vector3d tempstart = R * ptstart + t;
    Vector3d tempend = R * ptend + t;

    Vector6d tempvec;
    tempvec.block(0,0,3,1)=tempstart;
    tempvec.block(3,0,3,1)=tempend;
    line3d transform3dline(tempvec);

    return transform3dline;
}

line2d line3d::project3D(const Eigen::Matrix3d &K)
{
    Vector3d tmpstart = K * ptstart;
    double v1= tmpstart[0] / tmpstart[2];
    double v2= tmpstart[1] / tmpstart[2];
    Vector3d tmpend = K * ptend;
    double v3= tmpend[0] / tmpend[2];
    double v4= tmpend[1] / tmpend[2];
    line2d projectedline(Vector4d(v1,v2,v3,v4));
    return projectedline;
}

pairsmatch::pairsmatch(int indx, line2d tmp2d, line3d tmp3d)
{
    index = indx;
    line2dt = tmp2d;
    line3dt = tmp3d;
    distance << 2000,1000,1000;
    criteria << 2000,1000,1000, 1000, 1000, 1000;
}

bool GreaterSort(pairsmatch pair1,pairsmatch pair2 )
{
    return (pair1.distance.z()>pair2.distance.z());     // from big to small
}

bool Compare(pairsmatch pair1, pairsmatch pair2)
{
    return (pair1.criteria[0] < pair2.criteria[0]);   // from small to big
}

// Follow Brown, Mark, David Windridge, and Jean-Yves Guillemaut. 
//"A family of globally optimal branch-and-bound algorithms for 2D–3D correspondence-free registration."
// Pattern Recognition 93 (2019): 36-54.
Eigen::Vector3d pairsmatch::calcAngleDist(const Eigen::Matrix3d &K, const double & lambda)
{
    // the angle distance between two orientations
    Eigen::Vector3d circleNormal;
    circleNormal=line2dt.ComputeCircleNormal(K);
    double theta = abs(PI/2-acos(abs(line3dt.direction.transpose()*circleNormal)));

    if(theta>0.12) return Vector3d{PI, PI, 0}; //if the inter angle is too large, they should not be correspondences
    //mid 2D point to the latest point angle of 3D line projection (finite 3D line)
    Vector3d hmid = Vector3d(line2dt.ptmid[0],line2dt.ptmid[1],1);
    Vector3d bearing_vector = (K.inverse()*hmid).normalized();  // reproject to camera frame

    line2d project2dline = line3dt.project3D(K);
    // add penalty for large length error and non-overlap
    if (project2dline.length < 0.5 * line2dt.length)
        return Vector3d{PI, PI, 0};
    if(line2dt.point2flined(project2dline.ptstart) == line2dt.point2flined(project2dline.ptend))  //no overlap
    {
        return Vector3d{PI, PI, 0};
    }    
    
    Vector2d interpt=project2dline.point2flined(line2dt.ptmid);
    Vector3d hinterpt=Vector3d(interpt[0],interpt[1],1);
    Vector3d bearing_vector_interpt=(K.inverse()*hinterpt).normalized();
    double phi=acos(abs(bearing_vector.transpose()*bearing_vector_interpt));

    double dist = lambda*theta+(1-lambda)*phi;
    return Vector3d{dist, theta, phi};
}

Eigen::Vector4d CalAngleDist(line2d &lineM, line3d &lineE,    // lineE on the camera frame
                            const Eigen::Matrix3d &K, double &lambda,
                            double &degree_th)
{
    // theta: the angle on camera frame
    // phi: the midpoint and nearest angle
    // beta: the angle on image frame
    // alpha: the weighted angle
    double alpha, theta, phi, beta;
    Eigen::Vector4d MatchFactorA = Vector4d{PI, PI, PI, PI}; 
    Eigen::Vector3d circleNormal = lineM.ComputeCircleNormal(K);

    theta = abs(PI / 2 - acos(abs(lineE.direction.transpose() * circleNormal)));

    line2d projectedline = lineE.project3D(K);

    beta = acos(abs(lineM.direction.transpose() * projectedline.direction));

    Vector2d nearestP = projectedline.point2flined(lineM.ptmid);
    Vector3d nearestPN = Vector3d(nearestP[0], nearestP[1], 1);
    Vector3d NearP = (K.inverse() * nearestPN).normalized();
    Vector3d midP = Vector3d(lineM.ptmid[0], lineM.ptmid[1], 1);
    Vector3d MidP = (K.inverse() * midP).normalized();
    phi = acos(abs(NearP.transpose() * MidP));

    double lambda2 = lambda;
    alpha = (lambda / 2) * theta + (1 - lambda) * phi + (lambda2 / 2) * beta;

    MatchFactorA = Vector4d{alpha, theta, phi, beta};

    if (isnan(alpha) || isnan(theta) || isnan(phi) || isnan(beta))
        MatchFactorA = Vector4d{PI, PI, PI, PI};

    return MatchFactorA;

}
Eigen::Vector2d CalEulerDist(line2d &lineM, line2d &linePro,
                            double &overlap_th)
{
    Vector2d MatchFactorD;  // point to line dists, angle, overlap ratio
    int sampleLen = 20;
    int sampleNum = 10;
    double lengthM = lineM.length;
    double lengthP = linePro.length;
    double lengthMin;
    double distance = 10000.0;
    double overlap_ratio1 = 0.0, overlap_ratio2 = 0.0, overlap_ratio = 0.0;
    double angle = PI;

    MatchFactorD = Vector2d{distance, overlap_ratio};

    line2d line1, line2;  // line1 = short, line2 = long
    if (lengthM <= lengthP)
    {
        lengthMin = lengthM;
        line1 = lineM;
        line2 = linePro;
    }
    else
    {
        lengthMin = lengthP;
        line1 = linePro;
        line2 = lineM;
    }
    //sampleNum = int(lengthMin / sampleLen);

    overlap_ratio1 = ( (line2.point2flined(line1.ptstart)-line2.point2flined(line1.ptend)).norm()) / line2.length;
  //  overlap_ratio2 = ((line1.point2flined(line2.ptstart)-line1.point2flined(line2.ptend)).norm()) / line1.length;
  //  overlap_ratio = (overlap_ratio1 + overlap_ratio2) / 2;
    overlap_ratio = overlap_ratio1;

    double point_x = line1.ptstart[0], point_y = line1.ptstart[1];
    double len_x = line1.ptstart[0] - line1.ptend[0];
    double len_y = line1.ptstart[1] - line1.ptend[1];
    double step_x = len_x / sampleNum;
    double step_y = len_y / sampleNum;

    for (size_t i = 1; i < sampleNum; ++i)
    {
        double x = point_x + i * step_x;
        double y = point_y + i * step_y;
        distance = distance +  abs(line2.A * x + line2.B * y + line2.C) / line2.A2B2; 
    }
    distance = distance + 3 * abs(line2.A * line1.ptstart[0] + line2.B * line1.ptstart[1] + line2.C) / line2.A2B2;
    distance = distance + 3 * abs(line2.A * line1.ptend[0] + line2.B * line1.ptend[1] + line2.C) / line2.A2B2;

    distance = distance / (sampleNum + 5);

    MatchFactorD = Vector2d{distance, overlap_ratio};
    if (isnan(distance) || isnan(overlap_ratio))
        MatchFactorD = Vector2d{10000.0, 0.0};

    return MatchFactorD;
}

Eigen::Vector3d pairsmatch::calcEulerDist(const Eigen::Matrix3d &K, const double & theta)
{
    double d1 = 500, d2 = 500, dist = 1000;
    line2d project2dline = line3dt.project3D(K);   // projection 3d line (in the camera frame) into image plane
    
    double angle = acos(abs(line2dt.direction.transpose()*project2dline.direction));   //
    double overlap_dist = 0;

    if (angle < theta) 
    {
        // add penalty for large length error and non-overlap
        if (project2dline.length < 0.5 * line2dt.length)
            return Vector3d{dist, 3.14, 0};
        //overlap distance
        overlap_dist=(line2dt.point2flined(project2dline.ptstart)-line2dt.point2flined(project2dline.ptend)).norm();
        if(overlap_dist<0.5*min(project2dline.length, line2dt.length))
            return Vector3d{dist, 3.14, 0};
        d1 = abs(line2dt.A * project2dline.ptstart[0] + line2dt.B * project2dline.ptstart[1] + line2dt.C) / line2dt.A2B2;
        
        d2 = abs(line2dt.A * project2dline.ptend[0] + line2dt.B * project2dline.ptend[1] + line2dt.C) / line2dt.A2B2;
        
        dist = d1 + d2;
    }
    return Vector3d{dist, angle, overlap_dist};
}
void pairsmatch::printmsg()
{
    cout<<index<<"-th pairs, 3d("<<line3dt.ptstart[0]<<","<<line3dt.ptstart[1]<<","<<line3dt.ptstart[2]<<","<<line3dt.ptstart[3]<<","<<line3dt.ptstart[4]<<","<<line3dt.ptstart[5]<<")"<<endl;
    cout<<"2d("<<line2dt.ptstart[0]<<","<<line2dt.ptstart[1]<<","<<line2dt.ptend[0]<<","<<line2dt.ptend[1]<<")"<<endl;
}

//reject outliers
bool rejectoutliers(std::vector<pairsmatch> &matches,
                    const Eigen::Matrix3d &K,
                    const Eigen::Matrix3d &R,
                    const Eigen::Vector3d &t, double &lambda, double &outlier_threshold,
                    bool &UseAngleDist)
{
    vector<pairsmatch> updatematches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        Vector3d dist;
        line3d tmpline3d = matches[i].line3dt;
        pairsmatch tfmatch(i, matches[i].line2dt, tmpline3d.transform3D(R, t));    // tmpline3d.transform3D(R, t): return the 3d line on camera frame
        if (UseAngleDist)
            dist = tfmatch.calcAngleDist(K, lambda);
        else
        {
            dist = tfmatch.calcEulerDist(K, lambda);
        }

        if (dist.x() <= outlier_threshold)
        {
            matches[i].distance=dist;
            updatematches.push_back(matches[i]);
        }
    }
    bool hasoutliers = false;
    if (updatematches.size() < matches.size())
        hasoutliers = true;
    else
    {
        hasoutliers = false;
    }
    matches=updatematches;
    return hasoutliers;
}

void pairsmatch::setZero()
{
    line2dt.ptstart(0) = 0.0;
    line2dt.ptstart(1) = 0.0;
    line2dt.ptend(0) = 0.0;
    line2dt.ptend(1) = 0.0;
    line3dto2d.ptstart(0) = 0.0;
    line3dto2d.ptstart(1) = 0.0;
    line3dto2d.ptend(0) = 0.0;
    line3dto2d.ptend(1) = 0.0;
}

bool pairsmatch::isZero()
{
    if (line2dt.ptstart(0) == 0.0 && line2dt.ptstart(1) == 0.0 && line2dt.ptend(0) == 0.0 && line2dt.ptend(1) == 0.0
     && line3dto2d.ptstart(0) == 0.0 && line3dto2d.ptstart(1) == 0.0 && line3dto2d.ptend(0) == 0.0 && line3dto2d.ptend(1) == 0.0 )
        return true;
    else
        return false;
}

std::vector<pairsmatch> CorrespondenceFilter(std::vector<pairsmatch> &InputMatches,
                                             double &overlap_th, double &dist_th, double &degree_th,
                                             double &angle_th, double &outlier_th, int &per_inliers)
{
    vector<pairsmatch> OutputMatches;
    OutputMatches.clear();
    double AveAngle = 0.0;
    double AveDist = 0.0;
    double AveOverlap = 0.0;

    // delete repetitive pairs
    vector<pairsmatch> InputMatches_ = InputMatches;
    int RepetitiveCount = 0;

    for (int i = 0; i < InputMatches_.size(); ++i)
    {
        line2d line_pro = InputMatches_[i].line3dto2d;
        Eigen::Vector2d startP = line_pro.ptstart;
        Eigen::Vector2d endP = line_pro.ptend;
        if(!InputMatches_[i].repetitive)
        {
            for (int j = i + 1; j < InputMatches_.size(); ++j)
            {
                line2d line_pro_ = InputMatches_[j].line3dto2d;
                Eigen::Vector2d startP_ = line_pro_.ptstart;
                Eigen::Vector2d endP_ = line_pro_.ptend;
                if (startP(0) == startP_(0) && startP(1) == startP_(1) && endP(0) == endP_(0) && endP(1) == endP_(1))
                {
                    // judge the criteria
                    double dist = InputMatches_[i].criteria[0];
                    double dist_ = InputMatches_[j].criteria[0];
                    if (dist > dist_)
                    {
                        InputMatches_[i].repetitive = true;
                    }
                    else
                    {
                        InputMatches_[j].repetitive = true;
                    }
                    RepetitiveCount += 1;
                }
            }
        }
    }

    for (int i = 0; i < RepetitiveCount; ++i)
    {
        for (int j = 0; j < InputMatches_.size(); ++j)
        {
            if (InputMatches_[j].repetitive)
            {
                InputMatches_.erase(InputMatches_.begin() + j);
                break;
            }
        }
    }
    //  need sort
    sort(InputMatches_.begin(), InputMatches_.end(), Compare);

    for (size_t i = 0; i < InputMatches_.size(); ++i)
    {
        AveAngle += InputMatches_[i].criteria[5];
        AveDist += InputMatches_[i].criteria[0];
        AveOverlap += InputMatches_[i].criteria[1];
    }
    AveAngle = AveAngle / InputMatches_.size();
    AveDist = AveDist / InputMatches_.size();
    AveOverlap = AveOverlap / InputMatches_.size();

    double Dist_in = AveDist * outlier_th;
    double Angele_in = AveAngle * outlier_th;
    double Overlap_in = AveOverlap * outlier_th;

    for (size_t i = 0; i < InputMatches_.size(); ++i)
    {
        if(InputMatches_[i].criteria[0] <= Dist_in * 1.6  && InputMatches_[i].criteria[5] <= Angele_in * 1.6)
        {
            OutputMatches.push_back(InputMatches_[i]);
        }
    }

    int MaxMatch = 60;
    if (OutputMatches.size() > MaxMatch)
    {
        OutputMatches.erase(OutputMatches.begin() + MaxMatch, OutputMatches.end());
    }

    return OutputMatches;
}

std::vector<pairsmatch> CorrespondenceFilter_original(std::vector<pairsmatch> &InputMatches,
                                             double &overlap_th, double &dist_th, double &degree_th,
                                             double &angle_th, double &outlier_th, int &per_inliers)
{
    vector<pairsmatch> OutputMatches;
    OutputMatches.clear();
    double AveAngle = 0.0;
    double AveDist = 0.0;
    double AveOverlap = 0.0;
    //  need sort
    sort(InputMatches.begin(), InputMatches.end(), Compare);

    for (size_t i = 0; i < InputMatches.size(); ++i)
    {
        AveAngle += InputMatches[i].criteria[5];
        AveDist += InputMatches[i].criteria[0];
        AveOverlap += InputMatches[i].criteria[1];
    }
    AveAngle = AveAngle / InputMatches.size();
    AveDist = AveDist / InputMatches.size();
    AveOverlap = AveOverlap / InputMatches.size();

    double Dist_in = AveDist * outlier_th;
    double Angele_in = AveAngle * outlier_th;
    double Overlap_in = AveOverlap * outlier_th;

    for (size_t i = 0; i < InputMatches.size(); ++i)
    {
        if(InputMatches[i].criteria[0] <= Dist_in  && InputMatches[i].criteria[5] <= Angele_in)
        {
            OutputMatches.push_back(InputMatches[i]);
        }
    }

    int MaxMatch = 60;
    if (OutputMatches.size() > MaxMatch)
    {
        OutputMatches.erase(OutputMatches.begin() + MaxMatch, OutputMatches.end());
    }

    return OutputMatches;
}

//update correspondence from the oringinal 2D and 3D lines
std::vector<pairsmatch> updatecorrespondence(std::vector<line3d> &lines_3d,
                                             std::vector<line2d> &lines_2d,
                                             const Eigen::Matrix3d &K,
                                             const Eigen::Matrix3d &R,
                                             const Eigen::Vector3d &t,
                                             double &theta, double &outlier_threshold)
{
    vector<pairsmatch> updatemaches;
    int indx = 0;
    for (size_t i = 0; i < lines_2d.size(); i++)
    {
        int index = 0;
        double mindist = 10000;
        Vector3d vecdist{0,0,0};
        for (size_t j = 0; j < lines_3d.size(); j++)
        {
            // in this function, the 3d lines have been transformed into starting point frame
            pairsmatch tfmatch(i, lines_2d[i], lines_3d[j].transform3D(R, t));
            Vector3d dist = tfmatch.calcEulerDist(K, theta);
            if (dist.x() < mindist)
            {
                mindist = dist.x();
                vecdist=dist;
                index = j;
            }
        }
        if (mindist < outlier_threshold)
        {
            pairsmatch match(indx, lines_2d[i], lines_3d[index]);
            match.distance=vecdist;
            updatemaches.push_back(match);
            indx++;
        }
    }
    //select the former correspondences with largest overlap distance
    if (1)
    {
        sort(updatemaches.begin(), updatemaches.end(), GreaterSort);
        if (updatemaches.size() > 40) //only keep former 40 correspondences
            updatemaches.erase(updatemaches.begin() + 40, updatemaches.end());
    }
    return updatemaches;
}


std::vector<pairsmatch> LineCorrespondenceV2(std::vector<line3d> &lines_3d,
                                             std::vector<line2d> &lines_2d,
                                             const Eigen::Matrix3d &K,
                                             const Eigen::Matrix3d &R,
                                             const Eigen::Vector3d &t,
                                             double &overlap_th, double &dist_th,
                                             double &degree_th, double &angle_th,
                                             int &per_inliers, double &outlier_th)
{
    vector<pairsmatch> MATCHES_H, MATCHES_V;
    MATCHES_H.clear();
    MATCHES_V.clear();
    vector<line2d> line2d_proH, line2d_proV;
    line2d_proH.clear();
    line2d_proV.clear();
    vector<line3d> line3d_H, line3d_V;
    line3d_H.clear();
    line3d_V.clear();
    Eigen::Vector2d MatchFactorD;

    for (int i = 0; i < lines_3d.size(); ++i)
    {
        line3d line_c = lines_3d[i].transform3D(R, t);
        line2d project2dline = line_c.project3D(K);

        if (project2dline.horizon)
        {
            line2d_proH.push_back(project2dline);
            line3d_H.push_back(lines_3d[i]);
        }
        else
        {
            line2d_proV.push_back(project2dline);
            line3d_V.push_back(lines_3d[i]);
        }
    }

    int index = 0;
    for (int i = 0; i < lines_2d.size(); ++i)
    {
        int idxH;
        int idxV;
        double min_distH = 10000.0;
        double min_distV = 10000.0;
        Vector6d matchdata;
        if(lines_2d[i].horizon)
        {
            for (int j = 0; j < line2d_proH.size(); ++j)
            {
                double beta = acos(abs(lines_2d[i].direction.transpose() * line2d_proH[j].direction));
                if(beta > angle_th)
                    continue;
                MatchFactorD = CalEulerDist(lines_2d[i], line2d_proH[j], overlap_th);
                if (MatchFactorD[1] < overlap_th)
                    continue;
                if (MatchFactorD[0] < min_distH)
                {
                    min_distH = MatchFactorD[0];
                    matchdata << MatchFactorD[0], MatchFactorD[1], 0, 0, 0, beta;
                    idxH = j;
                }
            }
            if(min_distH < 1000)
            {
                pairsmatch match(index, lines_2d[i], line3d_H[idxH]);
                match.line3dto2d = line2d_proH[idxH];
                match.criteria = matchdata;   
                double Weight, w_1, w_2, w_3;
                w_1 = 1 / sqrt(matchdata[0] / 10) * 10;
                w_2 = matchdata[1];
                w_3 = 1 - (matchdata[5] / PI * 180) / 5;
                Weight = w_1 + w_2 + w_3;
                match.weight = Weight;
                match.weight = 0.02;

                MATCHES_H.push_back(match);
                index++;
            }
        }
        else
        {
            for (int j = 0; j < line2d_proV.size(); ++j)
            {
                double beta = acos(abs(lines_2d[i].direction.transpose() * line2d_proV[j].direction));
                if(beta > angle_th)
                    continue;
                MatchFactorD = CalEulerDist(lines_2d[i], line2d_proV[j], overlap_th);
                if (MatchFactorD[1] < overlap_th)
                    continue;
                if (MatchFactorD[0] < min_distV)
                {
                    min_distV = MatchFactorD[0];
                    matchdata << MatchFactorD[0], MatchFactorD[1], 0, 0, 0, beta;
                    idxV = j;
                }
            }
            if(min_distV < 1000)
            {
                pairsmatch match(index, lines_2d[i], line3d_V[idxV]);
                match.line3dto2d = line2d_proV[idxV];
                match.criteria = matchdata;   
                double Weight, w_1, w_2, w_3;
                w_1 = 1 / sqrt(matchdata[0] / 10) * 10;
                w_2 = matchdata[1];
                w_3 = 1 - (matchdata[5] / PI * 180) / 5;
                Weight = w_1 + w_2 + w_3;
                match.weight = Weight;
                match.weight = 0.02;

                MATCHES_V.push_back(match);
                index++;
            }
        }
    }
    vector<pairsmatch> OutputMatchesH = CorrespondenceFilter(MATCHES_H, overlap_th, dist_th, degree_th, angle_th,
                                                            outlier_th, per_inliers);
    vector<pairsmatch> OutputMatchesV = CorrespondenceFilter(MATCHES_V, overlap_th, dist_th, degree_th, angle_th,
                                                             outlier_th, per_inliers);
    vector<pairsmatch> OutputMatches = OutputMatchesH;
    for (int i = 0; i < OutputMatchesV.size(); ++i)
    {
        OutputMatches.push_back(OutputMatchesV[i]);
    }
    return OutputMatches;
}


// theta_th: the two 2d lines angle threshold on image plane on EulerDist
// overlap_th: the two 2d lines overlap length on image plane on EulerDist
// dist_th: the point to line distant on image plane on EulerDist
// angle_th: the angle match factor threshold on AngleDist
std::vector<pairsmatch> LineCorrespondence(std::vector<line3d> &lines_3d,
                                           std::vector<line2d> &lines_2d,
                                           const Eigen::Matrix3d &K,
                                           const Eigen::Matrix3d &R,
                                           const Eigen::Vector3d &t,
                                           double &overlap_th, double &dist_th,
                                           double &degree_th, double &angle_th,
                                           int &per_inliers, double &outlier_th)
{
    vector<pairsmatch> MATCHES;
    MATCHES.clear();
    vector<line2d> line2d_pro;
    line2d_pro.clear();
    vector<line3d> line3d_cam;
    line3d_cam.clear();
    Eigen::Vector2d MatchFactorD;
    Eigen::Vector4d MatchFactorA;
    double lambda = 0.5; 
    double weight = 0.5;

    for (size_t i = 0; i < lines_3d.size(); ++i)
    {
        line3d line_c = lines_3d[i].transform3D(R, t);
        line3d_cam.push_back(line_c);
        line2d project2dline = line_c.project3D(K);
        line2d_pro.push_back(project2dline);
    }

    for (size_t i = 0; i < lines_2d.size(); ++i)
    {
        int idx;
        double min_dist = 10000.0;
        double min_angle = 10.0;

        Vector6d matchdata;
        for (size_t j = 0; j < line2d_pro.size(); ++j)
        {
            // adjust the matching strategy
            MatchFactorA = CalAngleDist(lines_2d[i], line3d_cam[j], K, lambda, degree_th);
            if(MatchFactorA[3] >= angle_th)
            {
                continue;
            }
            MatchFactorD = CalEulerDist(lines_2d[i], line2d_pro[j], overlap_th);

            if(MatchFactorD[1] < overlap_th)
            {
                continue;
            }
            if (MatchFactorD[0] < min_dist)    // consider the euler dist and angle
            {
                //   min_angle = MatchFactorA[0];
                // 0 dist,
                // 1 overlap,
                // 2 alpha: the weighted angle
                // 3 theta: the angle on camera frame
                // 4 phi: the midpoint and nearest angle
                // 5 beta: the angle on image frame
                min_dist = MatchFactorD[0];
                matchdata << MatchFactorD[0], MatchFactorD[1], MatchFactorA[0],
                        MatchFactorA[1], MatchFactorA[2], MatchFactorA[3];
                idx = j;
            }
        }

        if(min_dist < 1000)
        {
            pairsmatch match(i, lines_2d[i], lines_3d[idx]);
            match.line3dto2d = line2d_pro[idx];
            match.criteria = matchdata;   

            double Weight, w_1, w_2, w_3;
            w_1 = 1 / sqrt(matchdata[0] / 10) * 10;
            w_2 = matchdata[1];
            w_3 = 1 - (matchdata[5] / PI * 180) / 5;
            Weight = w_1 + w_2 + w_3;
            match.weight = Weight;
            match.weight = 0.02;

            MATCHES.push_back(match);
        }
    }

    vector<pairsmatch> OutputMatches = CorrespondenceFilter(MATCHES, overlap_th, dist_th, degree_th, angle_th, outlier_th, per_inliers);

    return OutputMatches;
}

double computedistribution(const std::vector<pairsmatch> &matches)
{
    double confidence=0.0;
    Eigen::MatrixXd Vec(matches.size(),3);
    for (size_t i=0; i<matches.size(); i++)
    {
        Eigen::Vector3d dir=matches[i].line3dt.direction;
        Vec.row(i) <<dir.x(), dir.y(), dir.z();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Vec, Eigen::ComputeFullV | Eigen::ComputeFullU); 
	Eigen::MatrixXd singular_values = svd.singularValues();
    confidence=singular_values(singular_values.rows()-1);
    return confidence;
}