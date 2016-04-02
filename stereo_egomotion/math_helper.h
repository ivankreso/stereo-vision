#ifndef MATH_HELPER_H_
#define MATH_HELPER_H_

#include <vector>
#include <deque>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "extern/libviso2/src/matrix.h"

namespace vo {

class MathHelper
{
public:
   static void getReprojError(const std::deque<std::vector<cv::KeyPoint>>& features_left,
                              const std::deque<std::vector<cv::KeyPoint>>& features_right,
                              const double base, const cv::Mat& C,
                              const cv::Mat& Rt, double& err_left, double& err_right);
   static double getReprojError(cv::Point2f& px, cv::Point3f&pt, cv::Mat& C);
   static void matrixToQuaternion(const libviso::Matrix& m, double quat[4]);
   static void matToQuat(const cv::Mat& m, double quat[4]);
   static void matToQuat2(const cv::Mat& m, double q[4]);
   static void quatToMat(const double trans_vec[7], cv::Mat& Rt);
   static void matrixToMat(const libviso::Matrix& src, cv::Mat& dst);
   static void transMatToQuatVec(const cv::Mat& pose_mat, cv::Vec<double,7>& trans_vec);
   static void invTrans(const cv::Mat& src, cv::Mat& dst);
   static void CameraParamsMatToArray(const cv::Mat& Rt, const cv::Mat& C, double (&rot)[10],
                                      double* trans, double* cam_params);
};

}

#endif
