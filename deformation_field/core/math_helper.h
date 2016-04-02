#ifndef CORE_MATH_HELPER_H_
#define CORE_MATH_HELPER_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>

#include "types.h"

namespace core {

class MathHelper
{
 public:
  // Calculates a 3D point from 2D stereo features
  static void Triangulate(const Eigen::VectorXd& cam_params, const core::Point& pt_left,
                          const core::Point& pt_right, Eigen::Vector4d& pt3d);
  static void Triangulate(const double* cam_params, double x, double y,
                          double disp, Eigen::Vector4d& pt3d);
  static void triangulate(const double* cam_params, const core::Point& pt_left,
                          const core::Point& pt_right, Eigen::Vector4d& pt3d);
  static void MotionMatrixToParams(const Eigen::Matrix4d& Rt, std::array<double,4>& quaternion,
                                   std::array<double,3>& translation);

  //static void triangulate(const Eigen::MatrixXd& cam_params, const core::Point& pt_left,
  //                        const core::Point& pt_right, Eigen::Vector4d& pt3d);

  //static void triangulate(const cv::Mat& cam_params, const core::Point& pt_left,
  //                        const core::Point& pt_right, cv::Mat& pt3d);

  static void projectToStereo(const double* cam_params, const Eigen::Vector4d& pt3d,
                              core::Point& pt_left, core::Point& pt_right);
  static void project_stereo(const cv::Mat& cam_params, const cv::Mat& pt3d,
                             core::Point& pt_left, core::Point& pt_right);

  static void InverseTransform(const Eigen::Matrix4d& Rt, Eigen::Matrix4d& Rt_inv);
  static void invTrans(const cv::Mat& src, cv::Mat& dst);

  static void GetMotionError(const cv::Mat& rt1, const cv::Mat& rt2, double& trans_error);

  template<typename T>
  static T hammDist(cv::Mat v1, cv::Mat v2);
  static uint8_t hammDist(uint8_t x, uint8_t y);
  static uint64_t hammDist(cv::Mat v1, cv::Mat v2);
  static double getDist2D(const core::Point& pt1, const core::Point& pt2);
  static double GetDistanceL2(const core::Point& pt1, const core::Point& pt2);
  template<typename T>
  static double GetDistanceL1(const cv::Mat vec1, const cv::Mat vec2);
  template<typename T>
  static double GetDistanceL2(const cv::Mat vec1, const cv::Mat vec2);
  template<typename T>
  static double GetDistanceChiSq(const cv::Mat vec1, const cv::Mat vec2);
  template<typename T>
  static double GetDistanceNCC(const cv::Mat vec1, const cv::Mat vec2);

};

inline
void MathHelper::MotionMatrixToParams(const Eigen::Matrix4d& Rt, std::array<double,4>& quaternion,
                                      std::array<double,3>& translation) {
  translation[0] = Rt(0,3);
  translation[1] = Rt(1,3);
  translation[2] = Rt(2,3);

  Eigen::Matrix3d R;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      R(i,j) = Rt(i,j);

  Eigen::Quaterniond q(R);
  //std::cout << q.toRotationMatrix() << "\n\n";
  quaternion[0] = q.w();
  quaternion[1] = q.x();
  quaternion[2] = q.y();
  quaternion[3] = q.z();
}

inline
double MathHelper::GetDistanceL2(const core::Point& pt1, const core::Point& pt2) {
  double xdiff = pt1.x_ - pt2.x_;
  double ydiff = pt1.y_ - pt2.y_;
  double dist = std::sqrt(xdiff*xdiff + ydiff*ydiff);
  return dist;
}

inline
void MathHelper::Triangulate(const Eigen::VectorXd& cam_params, const core::Point& pt_left,
                             const core::Point& pt_right, Eigen::Vector4d& pt3d) {
  double f = cam_params[0];
  double cx = cam_params[2];
  double cy = cam_params[3];
  double b = cam_params[4];

  double disp = pt_left.x_ - pt_right.x_;
  if (disp < 0)
    std::cout << "[MathHelper] Warning: Negative disparity!\n";
  disp = std::max(disp, 0.001);
  pt3d(0) = (pt_left.x_ - cx) * b / disp;
  pt3d(1) = (pt_left.y_ - cy) * b / disp;
  pt3d(2) = f * b / disp;
  pt3d(3) = 1.0;
}

inline
void MathHelper::Triangulate(const double* cam_params, double x, double y,
                             double disp, Eigen::Vector4d& pt3d) {
  double f = cam_params[0];
  double cx = cam_params[2];
  double cy = cam_params[3];
  double b = cam_params[4];

  pt3d[0] = (x - cx) * b / disp;
  pt3d[1] = (y - cy) * b / disp;
  pt3d[2] = f * b / disp;
  pt3d[3] = 1.0;
}

inline
void MathHelper::triangulate(const double* cam_params, const core::Point& pt_left,
                             const core::Point& pt_right, Eigen::Vector4d& pt3d) {
  double f = cam_params[0];
  double cx = cam_params[2];
  double cy = cam_params[3];
  double b = cam_params[4];

  double disp = std::max(pt_left.x_ - pt_right.x_, 0.001);
  pt3d(0) = (pt_left.x_ - cx) * b / disp;
  pt3d(1) = (pt_left.y_ - cy) * b / disp;
  pt3d(2) = f * b / disp;
  pt3d(3) = 1.0;
}

//void MathHelper::triangulate(const cv::Mat& cam_params, const core::Point& pt_left,
//                             const core::Point& pt_right, cv::Mat& pt3d)
//{
//  double f = cam_params.at<double>(0);
//  double cx = cam_params.at<double>(2);
//  double cy = cam_params.at<double>(3);
//  double b = cam_params.at<double>(4);
//
//  double disp = std::max(pt_left.x_ - pt_right.x_, 0.001);
//  pt3d.at<double>(0) = (pt_left.x_ - cx) * b / disp;
//  pt3d.at<double>(1) = (pt_left.y_ - cy) * b / disp;
//  pt3d.at<double>(2) = f * b / disp;
//  pt3d.at<double>(3) = 1.0;
//}
//
//

inline
void MathHelper::InverseTransform(const Eigen::Matrix4d& Rt, Eigen::Matrix4d& Rt_inv) {
  Rt_inv = Eigen::Matrix4d::Identity();
  Rt_inv.block<3,3>(0,0) = Rt.block<3,3>(0,0).transpose();
  Rt_inv.block<3,1>(0,3) = - Rt_inv.block<3,3>(0,0) * Rt.block<3,1>(0,3);
}

inline
void MathHelper::invTrans(const cv::Mat& src, cv::Mat& dst) {
  cv::Mat R(src, cv::Range(0,3), cv::Range(0,3));
  dst = cv::Mat::eye(4, 4, CV_64F);
  //dst = tmp.clone();
  cv::Mat RT = R.t();
  cv::Mat t(src, cv::Range(0,3), cv::Range(3,4));
  cv::Mat RTt = - RT * t;
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
       dst.at<double>(i,j) = RT.at<double>(i,j);
  }
  for(int i = 0; i < 3; i++)
    dst.at<double>(i,3) = RTt.at<double>(i);
}

inline
void MathHelper::GetMotionError(const cv::Mat& rt1, const cv::Mat& rt2, double& trans_error)
{
  double dist_3d = 0.0;
  for(int i = 0; i < 3; i ++) {
    double t = rt1.at<double>(i,3) - rt2.at<double>(i,3);
    dist_3d += t*t;
  }
  trans_error = std::sqrt(dist_3d);
}

template<typename T>
inline
double MathHelper::GetDistanceNCC(const cv::Mat vec1, const cv::Mat vec2)
{
  double n = vec1.rows;
  double A1, A2 = 0.0;
  double B1, B2 = 0.0;
  double C1, C2 = 0.0;
  double D = 0.0;
  for(int i = 0; i < vec1.rows; i++) {
    T val1 = vec1.at<T>(i);
    T val2 = vec2.at<T>(i);
    A1 += (T)val1;
    A2 += (T)val2;
    B1 += (T)(val1 * val1);
    B2 += (T)(val2 * val2);
    D += (T)(val1 * val2);
  }
  C1 = 1.0 / std::sqrt((n * B1) - (A1 * A1));
  C2 = 1.0 / std::sqrt((n * B2) - (A2 * A2));
  double ncc = (n * D - (A1 * A2)) * C1 * C2;
  //if(ncc < -0.9)
  //  printf("NCC = %f\n", ncc);
  if(std::isnan(ncc) || std::isinf(ncc))
    ncc = 0.0;
  return ncc;
}

template<typename T>
inline
double MathHelper::GetDistanceChiSq(const cv::Mat vec1, const cv::Mat vec2)
{
  assert(vec1.rows > 1 && vec1.cols == 1);
  double chi = 0.0;
  for(int i = 0; i < vec1.rows; i++) {
    double diff = vec1.at<T>(i) - vec2.at<T>(i);
    double sum = std::max(vec1.at<T>(i) + vec2.at<T>(i), 0.001);
    if(std::isnan(diff))
      throw "BUG\n";
    chi += 0.5 * (diff*diff) / sum;
  }
  return chi;
}

template<typename T>
inline
double MathHelper::GetDistanceL1(const cv::Mat vec1, const cv::Mat vec2)
{
  assert(vec1.rows > 1 && vec1.cols == 1);
  double sum = 0.0;
  for(int i = 0; i < vec1.rows; i++) {
    double diff = vec1.at<T>(i) - vec2.at<T>(i);
    if(std::isnan(diff))
      throw "BUG\n";
    sum += std::abs(diff);
  }
  return sum;
}

template<typename T>
inline
double MathHelper::GetDistanceL2(const cv::Mat vec1, const cv::Mat vec2)
{
  assert(vec1.rows > 1 && vec1.cols == 1);
  double sum = 0.0;
  for(int i = 0; i < vec1.rows; i++) {
    double diff = vec1.at<T>(i) - vec2.at<T>(i);
    sum += diff*diff;
  }
  return std::sqrt(sum);
}

template<typename T>
inline
T MathHelper::hammDist(cv::Mat v1, cv::Mat v2)
{
  assert(v1.rows == v2.rows && v1.cols == v2.cols);
  assert(v1.rows == 1 && v1.cols > 0);

  T dist = static_cast<T>(0);
  for(int i = 0; i < v1.cols; i++)
    dist += static_cast<T>(hammDist(v1.at<uint8_t>(i), v2.at<uint8_t>(i)));

  return dist;
}

inline
uint8_t MathHelper::hammDist(uint8_t x, uint8_t y)
{
  uint8_t dist = 0;
  uint8_t val = x ^ y; // XOR
  // Count the number of set bits
  while(val) {
    ++dist;
    val &= val - 1;
  }
  return dist;
}

inline
uint64_t MathHelper::hammDist(cv::Mat v1, cv::Mat v2)
{
  assert(v1.rows == v2.rows && v1.cols == v2.cols);
  assert(v1.rows == 1 && v1.cols > 0);

  uint64_t dist = 0;
  for(int i = 0; i < v1.cols; i++)
    dist += (uint64_t) hammDist(v1.at<uint8_t>(i), v2.at<uint8_t>(i));

  return dist;
}

}

#endif
