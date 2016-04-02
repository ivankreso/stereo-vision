#include "math_helper.h"

#include <iostream>
#include <Eigen/LU>
#include <Eigen/Dense>

namespace recon
{

namespace MathHelper
{

void getTopViewHomography(const double* cparams, int src_w, int src_h, int dst_w, int dst_h, 
                          const Eigen::Vector4d& plane, Eigen::Matrix3d& H)
{
  double fx = cparams[0];
  double fy = cparams[1];
  double cx = cparams[2];
  double cy = cparams[3];

  Eigen::Vector3d n;
  n << plane[0], plane[1], plane[2];
  double d = plane[3];
  if(d < 0.0) {
    throw "Error!\n";
  }
  std::cout << "n = \n" << n << "\n\n";
  std::cout << "d = " << d << "\n\n";

  Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
  K1(0,0) = fx;
  K1(1,1) = fy;
  K1(0,2) = cx;
  K1(1,2) = cy;
  Eigen::Matrix3d K1_inv = K1.inverse();

  Eigen::Vector3d q1a;
  Eigen::Vector3d q1b;
  q1a << (double) 0.0, (double) src_h - 1.0, 1.0;
  q1b << (double) src_w - 1.0, (double) src_h - 1.0, 1.0;
  q1a = K1_inv * q1a;
  q1b = K1_inv * q1b;

  Eigen::Vector3d Q1a, Q1b;
  Q1a = (-d * q1a) / n.dot(q1a);
  Q1b = (-d * q1b) / n.dot(q1b);
  std::cout << "Q1a:\n" << Q1a << "\nQ1b:\n" << Q1b << "\n\n";

  // determine R matrix
  Eigen::Matrix3d R;
  //Eigen::Vector3d Q1diff = Q1a - Q1b;
  Eigen::Vector3d Q1diff = Q1b - Q1a;
  std::cout << Q1diff << "\n";
  R.row(0) = Q1diff / Q1diff.norm();
  R.row(2) = -n;
  R.row(1) = -(R.row(0).cross(R.row(2)));

  std::cout << "0 == " << Q1diff.dot(n) << "\n";

  //double Ty = -10.0 - (R.row(1) * Q1a);
  double Ty = 10.0 - (R.row(1) * Q1a);
  std::cout << "Ty = " << Ty << "\n";
  Eigen::Vector3d t;
  t << 0.0, Ty, 0.0;

  Eigen::Vector3d Q2a = R * Q1a + t;
  Eigen::Vector3d Q2b = R * Q1b + t;
  std::cout << "Q2a:\n" << Q2a << "\nQ2b:\n" << Q2b << "\n\n";;
  double dst_cx = dst_w / 2.0;
  double dst_cy = dst_h / 2.0;
  double f2 = (dst_cy - 1.0) * (Q2a[2] / Q2a[1]);

  Eigen::Matrix3d K2;
  K2 << f2,   0.0,  dst_cx,
        0.0,  f2,   dst_cy,
        0.0,  0.0,  1.0;
  std::cout << "K2:\n" << K2 << "\n";
  std::cout << "q2a:\n" << (K2 * Q2a) / Q2a[2] << "\nq2b:\n" << (K2 * Q2b) / Q2b[2] << "\n\n";;

  //std::cout << "d = " << d << "\n";
  //std::cout << "<n,Q1a> = " << n.dot(Q1a) << "\n\n";
  Eigen::Matrix3d H_n = R - ((t * n.transpose()) / d);
  std::cout << "Q2a_H:\n" << H_n * Q1a << "\n\n";
  H = K2 * H_n * K1_inv;
  Eigen::Vector3d q2a_H = H * q1a;
  q2a_H = q2a_H / q2a_H[2];
  std::cout << "q2a_H:\n" << q2a_H << "\n\n";
}

void debugTopViewHomography(const double* cparams, int src_w, int src_h, int dst_w, int dst_h,
                            const Eigen::Vector4d& plane, pcl::visualization::PCLVisualizer& viewer)
{
  double fx = cparams[0];
  double fy = cparams[1];
  double cx = cparams[2];
  double cy = cparams[3];

  Eigen::Vector3d n;
  n << plane[0], plane[1], plane[2];
  double d = plane[3];
  if(d < 0.0) {
    throw "Error!\n";
  }
  std::cout << "n = \n" << n << "\n\n";
  std::cout << "d = " << d << "\n\n";

  Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
  K1(0,0) = fx;
  K1(1,1) = fy;
  K1(0,2) = cx;
  K1(1,2) = cy;
  Eigen::Matrix3d K1_inv = K1.inverse();

  Eigen::Vector3d q1a;
  Eigen::Vector3d q1b;
  q1a << (double) 0.0, (double) src_h - 1.0, 1.0;
  q1b << (double) src_w - 1.0, (double) src_h - 1.0, 1.0;
  q1a = K1_inv * q1a;
  q1b = K1_inv * q1b;

  Eigen::Vector3d Q1a, Q1b;
  Q1a = (-d * q1a) / n.dot(q1a);
  Q1b = (-d * q1b) / n.dot(q1b);
  std::cout << "Q1a:\n" << Q1a << "\nQ1b:\n" << Q1b << "\n\n";

  pcl::PointXYZ pt1, pt2;
  for(int i = 0; i < 3; i++) {
    pt1.data[i] = Q1a[i];
    pt2.data[i] = Q1b[i];
  }
  viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(pt1, pt2, 0, 200, 0, "line1");
}

void projectPoint(const double* cam_params, Eigen::Vector3d& point, Eigen::Vector2d& proj)
{
  double fx = cam_params[0];
  double fy = cam_params[1];
  double cx = cam_params[2];
  double cy = cam_params[3];

  proj[0] = point[0] / point[2];
  proj[1] = point[1] / point[2];
  proj[0] = fx * proj[0] + cx;
  proj[1] = fy * proj[1] + cy;
}

void setTransform2DEM(Eigen::Vector4d& plane_model, Eigen::Matrix4d& transform)
{
  Eigen::Vector3d n1, n2;
  for(int i = 0; i < 3; i++) {
    n1[i] = plane_model[i];
    n2[i] = 0.0;
  }
  n2[1] = 1.0;
  // find axis with cross product

  // find angle with dot product

  // move Y with d after rotation
}

void createPolyFromPlane(const Eigen::Vector4d& plane, pcl::PointCloud<pcl::PointXYZ>::Ptr poly_cloud)
{
  double a = plane[0];
  double b = plane[1];
  double c = plane[2];
  double d = plane[3];

  double x[4] = {-20, 20, 20, -20};
  double z[4] = {  0,  0, 100, 100};
  for(int i = 0; i < 4; i++) {
    double y = (a*x[i] + c*z[i] + d) / (-b);
    pcl::PointXYZ pt(x[i], y, z[i]);
    poly_cloud->points.push_back(pt);
  }
}

void triangulate(const double (&cam_params)[5], double x, double y, double disp, cv::Mat& pt3d)
{
  double f = cam_params[0];
  double cx = cam_params[2];
  double cy = cam_params[3];
  double b = cam_params[4];

  pt3d.at<double>(0) = (x - cx) * b / disp;
  pt3d.at<double>(1) = (y - cy) * b / disp;
  pt3d.at<double>(2) = f * b / disp;
  pt3d.at<double>(3) = 1.0;
}

void getSignedDistancesToModel(const Eigen::Vector4d& plane, const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                                           std::vector<double>& distances)
{
  distances.assign(pc->points.size(), 0.0);
  double a = plane[0];
  double b = plane[1];
  double c = plane[2];
  double d = plane[3];
  double norm = std::sqrt(a*a + b*b + c*c);
  for(size_t i = 0; i < pc->points.size(); i++) {
    pcl::PointXYZ pt = pc->points[i];
    double p = a*pt.x + b*pt.y + c*pt.z + d;
    distances[i] = p / norm;
  }
}

void getDistancesToModel(const Eigen::Vector4d& plane, const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                         std::vector<double>& distances)
{
  distances.assign(pc->points.size(), 0.0);
  double a = plane[0];
  double b = plane[1];
  double c = plane[2];
  double d = plane[3];
  double norm = std::sqrt(a*a + b*b + c*c);
  for(size_t i = 0; i < pc->points.size(); i++) {
    pcl::PointXYZ pt = pc->points[i];
    double p = a*pt.x + b*pt.y + c*pt.z + d;
    distances[i] = std::abs(p / norm);
  }
}

}

}
