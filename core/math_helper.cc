#include "math_helper.h"

namespace core {

void MathHelper::project_stereo(const cv::Mat& cam_params, const cv::Mat& pt3d,
                                core::Point& pt_left, core::Point& pt_right)
{
  double f = cam_params.at<double>(0);
  double cu = cam_params.at<double>(2);
  double cv = cam_params.at<double>(3);
  double b = cam_params.at<double>(4);
  
  double X = pt3d.at<double>(0);
  double Y = pt3d.at<double>(1);
  double Z = pt3d.at<double>(2);

  double u = X / Z;
  double v = Y / Z;
  pt_left.x_ = f*u + cu;
  pt_left.y_ = f*v + cv;
  // right camera
  u = (X - b) / Z;
  pt_right.x_ = f*u + cu;
  pt_right.y_ = pt_left.y_;
}

void MathHelper::projectToStereo(const double* cam_params, const Eigen::Vector4d& pt3d,
                                 core::Point& pt_left, core::Point& pt_right)
{
  double f = cam_params[0];
  double cx = cam_params[2];
  double cy = cam_params[3];
  double b = cam_params[4];

  double x = pt3d[0] / pt3d[2];
  double y = pt3d[1] / pt3d[2];
  pt_left.x_ = f*x + cx;
  pt_left.y_ = f*y + cy;
  // right camera
  x = (pt3d[0] - b) / pt3d[2];
  pt_right.x_ = f*x + cx;
  pt_right.y_ = pt_left.y_;
}

double MathHelper::getDist2D(const core::Point& pt1, const core::Point& pt2)
{
  double xdiff = pt1.x_ - pt2.x_;
  double ydiff = pt1.y_ - pt2.y_;
  double dist = std::sqrt(xdiff*xdiff + ydiff*ydiff);
  return dist;
}

}
