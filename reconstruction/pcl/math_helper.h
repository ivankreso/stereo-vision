#ifndef RECONSTRUCTION_BASE_MATH_HELPER_
#define RECONSTRUCTION_BASE_MATH_HELPER_

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace recon
{

namespace MathHelper
{
  void getTopViewHomography(const double* cparams, int src_w, int src_h, int dst_w, int dst_h, 
                            const Eigen::Vector4d& plane, Eigen::Matrix3d& H);
  void debugTopViewHomography(const double* cparams, int src_w, int src_h, int dst_w, int dst_h, 
                              const Eigen::Vector4d& plane, pcl::visualization::PCLVisualizer& viewer);
  void warpImage(const cv::Mat& src, const Eigen::Matrix3d H, cv::Mat& dst);
  void projectPoint(const double* cam_params, Eigen::Vector3d& point, Eigen::Vector2d& proj);
  void getSignedDistancesToModel(const Eigen::Vector4d& plane, const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                                 std::vector<double>& distances);
  void getDistancesToModel(const Eigen::Vector4d& plane, const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                           std::vector<double>& distances);
  void triangulate(const double (&cam_params)[5], double x, double y, double disp, cv::Mat& pt3d);
  void createPolyFromPlane(const Eigen::Vector4d& plane, pcl::PointCloud<pcl::PointXYZ>::Ptr poly_cloud);
  void setTransform2DEM(Eigen::Vector4d& plane_model, Eigen::Matrix4d& transform);
}

}

#endif
