#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>

#include "../../../core/format_helper.h"
#include "../../../core/math_helper.h"

void SavePointCloud(const cv::Mat& img, const cv::Mat& disp_img, const double* calib)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  Eigen::Vector4d pt3d;
  double min_disp = 1.0;
  for (int y = 0; y < disp_img.rows; y++) {
    for (int x = 0; x < disp_img.cols; x++) {
      double disp = static_cast<double>(disp_img.at<uint16_t>(y,x)) / 256.0;
      //std::cout << "D = " << disp << "\n";
      if (disp > min_disp) {
        core::MathHelper::Triangulate(calib, x, y, disp, pt3d);
        pcl::PointXYZRGB point;
        point.x = pt3d[0];
        point.y = pt3d[1];
        point.z = pt3d[2];
        if (img.channels() == 3) {
          point.b = img.at<cv::Vec3b>(y,x)[0];
          point.g = img.at<cv::Vec3b>(y,x)[1];
          point.r = img.at<cv::Vec3b>(y,x)[2];
        }
        else if (img.channels() == 1) {
          point.b = img.at<uint8_t>(y,x);
          point.g = img.at<uint8_t>(y,x);
          point.r = img.at<uint8_t>(y,x);
        }
        else throw 1;
        point_cloud->points.push_back(point);
      }
    }
  }
  point_cloud->width = point_cloud->points.size();
  point_cloud->height = 1;
  pcl::io::savePCDFile("pcl.pcd", *point_cloud);
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::cerr << "Usage:\n\t" << argv[0] << " img disp_img calib_file\n";
    return 1;
  }
  std::string img_path = argv[1];
  std::string dispimg_path = argv[2];
  std::string calib_path = argv[3];
  double mono_cam[5];
  double color_cam[5];
  core::FormatHelper::ReadCalibKitti(calib_path, mono_cam, color_cam);

  cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
  cv::Mat disp_img = cv::imread(dispimg_path, CV_LOAD_IMAGE_ANYDEPTH);
  std::cout << "Bytes per pixel = " << disp_img.step / disp_img.cols << "\n";

  if (img.channels() == 3)
    SavePointCloud(img, disp_img, color_cam);
  else if (img.channels() == 1)
    SavePointCloud(img, disp_img, mono_cam);
  else throw 1;

  return 0;
}
