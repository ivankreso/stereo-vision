#ifndef TRACKER_HELPER_OPENCV_H_
#define TRACKER_HELPER_OPENCV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "../../core/image.h"

namespace track {

class HelperOpencv
{
public:
  static void MatToImage(const cv::Mat& mat, core::Image& img);
  static void ImageToMat(const core::Image& img, cv::Mat& mat);
  static void DrawDescriptor(const cv::Mat& desc, int rows, std::string name);
  static void DrawPoint(const core::Point& pt, const cv::Mat& img, std::string name);
  static void Keypoint2Point(const cv::KeyPoint& kp, core::Point& pt);
  static void FloatImageToMat(const core::Image& img, cv::Mat& mat);
  static void PointsToCvPoints(const std::vector<core::Point>& in_feats, std::vector<cv::Point2f>& out_feats);
  static void PointsToCvKeypoints(const std::vector<core::Point>& in_feats, std::vector<cv::KeyPoint>& out_feats);
  static void DrawPatch(const core::Point& pt, const cv::Mat& img, int wsize);
  static void DrawFloatDescriptor(const cv::Mat& desc, int rows, std::string name);
};

}

#endif
