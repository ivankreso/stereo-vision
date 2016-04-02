#ifndef TRACKER_FEATURE_DETECTOR_BASE_H_
#define TRACKER_FEATURE_DETECTOR_BASE_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "../../core/image.h"
#include "../../core/types.h"

namespace track {

class FeatureDetectorBase
{
 public:
  virtual ~FeatureDetectorBase() {}

  // sometimes detectors only detect
  virtual void detect(const core::Image& img, std::vector<core::Point>& features)
  { throw "Error\n"; }
  virtual void detect(const cv::Mat& img, std::vector<core::Point>& features)
  { throw "Error\n"; }
  virtual void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features)
  { throw "Error\n"; }

  // and sometimes they are connected to specific descriptors
  virtual void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features, cv::Mat& descriptors)
  { throw "[FeatureDetectorBase]: Empty function call!\n"; }
  virtual double compare(cv::Mat desc1, cv::Mat desc2)
  { throw "[FeatureDetectorBase]: Empty function call!\n"; }

  //virtual void config(std::string config) = 0;
};

}

#endif
