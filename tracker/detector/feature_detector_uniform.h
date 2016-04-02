#ifndef TRACKER_FEATURE_DETECTOR_UNIFORM_H_
#define TRACKER_FEATURE_DETECTOR_UNIFORM_H_

#include "feature_detector_base.h"

namespace track {

class FeatureDetectorUniform : public FeatureDetectorBase
{
 public:
  FeatureDetectorUniform(FeatureDetectorBase* detector, int h_bins, int v_bins, int fpb);
  virtual void detect(const cv::Mat& img, std::vector<core::Point>& features);
  virtual void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features);
  virtual void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features, cv::Mat& descriptors);

 private:
  int h_bins_, v_bins_, fpb_;
  FeatureDetectorBase* detector_;
};

}

#endif
