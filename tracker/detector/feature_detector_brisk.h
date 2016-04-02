#ifndef FEATURE_DETECTOR_BRISK_CV_
#define FEATURE_DETECTOR_BRISK_CV_

#include "feature_detector_base.h"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

namespace track {

class FeatureDetectorBRISK : public FeatureDetectorBase
{
public:
  FeatureDetectorBRISK(int thresh=30, int octaves=3, float patternScale=1.0);
  ~FeatureDetectorBRISK();

  virtual void detect(const core::Image& img, std::vector<core::Point>& features)
  { throw "[FeatureDetectorBRISK]: empty function!\n"; }

  virtual void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features, cv::Mat& descriptors);
  virtual double compare(cv::Mat desc1, cv::Mat desc2);

private:
  int thresh_;
  int octaves_;
  float pattern_scale_;
  cv::BRISK* detector_;
};

}

#endif
