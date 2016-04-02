#ifndef FEATURE_DETECTOR_HARRIS_FREAK_
#define FEATURE_DETECTOR_HARRIS_FREAK_

#include "feature_detector_base.h"
#include "feature_detector_harris_cv.h"

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

namespace track {

class FeatureDetectorHarrisFREAK : public FeatureDetectorBase
{
public:
  FeatureDetectorHarrisFREAK(FeatureDetectorHarrisCV* detector, cv::FREAK* extractor);
  ~FeatureDetectorHarrisFREAK();

  virtual void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features, cv::Mat& descriptors);
  virtual double compare(cv::Mat desc1, cv::Mat desc2);

private:
  FeatureDetectorHarrisCV* detector_; 
  cv::FREAK* extractor_;
};

}

#endif
