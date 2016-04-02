#ifndef TRACKER_DETECTOR_AGAST_
#define TRACKER_DETECTOR_AGAST_

#include <opencv2/xfeatures2d.hpp>

#include "feature_detector_base.h"


namespace track {

class FeatureDetectorAGAST : public FeatureDetectorBase
{
 public:
  FeatureDetectorAGAST(int threshold, bool nonmax_suppression, std::string type);
  void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features) override;
 private:
  cv::Ptr<cv::Feature2D> detector_;
};

}

#endif
