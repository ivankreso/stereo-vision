#include "feature_detector_agast.h"

#include "../../core/cpp_gems.h"

namespace track {

FeatureDetectorAGAST::FeatureDetectorAGAST(int threshold, bool nonmax_suppression,
                                           std::string type) {
  int ast_type = -1;
  if (type == "OAST_9_16")
    ast_type = cv::xfeatures2d::AgastFeatureDetector::OAST_9_16;
  else if (type == "AGAST_5_8")
    ast_type = cv::xfeatures2d::AgastFeatureDetector::AGAST_5_8;
  else {
    //ast_type = cv::xfeatures2d::AgastFeatureDetector::OAST_9_16;
    DEB << "Unknown type";
  }
  detector_ = cv::xfeatures2d::AgastFeatureDetector::create(threshold, nonmax_suppression, ast_type);
}

void FeatureDetectorAGAST::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features) {
  detector_->detect(img, features);
}

}
