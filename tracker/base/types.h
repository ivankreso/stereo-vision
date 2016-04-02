#ifndef TRACKER_BASE_TYPES_H_
#define TRACKER_BASE_TYPES_H_

#include "../../core/types.h"
#include <opencv2/core/core.hpp>

namespace track {

struct FeatureInfo {
  core::Point curr_;
  core::Point prev_;
  int age_;
  int status_;
  FeatureInfo(): age_(-1), status_(0) {}
};

struct TrackStats {
  TrackStats(double dist, double left_resp, double right_resp) : matching_distance(dist),
      left_response_sum(left_resp), right_response_sum(right_resp) {}
  double matching_distance;
  double left_response_sum;
  double right_response_sum;
};

struct FeaturePatch {
  cv::Mat mat_;
  double A_, B_, C_;
  FeaturePatch() : A_(0.0), B_(0.0), C_(0.0) {}
};

struct FeatureData
{
  FeatureInfo feat_;
  cv::Mat desc_prev_;
  cv::Mat desc_curr_;
  core::DescriptorNCC ncc_prev_;
  core::DescriptorNCC ncc_curr_;
};

//typedef float PixelType;
//static const int kPixelTypeOpenCV = CV_32F;

typedef uint8_t PixelType;
static const int kPixelTypeOpenCV = CV_8U;

}

#endif

