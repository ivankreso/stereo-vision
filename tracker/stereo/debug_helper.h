#ifndef _TRACKER_STEREO_DEBUG_HELPER_H__
#define _TRACKER_STEREO_DEBUG_HELPER_H__

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "../../core/types.h"
#include "../base/types.h"
#include "stereo_tracker_refiner.h"

namespace track {

class DebugHelper
{
public:
  static void DebugStereoRefiner(const cv::Mat& img_lp, const cv::Mat& img_rp,
      const cv::Mat& img_lc, const cv::Mat& img_rc, StereoTrackerRefiner& refiner,
      const cv::Mat& cvRt, const double* cam_params);

  static void renderPatch(const FeaturePatch& patch, cv::Mat& img);
  static void drawFeatures(const std::vector<core::Point>& feats, const cv::Scalar& color, cv::Mat& img);
  static void drawPoint(const core::Point& pt, const cv::Scalar& color, cv::Mat& img);
};

}

#endif
