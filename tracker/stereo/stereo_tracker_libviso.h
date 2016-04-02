#ifndef _STEREO_TRACKER_LIBVISO_
#define _STEREO_TRACKER_LIBVISO_

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>

#include "stereo_tracker_base.h"
#include "../../core/types.h"
#include "../../stereo_egomotion/extern/libviso2/src/matcher.h"
#include "../../stereo_egomotion/extern/libviso2/src/viso_stereo.h"

namespace track {

class StereoTrackerLibviso : public StereoTrackerBase
{
 public:
  // bucketing parameters
  struct bucketing {  
    int32_t max_features;  // maximal number of features per bucket 
    double  bucket_width;  // width of bucket
    double  bucket_height; // height of bucket
    bucketing () {
      max_features  = 2;
      bucket_width  = 50;
      bucket_height = 50;
    }
  };
  // general parameters
  struct parameters {
    libviso::Matcher::parameters matcher;
    bucketing   bucket;           // bucketing parameters

    // stereo-specific parameters (mandatory: base)
    double  base;             // baseline (meters)
    int32_t ransac_iters;     // number of RANSAC iterations
    double  inlier_threshold; // fundamental matrix inlier threshold
    bool    reweighting;      // lower border weights (more robust to calibration errors)
    parameters () {
      base             = 1.0;
      ransac_iters     = 200;
      inlier_threshold = 1.5;
      reweighting      = true;
    }
  };
  //StereoTrackerLibviso(libviso::VisualOdometryStereo::parameters param);
  StereoTrackerLibviso();
  ~StereoTrackerLibviso() {}
  virtual void init(core::Image& img_left, core::Image& img_right);
  virtual void track(core::Image& img_left, core::Image& img_right);
  virtual void init(const cv::Mat& img_left, const cv::Mat& img_right);
  virtual void track(const cv::Mat& img_left, const cv::Mat& img_right);
  virtual int countFeatures() const;
  virtual FeatureInfo featureLeft(int i) const;
  virtual FeatureInfo featureRight(int i) const;
  virtual void removeTrack(int id);
  virtual int countActiveTracks() const;

 protected:
  uint32_t frame_cnt_ = 0;

  void track(uint8_t* img_left, uint8_t* img_right, int rows, int cols);
  void saveFeatures(const std::vector<libviso::Matcher::p_match>& matches);
  std::vector<libviso::Matcher::p_match> filterFeatures(const std::vector<libviso::Matcher::p_match>& matches);
  libviso::Matcher* matcher_;
  //libviso::VisualOdometryStereo::parameters param_;
  parameters param_;

  int max_feats_;
  std::vector<FeatureInfo> feats_left_;
  std::vector<FeatureInfo> feats_right_;
  //std::vector<int> age_;
};

}

#endif
