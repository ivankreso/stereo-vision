#ifndef __STEREO_TRACKER_REFINER__
#define __STEREO_TRACKER_REFINER__

#include "stereo_tracker_base.h"
#include "../refiner/feature_refiner_base.h"

namespace track {

class StereoTrackerRefiner : public StereoTrackerBase
{
 public:
  StereoTrackerRefiner(StereoTrackerBase* tracker, refiner::FeatureRefinerBase* refiner,
                       bool debug_on);
  ~StereoTrackerRefiner();
  virtual void init(const cv::Mat& img_left, const cv::Mat& img_right);
  virtual void track(const cv::Mat& img_left, const cv::Mat& img_right);
  virtual void init(core::Image& img_left, core::Image& img_right);
  virtual void track(core::Image& img_left, core::Image& img_right);
  virtual int countFeatures() const;
  virtual int countActiveTracks() const;
  virtual void removeTrack(int id);
  virtual FeatureInfo featureLeft(int i) const;
  virtual FeatureInfo featureRight(int i) const;

  void printStats() const;
  void debug() const;

  std::vector<track::refiner::FeatureData> fdata_lp_, fdata_rp_, fdata_lc_, fdata_rc_;
  StereoTrackerBase* tracker_;

 private:
  refiner::FeatureRefinerBase* refiner_;
  core::ImageSetExact imgset_left_, imgset_left_prev_;
  core::ImageSetExact imgset_right_, imgset_right_prev_;
  cv::Mat img_lp_, img_rp_;
  std::vector<core::Point> points_lp_, points_rp_, points_lc_, points_rc_;
  std::vector<int> age_;
  //std::vector<std::tuple<double,double>> residue_lp_, residue_rp_, residue_lc_, residue_rc_;
  bool debug_on_;

  int max_feats_;
};

}

#endif
