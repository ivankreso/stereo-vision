#ifndef MONO_TRACKER_BFM_H_
#define MONO_TRACKER_BFM_H_

#include "tracker_base.h"
#include "../detector/feature_detector_base.h"
#include "../../reconstruction/base/stereo_costs.h"
#include "../base/helper_opencv.h"
#include "../base/types.h"
#include "../stereo/debug_helper.h"
#include "../../reconstruction/base/stereo_costs.h"

namespace track {

class TrackerBFM : public TrackerBase {
public:
  TrackerBFM(FeatureDetectorBase& detector, int max_features, double min_ncc, int patch_size,
             int wsz, bool match_with_oldest = true);

  virtual int init(const cv::Mat& img);
  virtual int track(const cv::Mat& img);
  virtual int countTracked();
  virtual int countFeatures();
  virtual FeatureInfo feature(int i);
  virtual FeatureData getFeatureData(int i);

  virtual int getAge(int idx) const { return age_[idx]; }
  virtual bool isAlive(int idx) const { return age_[idx] > 0 ? true : false; }
  virtual void removeTrack(int i) { age_[i] = -1; }

  virtual void showTrack(int i);

  void printStats();

private:
  void match_features(const std::vector<cv::KeyPoint>& feats1,
                      const std::vector<cv::KeyPoint>& feats2,
                      const std::vector<core::DescriptorNCC>& desc1,
                      const std::vector<core::DescriptorNCC>& desc2,
                      std::vector<int>& match_index,
                      const std::vector<int>& ages, bool replacing_dead);

  void update_alive_tracks(const std::vector<cv::KeyPoint>& feats,
                           const std::vector<core::DescriptorNCC>& desc,
                           const std::vector<int>& match_index,
                           std::vector<bool>& unused_features);

  void replace_dead_tracks(const std::vector<cv::KeyPoint>& feats_c,
                           const std::vector<core::DescriptorNCC>& desc,
                           std::vector<bool>& unused_features);

  void save_unused_features(const std::vector<cv::KeyPoint>& feats,
                            const std::vector<core::DescriptorNCC>& descriptors,
                            std::vector<bool>& unused_features);

  void compute_ncc_descriptors(const cv::Mat& img, const std::vector<cv::KeyPoint>& features,
                               std::vector<core::DescriptorNCC>& patches);

  FeatureDetectorBase& detector_;
  cv::Mat cvimg_p_, cvimg_c_;
  std::vector<cv::KeyPoint> matches_p_, matches_c_;
  std::vector<cv::KeyPoint> prev_unused_feats_;
  std::vector<core::DescriptorNCC> desc_curr_, desc_ref_;
  std::vector<core::DescriptorNCC> prev_unused_desc_;

  int max_feats_;
  int patch_size_;
  double min_ncc_;
  bool match_with_oldest_ = true;
  uint64_t age_acc_ = 0;
  uint64_t death_count_ = 0;
  int wsize_;
  double max_dist_x_, max_dist_y_;  
  std::vector<int> age_;
};

}

#endif
