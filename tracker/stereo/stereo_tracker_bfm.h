#ifndef TRACKER_STEREO_TRACKER_BFM_H_
#define TRACKER_STEREO_TRACKER_BFM_H_

#include <vector>
#include <deque>
#include <algorithm>

#include "stereo_tracker_base.h"
#include "debug_helper.h"
#include "../base/helper_opencv.h"
#include "../mono/tracker_base.h"
#include "../../core/image.h"
#include "../../core/types.h"
#include "../detector/feature_detector_base.h"

namespace track {

class StereoTrackerBFM : public StereoTrackerBase
{
 public:
  StereoTrackerBFM(FeatureDetectorBase* detector, int max_features, double min_crosscorr,
                   int patch_size, int window_size);
  ~StereoTrackerBFM();
  virtual void init(const cv::Mat& img_left, const cv::Mat& img_right);
  virtual void track(const cv::Mat& img_left, const cv::Mat& img_right);
  virtual int countFeatures() const;
  virtual FeatureInfo featureLeft(int i) const;
  virtual FeatureInfo featureRight(int i) const;
  virtual void removeTrack(int id);
  virtual int countActiveTracks() const;

  virtual FeatureData getLeftFeatureData(int i);
  virtual FeatureData getRightFeatureData(int i);

  virtual void showTrack(int i) const;
  void printStats();
  //const std::vector<core::Point>& getLeftFeatures() { return feats_left_; }
  //const std::vector<core::Point>& getRightFeatures() { return feats_right_; }

 protected:
  void copyPatches(const cv::Mat& img, std::vector<core::Point>& features, std::vector<FeaturePatch>& patches);

  void updateMatches(const std::vector<core::Point>& feats_left,
                                     const std::vector<core::Point>& feats_right,
                                     const std::vector<FeaturePatch>& patches_left,
                                     const std::vector<FeaturePatch>& patches_right,
                                     const std::vector<int>& match_index_left,
                                     const std::vector<int>& match_index_right,
                                     const std::vector<int>& match_index_epi,
                                     std::vector<bool>& unused_features);

  void replaceDeadFeatures(const std::vector<core::Point>& feats_left,
                           const std::vector<core::Point>& feats_right,
                           const std::vector<FeaturePatch>& patches_left,
                           const std::vector<FeaturePatch>& patches_right,
                           const std::vector<int>& match_index_epi,
                           std::vector<bool>& unused_features);

  void matchFeatures(const cv::Mat& img_1, const cv::Mat& img_2,
      const std::vector<core::Point>& feats1, const std::vector<core::Point>& feats2,
      const std::vector<FeaturePatch>& patches1, const std::vector<FeaturePatch>& patches2,
      std::vector<int>& match_index, double dxl, double dxr, double dyu, double dyd, bool is_temporal,
      const std::vector<int>& ages, bool debug);

  void initMatches(const std::vector<core::Point>& feats1, const std::vector<core::Point>& feats2,
      const std::vector<FeaturePatch>& in_patches1, const std::vector<FeaturePatch>& in_patches2,
      const std::vector<int>& match_index,
      std::vector<core::Point>& matches1, std::vector<core::Point>& matches2,
      std::vector<FeaturePatch>& out_patches1, std::vector<FeaturePatch>& out_patches2);

  void filterUnmatched(std::vector<core::Point>& feats1, std::vector<core::Point>& feats2,
      std::vector<FeaturePatch>& patches1, std::vector<FeaturePatch>& patches2,
      std::vector<int>& match_index);

  void filterBadTracks();

  std::vector<size_t> getSortedIndices(std::vector<double> const& values);
  double getCorrelation(const FeaturePatch& p1, const FeaturePatch& p2);

  FeatureDetectorBase* detector_;
  cv::Mat cvimg_lp_, cvimg_rp_, cvimg_lc_, cvimg_rc_;

  int max_feats_;
  bool use_smoothing_;
  uint64_t age_acc_ = 0;
  uint64_t death_count_ = 0;
  double min_crosscorr_;
  double max_disp_change_;
  int wsize_left_, wsize_right_, wsize_up_, wsize_down_;
  std::vector<core::Point> matches_lp_, matches_rp_, matches_lc_, matches_rc_;
  std::vector<int> age_;
  std::vector<int> status_;
  std::vector<FeaturePatch> patches_lp_, patches_rp_, patches_lc_, patches_rc_;
  int cbw_, cbh_; // normalized correlation block width and height

  std::vector<FeatureInfo> matches_left_;
  std::vector<FeatureInfo> matches_right_;

  //std::vector<std::vector<FeatureInfo*>> fgrid_; maybe
};

}

#endif
