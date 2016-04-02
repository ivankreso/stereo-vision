#ifndef STEREO_TRACKER_ARTIFICIAL_
#define STEREO_TRACKER_ARTIFICIAL_

#include <string>
#include <vector>
#include <fstream>

#include <opencv2/core/core.hpp>

#include "stereo_tracker_base.h"
#include "../../core/format_helper.h"

namespace track {

class StereoTrackerArtificial : public StereoTrackerBase
{
 public:
  StereoTrackerArtificial(std::string src_folder, int max_feats);
  virtual void init(core::Image& img_left, core::Image& img_right);
  virtual void track(core::Image& img_left, core::Image& img_right);
  virtual int countFeatures() const;
  virtual FeatureInfo featureLeft(int i) const;
  virtual FeatureInfo featureRight(int i) const;
  virtual void removeTrack(int id) {}

 protected:
  void readNextFrame();
  std::vector<core::Point> matches_lp_;
  std::vector<core::Point> matches_rp_;
  std::vector<core::Point> matches_lc_;
  std::vector<core::Point> matches_rc_;
  std::vector<int> age_;

  std::string src_folder_;
  std::vector<std::string> filelist_;
  uint32_t frame_cnt_ = 0;
  int max_feats_;
};

}

#endif
