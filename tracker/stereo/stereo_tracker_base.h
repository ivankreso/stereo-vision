#ifndef STEREO_TRACKER_BASE_
#define STEREO_TRACKER_BASE_

#include "../base/types.h"
#include "../../core/image.h"
#include "../../core/types.h"

namespace track {

class StereoTrackerBase {
 public:
  virtual void init(core::Image& img_left, core::Image& img_right) {
    (void)img_left;
    (void)img_right;
    throw 1;
  }
  virtual void track(core::Image& img_left, core::Image& img_right) {
    (void)img_left;
    (void)img_right;
    throw 1;
  }
  virtual void init(const cv::Mat& img_left, const cv::Mat& img_right) {
    (void)img_left;
    (void)img_right;
    throw 1;
  }
  virtual void track(const cv::Mat& img_left, const cv::Mat& img_right) {
    (void)img_left;
    (void)img_right;
    throw 1;
  }
  virtual int countFeatures() const = 0;
  virtual FeatureInfo featureLeft(int i) const = 0;
  virtual FeatureInfo featureRight(int i) const = 0;
  virtual void removeTrack(int i) = 0;
  virtual bool IsAlive(int i) const { throw 1; }
  virtual TrackStats GetTrackStats(int i) const { throw 1; }

  virtual const std::vector<int>& GetLiveTracks() const { throw 1; }
  virtual const FeatureInfo& LeftTrack(int i) const { throw 1; }
  virtual const FeatureInfo& RightTrack(int i) const { throw 1; }

  virtual FeatureData getLeftFeatureData(int) { throw "Error"; }
  virtual FeatureData getRightFeatureData(int) { throw "Error"; }
  virtual void showTrack(int) const { throw 1; }
  virtual int countActiveTracks() const { throw 1; }

  virtual ~StereoTrackerBase() {}
};

}

#endif
