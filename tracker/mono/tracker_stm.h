#ifndef MONO_TRACKER_STM_H_
#define MONO_TRACKER_STM_H_

#include "tracker_base.h"
#include "../detector/feature_detector_base.h"

namespace track {

class TrackerSTM : public TrackerBase {
typedef double DistType;
public:
  //TrackerSTM(TrackerBase& tracker, double Q=1.2, double a=3.0);
  TrackerSTM(TrackerBase* tracker, double Q=1.05, double a=3.0);
  ~TrackerSTM();

  virtual int init(const cv::Mat& img);
  virtual int track(const cv::Mat& img);
  virtual int countTracked() { return tracker_->countTracked(); }
  virtual int countFeatures() { return tracker_->countFeatures(); }
  virtual FeatureInfo feature(int idx) { return tracker_->feature(idx); }

  virtual int getAge(int idx) const { return tracker_->getAge(idx); }
  virtual bool isAlive(int idx) const { return tracker_->isAlive(idx); }
  virtual void removeTrack(int idx) { tracker_->removeTrack(idx); }

private:
  TrackerBase* tracker_ = nullptr;
  double Q_, a_;
  std::vector<cv::Mat> stm_;
  //std::vector<core::DescriptorNCC> ncc_descriptors_;
  //std::vector<std::vector<uint32_t>> stm_dists_;
  std::vector<std::vector<DistType>> stm_dists_;
};

}

#endif
