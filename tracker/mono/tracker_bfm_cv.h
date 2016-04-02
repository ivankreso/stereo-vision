#ifndef MONO_TRACKER_BFM_CV_H_
#define MONO_TRACKER_BFM_CV_H_

#include "tracker_base.h"
#include "../detector/feature_detector_base.h"

namespace track {

class TrackerBFMcv : public TrackerBase {
public:
  TrackerBFMcv(FeatureDetectorBase& detector, int max_features, int window_size, double max_distance);
  //TrackerBFMcv(FeatureDetectorBase& detector, int max_features, int ws_left, int ws_right, int ws_up,
  //             int ws_down, double max_distance);

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
  void match_features(const std::vector<cv::KeyPoint>& feats1, const std::vector<cv::KeyPoint>& feats2,
                      const cv::Mat& desc1, const cv::Mat& desc2, std::vector<int>& match_index,
                      const std::vector<int>& ages, bool debug);

  void update_alive_tracks(const std::vector<cv::KeyPoint>& feats, const cv::Mat& desc,
                           const std::vector<int>& match_index, std::vector<bool>& unused_features);

  void replace_dead_tracks(const std::vector<cv::KeyPoint>& feats_c, const cv::Mat& desc,
                           std::vector<bool>& unused_features);

  void save_unused_features(const std::vector<cv::KeyPoint>& feats_c, const cv::Mat& desc,
                            std::vector<bool>& unused_features);

  FeatureDetectorBase& detector_;
  cv::Mat cvimg_p_, cvimg_c_;
  std::vector<cv::KeyPoint> matches_p_, matches_c_;
  cv::Mat desc_prev_, desc_curr_;
  std::vector<cv::KeyPoint> prev_unused_feats_;
  //std::vector<cv::Mat> prev_unused_desc_;
  cv::Mat prev_unused_desc_;

  int max_feats_;
  double max_distance_;
  uint64_t age_acc_ = 0;
  uint64_t death_count_ = 0;
  int max_dist_x_, max_dist_y_;
  int wsize_;
  std::vector<int> age_;
};

}

#endif
