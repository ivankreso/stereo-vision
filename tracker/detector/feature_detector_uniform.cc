#include "feature_detector_uniform.h"

namespace track
{

FeatureDetectorUniform::FeatureDetectorUniform(FeatureDetectorBase* detector, int h_bins, int v_bins, int fpb)
                                               : detector_(detector)
{
  h_bins_ = h_bins;
  v_bins_ = v_bins;
  fpb_ = fpb;
}

void FeatureDetectorUniform::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features)
{
  std::vector<cv::KeyPoint> tmp_features;
  detector_->detect(img, tmp_features);

  //features.resize(keypoints_.size());
  //for(size_t i = 0; i < features.size(); i++) {
  //   features[i].x_ = keypoints_[i].pt.x;
  //   features[i].y_ = keypoints_[i].pt.y;
  //}

  std::vector<int> bins(h_bins_ * v_bins_, 0);
  double bin_w = (double)img.cols / h_bins_;
  double bin_h = (double)img.rows / v_bins_;
  int r, c;

  features.clear();
  for(size_t i = 0; i < tmp_features.size(); i++) {
    cv::KeyPoint pt = tmp_features[i];
    //pt.pt.x_ = tmp_features[i].pt.x;
    //pt.pt.y_ = tmp_features[i].pt.y;
    //pt.size = tmp_features[i].size;
    //std::cout << x1_ << " - " << y1;
    // filter uniformly
    c = (int)pt.pt.x / bin_w;
    r = (int)pt.pt.y / bin_h;
    assert(c >= 0 && c < h_bins_);
    assert(r >= 0 && r < v_bins_);
    if(bins[r*h_bins_ + c] <= fpb_) {
      features.push_back(pt);
      bins[r*h_bins_ + c]++;
    }
  }
  std::cout << "[FeatureDetectorUniform]: Detected features = " << features.size() << "\n";
}

void FeatureDetectorUniform::detect(const cv::Mat& img, std::vector<core::Point>& features)
{
  std::vector<core::Point> tmp_features;
  detector_->detect(img, tmp_features);

  //features.resize(keypoints_.size());
  //for(size_t i = 0; i < features.size(); i++) {
  //   features[i].x_ = keypoints_[i].pt.x;
  //   features[i].y_ = keypoints_[i].pt.y;
  //}

  std::vector<int> bins(h_bins_ * v_bins_, 0);
  double bin_w = (double)img.cols / h_bins_;
  double bin_h = (double)img.rows / v_bins_;
  int r, c;

  features.clear();
  core::Point pt;
  for(size_t i = 0; i < tmp_features.size(); i++) {
    pt.x_ = tmp_features[i].x_;
    pt.y_ = tmp_features[i].y_;
    //std::cout << x1_ << " - " << y1;
    // filter uniformly
    c = (int)pt.x_ / bin_w;
    r = (int)pt.y_ / bin_h;
    assert(c >= 0 && c < h_bins_);
    assert(r >= 0 && r < v_bins_);
    if(bins[r*h_bins_ + c] <= fpb_) {
      features.push_back(pt);
      bins[r*h_bins_ + c]++;
    }
  }
}

void FeatureDetectorUniform::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features,
                                    cv::Mat& descriptors)
{
  throw 1;
  //std::vector<core::Point> tmp_features;
  //detector_->detect(img, tmp_features);

  ////features.resize(keypoints_.size());
  ////for(size_t i = 0; i < features.size(); i++) {
  ////   features[i].x_ = keypoints_[i].pt.x;
  ////   features[i].y_ = keypoints_[i].pt.y;
  ////}

  //std::vector<int> bins(h_bins_ * v_bins_, 0);
  //double bin_w = (double)img.cols / h_bins_;
  //double bin_h = (double)img.rows / v_bins_;
  //int r, c;

  //features.clear();
  //core::Point pt;
  //for(size_t i = 0; i < tmp_features.size(); i++) {
  //  pt.x_ = tmp_features[i].x_;
  //  pt.y_ = tmp_features[i].y_;
  //  //std::cout << x1_ << " - " << y1;
  //  // filter uniformly
  //  c = (int)pt.x_ / bin_w;
  //  r = (int)pt.y_ / bin_h;
  //  assert(c >= 0 && c < h_bins_);
  //  assert(r >= 0 && r < v_bins_);
  //  if(bins[r*h_bins_ + c] <= fpb_) {
  //    features.push_back(pt);
  //    bins[r*h_bins_ + c]++;
  //  }
  //}
}

}
