#ifndef TRACKER_STEREO_TRACKER_H_
#define TRACKER_STEREO_TRACKER_H_

#include <vector>
#include <unordered_map>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../base/types.h"
#include "stereo_tracker_base.h"
#include "debug_helper.h"
#include "../base/helper_opencv.h"
#include "../mono/tracker_base.h"
#include "../../core/image.h"
#include "../../core/types.h"
#include "../detector/feature_detector_base.h"
#include "../../reconstruction/base/stereo_costs.h"

namespace track {

class StereoTracker : public StereoTrackerBase
{
 public:
  StereoTracker(TrackerBase& tracker, int max_disparity, int stereo_wsz,
                double ncc_thresh, bool estimate_subpixel,
                bool use_df, const std::string& deformation_field_path);
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

private:
  friend class boost::serialization::access;
  // When the class Archive corresponds to an output archive, the
  // & operator is defined similar to <<.  Likewise, when the class Archive
  // is a type of input archive the & operator is defined similar to >>.
  template<class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
      // serialize base class information
      //ar & boost::serialization::base_object<StereoTrackerBase>(*this);
      ar & max_feats_;
      ar & img_size_;
      ar & max_disparity_;
      ar & stereo_wsz_;
      ar & margin_sz_;
      ar & ncc_thresh_;
      ar & estimate_subpixel_;
      ar & age_;
      ar & pts_left_prev_;
      ar & pts_left_curr_;
      ar & pts_right_prev_;
      ar & pts_right_curr_;
  }

  void AddMissingDescriptors(const cv::Mat& img, const core::Point& point, int window_size,
                             std::vector<std::pair<bool, core::DescriptorNCC>>& descriptors_rprev_);

  bool stereo_match_ncc(const core::DescriptorNCC& desc_left,
                        const std::vector<std::pair<bool, core::DescriptorNCC>>& descriptors_right,
                        const core::Point& left_pt, const cv::Mat& img_right,
                        bool debug, core::Point& right_pt);

  // deformation field functions
  void ComputeCellCenters();
  void GetPointCell(const core::Point& pt, int& row, int& col);
  void InterpolateBilinear(const cv::Mat& mat, const int row, const int col,
                           const double x, const double y, double& ival);
  void InterpolateLinear(const double val1, const double val2, const double x,
                         const double size, double& ival);
  void ApplyDeformationField(const cv::Mat& def_x, const cv::Mat& def_y, core::Point& pt);

  track::TrackerBase& tracker_;
  int max_feats_;
  int img_size_;
  int max_disparity_;
  int stereo_wsz_, margin_sz_;
  double ncc_thresh_;
  bool estimate_subpixel_;
  cv::Mat img_lp_, img_rp_, img_lc_, img_rc_;
  std::vector<std::pair<bool, core::DescriptorNCC>> descriptors_rprev_, descriptors_rcurr_;
  //cv::Mat desc_rprev_, desc_rcurr_;
  //std::vector<double> distances_prev_, distances_curr_;

  std::vector<int> age_;
  std::vector<core::Point> pts_left_prev_, pts_left_curr_;
  std::vector<core::Point> pts_right_prev_, pts_right_curr_;
  std::vector<core::Point> df_left_prev_, df_left_curr_;
  std::vector<core::Point> df_right_prev_, df_right_curr_;

  bool use_deformation_field_ = false; 
  cv::Mat left_dx_, left_dy_;
  cv::Mat right_dx_, right_dy_;
  cv::Mat cell_centers_x_;
  cv::Mat cell_centers_y_;
  int img_rows_;
  int img_cols_;
  int cell_width_, cell_height_;
};

inline
FeatureData StereoTracker::getLeftFeatureData(int i)
{
  FeatureData fdata = tracker_.getFeatureData(i);
  return fdata;
}

inline
FeatureData StereoTracker::getRightFeatureData(int i)
{
  FeatureData fdata;
  fdata.feat_ = featureRight(i);
  //fdata.desc_prev_ = desc_rprev_.row(i).reshape(1, stereo_wsz_);
  //fdata.desc_curr_ = desc_rcurr_.row(i).reshape(1, stereo_wsz_);
  return fdata;
}

inline
bool StereoTracker::stereo_match_ncc(const core::DescriptorNCC& desc_left,
                                     const std::vector<std::pair<bool, core::DescriptorNCC>>& descriptors_right,
                                     const core::Point& left_pt, const cv::Mat& img_right,
                                     bool debug, core::Point& right_pt)
{
  bool success = false;
  std::vector<double> costs;
  costs.assign(max_disparity_, std::numeric_limits<double>::max());
  int x = static_cast<int>(left_pt.x_);
  int y = static_cast<int>(left_pt.y_);
  //int min_x = std::max(margin_sz_, int(left_pt.x_) - max_disparity_);
  int max_disp = std::min(max_disparity_, static_cast<int>(left_pt.x_) - margin_sz_);
  int best_d = -1;
  double best_cost = 0.0;
  //for (; x >= min_x; x--, d++) {
  int row_start = y * img_right.cols;
  #pragma omp parallel for
  for (int d = 0; d <= max_disp; d++) {
    int key = row_start + x - d;
    assert(descriptors_right[key].first == true);
    const core::DescriptorNCC& desc_right = descriptors_right[key].second;
    costs[d] = recon::StereoCosts::get_cost_NCC(desc_left, desc_right);
    if (debug) {
      printf("d = %d\nNCC = %f\n\b", d, costs[d]);
      HelperOpencv::DrawPoint(core::Point(left_pt.x_ - d, left_pt.y_), img_right, "right_point");
      HelperOpencv::DrawPoint(core::Point(left_pt.x_ - best_d, left_pt.y_), img_right, "best_right_point");
      HelperOpencv::DrawDescriptor(desc_left.vec, stereo_wsz_, "desc_left");
      HelperOpencv::DrawDescriptor(desc_right.vec, stereo_wsz_, "desc_right");
      int key = cv::waitKey(0);
      if (key == 27) debug = false;
    }
    if (std::isnan(costs[d])) {
      //printf("nan skipped!\n", d, costs[d]);
      continue;
    }
    if (costs[d] > best_cost) {
      best_cost = costs[d];
      best_d = d;
    }
  }
  //printf("Min cost = %d -- d = %d\n", static_cast<int>(min_cost), best_d);
  if (best_d >= 0 && best_cost >= ncc_thresh_) {
    //descriptors_right[best_desc_idx].vec.copyTo(save_descriptor);
    //save_descriptor = descriptors_right[best_desc_idx].vec.t();
    //if (best_d > (max_disparity_ - 2))
    //  printf("Warning: big disp, best cost = %f -- d = %d. max_disp = %d\n", best_cost, best_d, max_disparity_);
      //std::cout << left_pt << "\n";
    success = true;
    right_pt.y_ = left_pt.y_;
    // perform parabolic subpixel interpolation if we can
    if (estimate_subpixel_ && best_d >= 1 && best_d < (max_disp - 1)
       && !std::isnan(costs[best_d-1]) && !std::isnan(costs[best_d+1])) {
      double C_left = 2.0 - costs[best_d-1];
      double C_center = 2.0 - costs[best_d];
      double C_right = 2.0 - costs[best_d+1];
      double d_s = (C_left - C_right) / (2.0*C_left - 4.0*C_center + 2.0*C_right);
      //printf("Parabolic fitting: %d --> %f\n", best_d, best_d+d_s);
      right_pt.x_ = left_pt.x_ - (static_cast<double>(best_d) + d_s);
    }
    else
      right_pt.x_ = left_pt.x_ - static_cast<double>(best_d);
    // perform equiangular subpixel interpolation
    //if (best_d >= 1 && best_d < (max_disp - 1)) {
    //  double C_left = 2.0 - costs[best_d-1];
    //  double C_center = 2.0 - costs[best_d];
    //  double C_right = 2.0 - costs[best_d+1];
    //  double d_s;
    //  if (C_right < C_left)
    //    d_s = 0.5f * (C_right - C_left) / (C_center - C_left);
    //  else
    //    d_s = 0.5f * (C_right - C_left) / (C_center - C_right);
    //  printf("Equangular fitting: %d --> %f\n", best_d, best_d+d_s);
    //  //right_pt.x_ = left_pt.x_ - (static_cast<double>(best_d) + d_s);
    //}
    ////else
    ////  right_pt.x_ = left_pt.x_ - static_cast<double>(best_d);
  }
  else {
    right_pt.x_ = std::numeric_limits<double>::max();
    right_pt.y_ = std::numeric_limits<double>::max();
  }
  assert(!std::isnan(right_pt.x_) && !std::isnan(right_pt.y_));
  return success;
}

inline
void StereoTracker::InterpolateBilinear(const cv::Mat& mat, const int row, const int col,
                                               const double x, const double y, double& ival) {
  double q1 = mat.at<double>(row, col);
  double q2 = mat.at<double>(row, col+1);
  double q3 = mat.at<double>(row+1, col);
  double q4 = mat.at<double>(row+1, col+1);

  double w = cell_width_;
  double h = cell_height_;
  double q12 = ((w-x) / w) * q1 + (x / w) * q2;
  double q34 = ((w-x) / w) * q3 + (x / w) * q4;
  ival = ((h-y) / h) * q12 + (y / h) * q34;
}

inline
void StereoTracker::InterpolateLinear(const double val1, const double val2, const double x,
                                             const double size, double& ival) {
  ival = ((size - x) / size) * val1 + (x / size) * val2;
}

inline
void StereoTracker::GetPointCell(const core::Point& pt, int& row, int& col) {
  col = pt.x_ / cell_width_;
  row = pt.y_ / cell_height_;
  //int bin_num = row * bin_cols_ + col;
}

//inline
//bool StereoTracker::stereo_match_census(int max_disparity, int margin_sz, uint32_t census,
//                                        const cv::Mat& census_img,
//                                        const core::Point& left_pt,
//                                        core::Point& right_pt)
//{
//  bool success = false;
//  std::vector<uint8_t> costs;
//  costs.assign(max_disparity, std::numeric_limits<uint8_t>::max());
//  int cx = static_cast<int>(left_pt.x_) - margin_sz;
//  int cy = static_cast<int>(left_pt.y_) - margin_sz;
//  int max_disp = std::min(max_disparity, cx);
//  int best_d = -1;
//  uint8_t min_cost = std::numeric_limits<uint8_t>::max();
//  for (int d = 0; d <= max_disp; d++) {
//    costs[d] = recon::StereoCosts::hamming_dist(census, census_img.at<uint32_t>(cy,cx-d));
//    if (costs[d] < min_cost) {
//      min_cost = costs[d];
//      best_d = d;
//    }
//  }
//  //printf("Min cost = %d -- d = %d\n", static_cast<int>(min_cost), best_d);
//  if (best_d >= 0 && (int)(min_cost) == 0) {
//    printf("Min cost = %d -- d = %d\n", static_cast<int>(min_cost), best_d);
//    success = true;
//    right_pt.y_ = left_pt.y_;
//    // perform equiangular subpixel interpolation
//    if (best_d >= 1 && best_d < (max_disp - 1) && !std::isnan(costs[best_d-1]) && !std::isnan(costs[best_d+1])) {
//      double C_left = static_cast<double>(costs[best_d-1]);
//      double C_center = static_cast<double>(costs[best_d]);
//      double C_right = static_cast<double>(costs[best_d+1]);
//      double d_s = 0;
//      if (C_right < C_left)
//        d_s = 0.5f * (C_right - C_left) / (C_center - C_left);
//      else
//        d_s = 0.5f * (C_right - C_left) / (C_center - C_right);
//      //std::cout << d << " -- " << d+d_s << "\n";
//      right_pt.x_ = left_pt.x_ - (static_cast<double>(best_d) + d_s);
//    }
//    else
//      right_pt.x_ = left_pt.x_ - static_cast<double>(best_d);
//  }
//  return success;
//}

}

#endif
