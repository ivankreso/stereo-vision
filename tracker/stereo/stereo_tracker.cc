#include "stereo_tracker.h"

//#define DEBUG_ON

namespace
{

void SmoothImage(core::Image& img, core::Image& img_smooth)
{
  cv::Mat cvimg_tmp, cvimg_smooth;
  track::HelperOpencv::ImageToMat(img, cvimg_tmp);
  //cv::imshow("img_nosmooth", cvimg_tmp);
  cv::GaussianBlur(cvimg_tmp, cvimg_smooth, cv::Size(3,3), 0.7);
  //cv::imshow("img_smooth", cvimg_smooth);
  track::HelperOpencv::MatToImage(cvimg_smooth, img_smooth);
}

void DrawFeature(core::Point& pt, cv::Scalar& color, cv::Mat& img)
{
  cv::Point cvpt;
  cvpt.x = pt.x_;
  cvpt.y = pt.y_;
  cv::circle(img, cvpt, 2, color, -1, 8);
}

}

namespace track
{

StereoTracker::StereoTracker(TrackerBase& tracker, int max_disparity, int stereo_wsz,
                             double ncc_thresh, bool estimate_subpixel,
                             bool use_df, const std::string& deformation_field_path) :
                             tracker_(tracker), max_feats_(tracker.countFeatures()),
                             max_disparity_(max_disparity), stereo_wsz_(stereo_wsz),
                             ncc_thresh_(ncc_thresh), estimate_subpixel_(estimate_subpixel) {
  pts_right_prev_.assign(max_feats_, core::Point(0,0));
  pts_right_curr_.assign(max_feats_, core::Point(0,0));
  pts_left_prev_.assign(max_feats_, core::Point(0,0));
  pts_left_curr_.assign(max_feats_, core::Point(0,0));
  age_.resize(max_feats_);
  margin_sz_ = stereo_wsz / 2;

  use_deformation_field_ = use_df;
  if (use_df) {
    df_right_prev_.assign(max_feats_, core::Point(0,0));
    df_right_curr_.assign(max_feats_, core::Point(0,0));
    df_left_prev_.assign(max_feats_, core::Point(0,0));
    df_left_curr_.assign(max_feats_, core::Point(0,0));
    cv::FileStorage mat_file(deformation_field_path, cv::FileStorage::READ);
    if (!mat_file.isOpened()) {
      std::cout << "Deformation file missing!\n";
      throw 1;
    }
    mat_file["left_dx"] >> left_dx_;
    mat_file["left_dy"] >> left_dy_;
    mat_file["right_dx"] >> right_dx_;
    mat_file["right_dy"] >> right_dy_;
    mat_file["img_rows"] >> img_rows_;
    mat_file["img_cols"] >> img_cols_;

    cell_width_ = (double)img_cols_ / left_dx_.cols;
    cell_height_ = (double)img_rows_ / left_dx_.rows;

    ComputeCellCenters();
  }
}

void StereoTracker::init(const cv::Mat& img_left, const cv::Mat& img_right) {
  img_left.convertTo(img_lc_, kPixelTypeOpenCV);
  img_right.convertTo(img_rc_, kPixelTypeOpenCV);
  cv::GaussianBlur(img_lc_, img_lc_, cv::Size(3,3), 0.7);
  cv::GaussianBlur(img_rc_, img_rc_, cv::Size(3,3), 0.7);
  //std::cout << img_lc_;

  //cv::GaussianBlur(img_left, img_lc_, cv::Size(3,3), 0.7);
  //cv::GaussianBlur(img_right, img_rc_, cv::Size(3,3), 0.7);

  //recon::StereoCosts::census_transform(img_right, stereo_wsz_, census_rcurr_);
  //recon::StereoCosts::compute_image_ncc_descriptors(img_rc_, stereo_wsz_, descriptors_rcurr_);
  img_size_ = img_left.rows * img_left.cols;
  descriptors_rprev_.assign(img_size_, std::make_pair(false, core::DescriptorNCC()));
  descriptors_rcurr_.assign(img_size_, std::make_pair(false, core::DescriptorNCC()));
  tracker_.init(img_lc_);
}

void StereoTracker::track(const cv::Mat& img_left, const cv::Mat& img_right) {
  cv::swap(img_lp_, img_lc_);
  cv::swap(img_rp_, img_rc_);

  img_left.convertTo(img_lc_, kPixelTypeOpenCV);
  img_right.convertTo(img_rc_, kPixelTypeOpenCV);
  cv::GaussianBlur(img_lc_, img_lc_, cv::Size(3,3), 0.7);
  cv::GaussianBlur(img_rc_, img_rc_, cv::Size(3,3), 0.7);

  //cv::GaussianBlur(img_left, img_lc_, cv::Size(3,3), 0.7);
  //cv::GaussianBlur(img_right, img_rc_, cv::Size(3,3), 0.7);

  tracker_.track(img_lc_);

  std::swap(pts_left_prev_, pts_left_curr_);
  std::swap(pts_right_prev_, pts_right_curr_);
  if (use_deformation_field_) {
    std::swap(df_left_prev_, df_left_curr_);
    std::swap(df_right_prev_, df_right_curr_);
  }
  //pts_left_prev_ = pts_left_curr_;
  //pts_right_prev_ = pts_right_curr_;
  //if (use_deformation_field_) {
  //  df_left_prev_ = df_left_curr_;
  //  df_right_prev_ = df_right_curr_;
  //}

  std::swap(descriptors_rprev_, descriptors_rcurr_);
  #pragma omp parallel for
  for (size_t i = 0; i <  descriptors_rcurr_.size(); i++)
    descriptors_rcurr_[i].first = false;

  // precompute the descriptors for all image pixels
  //recon::StereoCosts::census_transform(img_right, stereo_wsz_, census_rcurr_);
  //recon::StereoCosts::compute_image_ncc_descriptors(img_rc_, stereo_wsz_, descriptors_rcurr_);
  int alive_before = tracker_.countTracked();

  // precompute NCC plain patch descriptors in current right image
  //std::cout << "COMPUTE NCC begin\n";
  for (int i = 0; i < tracker_.countFeatures(); i++) {
    if (tracker_.isAlive(i)) {
      FeatureInfo pts = tracker_.feature(i);
      // if new only compute descriptors in previous image
      if (pts.age_ == 1) {
        // precompute any missing NCC plain patch descriptors in previous right image
        AddMissingDescriptors(img_rp_, pts.prev_, stereo_wsz_, descriptors_rprev_);
      }
      // compute descriptors in current image
      AddMissingDescriptors(img_rc_, pts.curr_, stereo_wsz_, descriptors_rcurr_);
    }
  }
  //std::cout << "COMPUTE NCC end\n";

#ifndef DEBUG_ON
  #pragma omp parallel for
#endif
  for(int i = 0; i < tracker_.countFeatures(); i++) {
    bool debug = false;
#ifdef DEBUG_ON
    debug = true;
#endif
    //if (i == 273)
    //  debug = true;
    if (tracker_.isAlive(i)) {
      bool ok = false;
      //uint32_t census;
      core::DescriptorNCC ncc_desc;
      FeatureInfo left_feat = tracker_.feature(i);
      // if new feature we need to find disparity for both frames
      if (left_feat.age_ == 1) {
        // census is to robust...
        // census = recon::StereoCosts::census_transform_point(left_feat.prev_, img_lp_, stereo_wsz_);
        // ok = stereo_match_census(max_disparity_, margin_sz, census, census_rprev_,
        //                         left_feat.prev_, pts_right_prev_[i]);
        // NCC
        if(debug) {
          std::cout << "Prev frame:\n";
          HelperOpencv::DrawPoint(left_feat.prev_, img_lp_, "left_point");
        }
        recon::StereoCosts::compute_ncc_descriptor<PixelType>(img_lp_, left_feat.prev_, stereo_wsz_,
                                                              kPixelTypeOpenCV, ncc_desc);
        ok = stereo_match_ncc(ncc_desc, descriptors_rprev_, left_feat.prev_, img_rp_, debug,
                              pts_right_prev_[i]);
        if(!ok) {
          tracker_.removeTrack(i);
          continue;
        }
      }
      // NCC
      //HelperOpencv::DrawPoint(left_feat.curr_, img_lc_, "left_point");
      if(debug) {
        std::cout << "Current frame:\n";
        HelperOpencv::DrawPoint(left_feat.curr_, img_lc_, "left_point");
      }
      // find disparity for in current
      recon::StereoCosts::compute_ncc_descriptor<PixelType>(img_lc_, left_feat.curr_, stereo_wsz_,
                                                            kPixelTypeOpenCV, ncc_desc);
      ok = stereo_match_ncc(ncc_desc, descriptors_rcurr_, left_feat.curr_, img_rc_, debug,
                            pts_right_curr_[i]);
      if(!ok)
        tracker_.removeTrack(i);
      // if ok pull the left tracker data
      else {
        if (left_feat.age_ == 1)
          pts_left_prev_[i] = left_feat.prev_;
        pts_left_curr_[i] = left_feat.curr_;
        // Apply deformation field
        if (use_deformation_field_) {
          if (left_feat.age_ == 1) {
            df_left_prev_[i] = pts_left_prev_[i];
            df_right_prev_[i] = pts_right_prev_[i];
            ApplyDeformationField(left_dx_, left_dy_, df_left_prev_[i]);
            ApplyDeformationField(right_dx_, right_dy_, df_right_prev_[i]);
          }
          df_left_curr_[i] = pts_left_curr_[i];
          df_right_curr_[i] = pts_right_curr_[i];
          ApplyDeformationField(left_dx_, left_dy_, df_left_curr_[i]);
          ApplyDeformationField(right_dx_, right_dy_, df_right_curr_[i]);
        }
      }
    }
  }

  int alive_after = 0;
  for(int i = 0; i < tracker_.countFeatures(); i++) {
    age_[i] = tracker_.getAge(i);
    if (age_[i] > 0)
      alive_after++;
  }
  assert(alive_after <= alive_before);
  std::cout << "[StereoTracker]: Matched = " << alive_after << " / " << alive_before << "\n";
}

int StereoTracker::countFeatures() const {
   return max_feats_;
}

int StereoTracker::countActiveTracks() const {
  int cnt = 0;
  for(int i = 0; i < max_feats_; i++) {
    if(age_[i] > 0)
      cnt++;
  }
  return cnt;
}

FeatureInfo StereoTracker::featureLeft(int i) const {
  FeatureInfo feat;
  if (!use_deformation_field_) {
    feat.prev_ = pts_left_prev_[i];
    feat.curr_ = pts_left_curr_[i];
  }
  else {
    feat.prev_ = df_left_prev_[i];
    feat.curr_ = df_left_curr_[i];
  }
  feat.age_ = age_[i];
  return feat;
}

FeatureInfo StereoTracker::featureRight(int i) const {
  FeatureInfo feat;
  if (!use_deformation_field_) {
    feat.prev_ = pts_right_prev_[i];
    feat.curr_ = pts_right_curr_[i];
  }
  else {
    feat.prev_ = df_right_prev_[i];
    feat.curr_ = df_right_curr_[i];
  }
  feat.age_ = age_[i];
  return feat;
}

void StereoTracker::removeTrack(int id) {
  //if(age_[id] > 0) {
  //  age_acc_ += age_[id];
  //  death_count_++;
  //}
  age_[id] = -1;
  tracker_.removeTrack(id);
}

void StereoTracker::ComputeCellCenters() {
  int rows = left_dx_.rows;
  int cols = left_dx_.cols;
  cell_centers_x_ = cv::Mat::zeros(rows, cols, CV_64F);
  cell_centers_y_ = cv::Mat::zeros(rows, cols, CV_64F);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cell_centers_x_.at<double>(i,j) = (j * cell_width_) + (cell_width_ / 2.0);
      cell_centers_y_.at<double>(i,j) = (i * cell_height_) + (cell_height_ / 2.0);
    }
  }
  //std::cout << cell_centers_x_ << "\n\n" << cell_centers_y_ << "\n";
}

void StereoTracker::ApplyDeformationField(const cv::Mat& def_x, const cv::Mat& def_y,
                                          core::Point& pt)
{
  double dx, dy;
  // number of rows/cols in interpolation grid is smaller by 1
  int rows = left_dx_.rows - 1;
  int cols = left_dx_.cols - 1;
  double half_width = cell_width_ / 2.0;
  double half_height = cell_height_ / 2.0;
  int real_row = static_cast<int>(pt.y_ / cell_height_);
  int real_col = static_cast<int>(pt.x_ / cell_width_);
  int row = static_cast<int>(std::floor((pt.y_ - half_height) / cell_height_));
  int col = static_cast<int>(std::floor((pt.x_ - half_width) / cell_width_));
  double cell_x = 0.0, cell_y = 0.0;
  if (row >= 0)
    cell_x = (pt.x_ - half_width) - (col * cell_width_);
  if (col >= 0)
    cell_y = (pt.y_ - half_height) - (row * cell_height_);

  // compute bilinear interpolation
  if (row >= 0 && row < rows && col >= 0 && col < cols) {
    assert(cell_x >= 0.0 && cell_x <= cell_width_);
    assert(cell_y >= 0.0 && cell_y <= cell_height_);
    InterpolateBilinear(def_x, row, col, cell_x, cell_y, dx);
    InterpolateBilinear(def_y, row, col, cell_x, cell_y, dy);
  }
  // compute left-right liner interpolation on horizontal edges
  else if ((row < 0 || row >= rows) && (col >= 0 || col < cols)) {
    assert(cell_x >= 0.0 && cell_x <= cell_width_);
    double q1 = def_x.at<double>(real_row, col);
    double q2 = def_x.at<double>(real_row, col+1);
    InterpolateLinear(q1, q2, cell_x, cell_width_, dx);
    q1 = def_y.at<double>(real_row, col);
    q2 = def_y.at<double>(real_row, col+1);
    InterpolateLinear(q1, q2, cell_x, cell_width_, dy);
  }
  // compute up-down linear interpolation on vertical edges
  else if ((row >= 0 || row < rows) && (col < 0 || col >= cols)) {
    assert(cell_y >= 0.0 && cell_y <= cell_height_);
    double q1 = def_x.at<double>(row, real_col);
    double q2 = def_x.at<double>(row+1, real_col);
    InterpolateLinear(q1, q2, cell_y, cell_height_, dx);
    q1 = def_y.at<double>(row, real_col);
    q2 = def_y.at<double>(row+1, real_col);
    InterpolateLinear(q1, q2, cell_y, cell_height_, dy);
  }
  // we can't interpolate on corners
  else {
    dx = left_dx_.at<double>(real_row, real_col);
    dy = left_dy_.at<double>(real_row, real_col);
  }
  pt.x_ += dx;
  pt.y_ += dy;
}

void StereoTracker::AddMissingDescriptors(const cv::Mat& img, const core::Point& point,
    int window_size, std::vector<std::pair<bool, core::DescriptorNCC>>& descriptors) {
  int y = int(point.y_);
  //int min_x = std::max(margin_sz_, int(point.x_) - max_disparity);
  int max_disp = std::min(max_disparity_, static_cast<int>(point.x_) - margin_sz_);
  int row_start = y * img.cols;
  int pt_x = static_cast<int>(point.x_);
  //for (int x = pt_x; x >= min_x; x--) {
  #pragma omp parallel for
  for (int d = 0; d <= max_disp; d++) {
    int x = pt_x - d;
    //printf("%d\n", x);
    int key = row_start + x;
    // if descriptor is not already extracted do it
    if (descriptors[key].first == false) {
      descriptors[key].first = true;
      recon::StereoCosts::compute_ncc_descriptor<PixelType>(img, x, y, window_size, kPixelTypeOpenCV,
                                                            descriptors[key].second);
    }
  }
}

void StereoTracker::showTrack(int i) const
{
  cv::Mat img_lp, img_lc, img_rp, img_rc;
  cv::cvtColor(img_lp_, img_lp, cv::COLOR_GRAY2RGB);
  cv::cvtColor(img_lc_, img_lc, cv::COLOR_GRAY2RGB);
  cv::cvtColor(img_rp_, img_rp, cv::COLOR_GRAY2RGB);
  cv::cvtColor(img_rc_, img_rc, cv::COLOR_GRAY2RGB);
  FeatureInfo feat_left = featureLeft(i);
  FeatureInfo feat_right = featureRight(i);
  if(feat_left.age_ <= 0) throw "Error\n";
  cv::Scalar color(0,255,0);
  DrawFeature(feat_left.prev_, color, img_lp);
  DrawFeature(feat_left.curr_, color, img_lc);
  DrawFeature(feat_right.prev_, color, img_rp);
  DrawFeature(feat_right.curr_, color, img_rc);

  double ssz = 1.6;
  cv::resize(img_lp, img_lp, cv::Size(), ssz, ssz);
  cv::resize(img_rp, img_rp, cv::Size(), ssz, ssz);
  cv::resize(img_lc, img_lc, cv::Size(), ssz, ssz);
  cv::resize(img_rc, img_rc, cv::Size(), ssz, ssz);
  cv::imshow("left_prev", img_lp);
  cv::imshow("left_curr", img_lc);
  cv::imshow("right_prev", img_rp);
  cv::imshow("right_curr", img_rc);
  cv::waitKey(0);
}

}
