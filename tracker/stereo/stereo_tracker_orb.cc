#include "stereo_tracker_orb.h"

#include <chrono>

namespace track {

namespace {

void SortInRows(const std::vector<cv::KeyPoint>& points, size_t img_rows,
                std::vector<std::vector<size_t>>& row_indices) {
  row_indices.resize(img_rows);
  for (size_t i = 0; i < points.size(); i++) {
    int row = static_cast<int>(points[i].pt.y);
    assert(row >= 0 && row < img_rows);
    row_indices[row].push_back(i);
  }
  for (size_t i = 0; i < row_indices.size(); i++) {
    if (row_indices[i].size() > 1)
      std::sort(row_indices[i].begin(), row_indices[i].end(), [&points](size_t a, size_t b) {
            return points[a].pt.x < points[b].pt.x;
            //return points[a].pt.x <= points[b].pt.x;
          });
  }
}

void DrawStereoMatches(const std::vector<cv::KeyPoint>& left,
                const std::vector<cv::KeyPoint>& right,
                const std::vector<std::vector<cv::DMatch>>& matches,
                const cv::Mat& img) {
  cv::Mat disp_img;
  cv::cvtColor(img, disp_img, cv::COLOR_GRAY2BGR);
  for (size_t i = 0; i < matches.size(); i++) {
    if (matches[i].size() > 0)
      cv::arrowedLine(disp_img, left[matches[i][0].queryIdx].pt, right[matches[i][0].trainIdx].pt,
                      cv::Scalar(0,255,0), 1, 8, 0, 0.1);
  }
  cv::imshow("stereo", disp_img);
  cv::waitKey(0);
}

}

void StereoTrackerORB::DrawKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& points,
                                     std::string window_name) const {
  cv::Mat disp_img;
  cv::drawKeypoints(img, points, disp_img, cv::Scalar(0,0,255));
  cv::imshow(window_name, disp_img);
  //cv::waitKey(0);
}

void StereoTrackerORB::DrawFullTracks() const {
  cv::Mat disp_lp, disp_rp, disp_lc, disp_rc;
  int thickness = -1;
  int radius = 3;
  cv::Scalar color = cv::Scalar(0,0,255);
  for (size_t i = 0; i < tracks_lp_.size(); i++) {
    if (age_[i] < 1) continue;
    cv::cvtColor(img_lp_, disp_lp, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img_rp_, disp_rp, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img_lc_, disp_lc, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img_rc_, disp_rc, cv::COLOR_GRAY2BGR);
    //cv::cvtColor(img_lp_, disp_img, cv::COLOR_GRAY2BGR);
    //std::cout << age_[i] << "\n";
    cv::circle(disp_lp, tracks_lp_[i].pt, radius, color, thickness);
    cv::circle(disp_rp, tracks_rp_[i].pt, radius, color, thickness);
    cv::circle(disp_lc, tracks_lc_[i].pt, radius, color, thickness);
    cv::circle(disp_rc, tracks_rc_[i].pt, radius, color, thickness);
    std::cout << "\nTrack ID = " << i << "\n";
    std::cout << "LP = " << tracks_lp_[i].pt << "\n";
    std::cout << "RP = " << tracks_rp_[i].pt << "\n";
    std::cout << "LC = " << tracks_lc_[i].pt << "\n";
    std::cout << "RC = " << tracks_rc_[i].pt << "\n";
    std::cout << "age = " << age_[i] << "\n";
    cv::imshow("left_prev", disp_lp);
    cv::imshow("right_prev", disp_rp);
    cv::imshow("left_curr", disp_lc);
    cv::imshow("right_curr", disp_rc);
    cv::waitKey(0);
  }
}
void StereoTrackerORB::DrawMatches() const {
  cv::Mat disp_img;
  cv::cvtColor(img_lp_, disp_img, cv::COLOR_GRAY2BGR);
  for (size_t i = 0; i < tracks_lp_.size(); i++) {
    if (age_[i] < 1) continue;
    //cv::cvtColor(img_lp_, disp_img, cv::COLOR_GRAY2BGR);
    //std::cout << age_[i] << "\n";
    cv::arrowedLine(disp_img, tracks_lp_[i].pt, tracks_lc_[i].pt, cv::Scalar(0,255,0), 1, 8, 0, 0.1);
  }
  cv::imshow("tracks", disp_img);
  cv::waitKey(0);
}

void StereoTrackerORB::DrawStereo() const {
  cv::Mat disp_img;
  cv::cvtColor(img_lc_, disp_img, cv::COLOR_GRAY2BGR);
  for (size_t i = 0; i < tracks_lp_.size(); i++) {
    if (age_[i] >= 0)
      cv::arrowedLine(disp_img, tracks_lc_[i].pt, tracks_rc_[i].pt, cv::Scalar(0,255,0), 1, 8, 0, 0.1);
  }
  cv::imshow("stereo", disp_img);
  cv::waitKey(0);
}

StereoTrackerORB::StereoTrackerORB(size_t max_tracks, size_t max_xdiff, double max_epipolar_diff,
                                   size_t max_disp, size_t max_disp_diff, int patch_size,
                                   float scale_factor, int num_levels,
                                   int maxdist_stereo, int maxdist_temp) :
      max_tracks_(max_tracks), max_xdiff_(max_xdiff), max_epipolar_diff_(max_epipolar_diff),
      max_disp_(max_disp), max_disp_diff_(max_disp_diff), maxdist_temp_(maxdist_temp),
      maxdist_stereo_(maxdist_stereo) {
  max_ydiff_ = max_xdiff / 2;
  img_rows_ = 0;
  verbose_ = false;

  //max_epipolar_diff_ = 1.0;
  //max_disp_ = 160;
  //max_disp_diff_ = 40;
  //int patch_size = 21;
  //float scale_factor = 1.1;
  //int num_levels = 1;
  //maxdist_stereo_ = 70;
  //maxdist_temp_ = 50;

  //std::cout << "GPU count = " << cv::cuda::getCudaEnabledDeviceCount() << "\n\n";
  // doesnt work on (1)
  cv::cuda::setDevice(0);
  detector_ = cv::cuda::ORB::create(2*max_tracks_, scale_factor, num_levels, patch_size, 0, 2,
                                    cv::cuda::ORB::HARRIS_SCORE, patch_size);
  matcher_ = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
}

void StereoTrackerORB::init(const cv::Mat& img_left, const cv::Mat& img_right) {
  img_rows_ = img_left.rows;
  img_cols_ = img_left.cols;
  tracks_lp_.resize(max_tracks_);
  tracks_rp_.resize(max_tracks_);
  tracks_lc_.resize(max_tracks_);
  tracks_rc_.resize(max_tracks_);
  age_.assign(max_tracks_, -1);
  descriptors_lp_.create(max_tracks_, detector_->descriptorSize(), detector_->descriptorType());
  //index_map_.assign(max_tracks_, -1);

  img_left.copyTo(img_lc_);
  img_right.copyTo(img_rc_);
  gpuimg_left_.upload(img_lc_);
  gpuimg_right_.upload(img_rc_);

  std::vector<cv::KeyPoint> points_left, points_right;
  cv::cuda::GpuMat gpu_points_left, gpu_points_right;
  cv::cuda::GpuMat desc_left, desc_right;
  detector_->detectAndComputeAsync(gpuimg_left_, cv::cuda::GpuMat(), gpu_points_left,
                                   desc_left, false, cuda_stream_);
  detector_->detectAndComputeAsync(gpuimg_right_, cv::cuda::GpuMat(), gpu_points_right,
                                   desc_right, false, cuda_stream_);
  cuda_stream_.waitForCompletion();
  detector_->convert(gpu_points_left, points_left);
  detector_->convert(gpu_points_right, points_right);
  std::vector<std::vector<size_t>> row_indices;
  SortInRows(points_right, img_rows_, row_indices);
  // TODO: faster if we split features in rows and then run maching for each row in async mode
  cv::Mat cpu_mask;
  ApplyEpipolarConstraint(points_left, points_right, row_indices, cpu_mask);
  cv::cuda::GpuMat mask;
  mask.upload(cpu_mask);
  //std::vector<std::vector<cv::DMatch>> matches;
  //matcher_->radiusMatch(desc_left, desc_right, matches, maxdist_stereo_, mask);
  std::vector<std::vector<cv::DMatch>> matches;
  matcher_->radiusMatch(desc_left, desc_right, matches, maxdist_stereo_, mask);
  //std::vector<cv::DMatch> final_matches;
  // use epipolar constraint to filter the matches
  // store them as unused matches
  //cv::Mat cpu_desc_left;
  //desc_left.download(cpu_desc_left);
  assert(matches.size() == points_left.size());
  size_t i = 0;
  for (size_t j = 0; j < matches.size(); j++) {
    assert(i < max_tracks_);
    if (i >= max_tracks_)
      break;
    if (matches[j].size() == 0)
      continue;
    const auto& m = matches[j][0];
    tracks_lc_[i] = points_left[m.queryIdx];
    tracks_rc_[i] = points_right[m.trainIdx];
    desc_left.row(m.queryIdx).copyTo(descriptors_lp_.row(i));
    age_[i] = 0;
    i++;
  }
  if (verbose_) {
    std::cout << "Features detected = " << points_left.size() << " -- " << points_right.size() << "\n";
    std::cout << "Stereo matched = " << i << "\n";
  }
  //DrawStereo();
}

void StereoTrackerORB::track(const cv::Mat& img_left, const cv::Mat& img_right) {
  gpuimg_left_.upload(img_left);
  gpuimg_right_.upload(img_right);
  cv::swap(img_lp_, img_lc_);
  cv::swap(img_rp_, img_rc_);
  img_left.copyTo(img_lc_);
  img_right.copyTo(img_rc_);

  //points_lp_.clear();
  //points_lp_ = std::move(points_lc_);
  std::swap(tracks_lp_, tracks_lc_);
  std::swap(tracks_rp_, tracks_rc_);

  std::vector<cv::KeyPoint> points_left, points_right;
  cv::cuda::GpuMat gpu_points_left, gpu_points_right;
  cv::cuda::GpuMat descriptors_left, descriptors_right;
  detector_->detectAndComputeAsync(gpuimg_left_, cv::cuda::GpuMat(), gpu_points_left,
                                   descriptors_left, false, cuda_stream_);
  detector_->detectAndComputeAsync(gpuimg_right_, cv::cuda::GpuMat(), gpu_points_right,
                                   descriptors_right, false, cuda_stream_);
  cuda_stream_.waitForCompletion();
  detector_->convert(gpu_points_left, points_left);
  detector_->convert(gpu_points_right, points_right);

  //DrawKeypoints(img_lc_, points_lc_, "keypoints left");
  //DrawKeypoints(img_rc_, points_right, "keypoints right");


  // Perform stereo and teporal matching independently
  //int k = 3;
  std::vector<std::vector<size_t>> row_indices;
  SortInRows(points_right, img_rows_, row_indices);
  cv::cuda::GpuMat gpu_temp_matches, gpu_stereo_matches;
  cv::Mat cpu_mask_stereo, cpu_mask_temp;
  cv::cuda::GpuMat mask_stereo, mask_temp;
  std::vector<cv::DMatch> temp_matches_compact, stereo_matches_compact;
  ApplyEpipolarConstraint(points_left, points_right, row_indices, cpu_mask_stereo);
  mask_stereo.upload(cpu_mask_stereo);
  matcher_->matchAsync(descriptors_left, descriptors_right, gpu_stereo_matches,
                       mask_stereo, cuda_stream_);
  //matcher_->match(descriptors_left, descriptors_right, stereo_matches_compact, mask_stereo);

  ApplyTemporalConstraint(points_left, cpu_mask_temp);
  mask_temp.upload(cpu_mask_temp);
  matcher_->matchAsync(descriptors_lp_, descriptors_left, gpu_temp_matches,
                       mask_temp, cuda_stream_);
  cuda_stream_.waitForCompletion();
  //matcher_->match(descriptors_lp_, descriptors_left, temp_matches, mask_temp);

  matcher_->matchConvert(gpu_stereo_matches, stereo_matches_compact);
  matcher_->matchConvert(gpu_temp_matches, temp_matches_compact);

  //auto start = std::chrono::system_clock::now();
  //cv::Mat tmatches, smatches;
  //gpu_temp_matches.download(tmatches);
  //gpu_stereo_matches.download(smatches);
  //std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
  //std::cout << "[Matching]: Time = " << elapsed.count() << " sec\n";
  //std::cout << tmatches.rows << " x " << tmatches.cols << "\n" << smatches.size() << "\n";

  //DrawStereoMatches(points_left, points_right, stereo_matches, img_lc_);

  temp_matches_status_.assign(tracks_lp_.size(), false);
  std::vector<int> stereo_matches(points_left.size(), -1);
  std::vector<bool> used_matches(points_left.size(), false);
  //#pragma omp parallel for
  for (size_t i = 0; i < stereo_matches_compact.size(); i++) {
    if (stereo_matches_compact[i].distance <= maxdist_stereo_) {
      int left_idx = stereo_matches_compact[i].queryIdx;
      int right_idx = stereo_matches_compact[i].trainIdx;
      stereo_matches[left_idx] = right_idx;
    }
  }
  // choose the best match which has disp_diff below threshold
  for (const auto& m : temp_matches_compact) {
    int curr_idx = m.trainIdx;
    if (stereo_matches[curr_idx] >= 0 && m.distance <= maxdist_temp_) {
      used_matches[curr_idx] = true;
      temp_matches_status_[m.queryIdx] = true;
      tracks_lc_[m.queryIdx] = points_left[curr_idx];
      // TODO take the best where disp_diff < max_disp_diff
      tracks_rc_[m.queryIdx] = points_right[stereo_matches[curr_idx]];
      age_[m.queryIdx]++;
      descriptors_left.row(curr_idx).copyTo(descriptors_lp_.row(m.queryIdx));
    }
    else age_[m.queryIdx] = -1;
  }

  alive_cnt_ = 0;
  alive_indices_.clear();
  for (size_t i = 0; i < age_.size(); i++) {
    // clear unmatched potential tracks [age == 0] and older tracks [age > 0] to make room for new ones
    if (temp_matches_status_[i] == false && age_[i] >= 0)
      age_[i] = -1;
    else if (age_[i] > 0) {
      alive_cnt_++;
      alive_indices_.push_back(i);
    }
  }

  size_t cnt = 0;
  for (size_t i = 0; i < used_matches.size(); i++) {
    int right_idx = stereo_matches[i];
    if (used_matches[i] == false && right_idx >= 0) {
      assert(cnt < age_.size());
      while (cnt < age_.size()) {
        if (age_[cnt] < 0) {
          tracks_lc_[cnt] = points_left[i];
          tracks_rc_[cnt] = points_right[right_idx];
          descriptors_left.row(i).copyTo(descriptors_lp_.row(cnt));
          age_[cnt++] = 0;
          break;
        }
        cnt++;
      }
    }
  }

  if (verbose_)
    std::cout << "Matched tracks = " << countActiveTracks() << "\n";
  //DrawMatches();
  //DrawFullTracks();
}

FeatureInfo StereoTrackerORB::featureLeft(int i) const {
  FeatureInfo feat;
  feat.age_ = age_[i];
  feat.prev_.x_ = tracks_lp_[i].pt.x;
  feat.prev_.y_ = tracks_lp_[i].pt.y;
  feat.curr_.x_ = tracks_lc_[i].pt.x;
  feat.curr_.y_ = tracks_lc_[i].pt.y;
  return feat;
}

FeatureInfo StereoTrackerORB::featureRight(int i) const {
  FeatureInfo feat;
  feat.age_ = age_[i];
  feat.prev_.x_ = tracks_rp_[i].pt.x;
  feat.prev_.y_ = tracks_rp_[i].pt.y;
  feat.curr_.x_ = tracks_rc_[i].pt.x;
  feat.curr_.y_ = tracks_rc_[i].pt.y;
  return feat;
}

void StereoTrackerORB::ApplyEpipolarConstraint(
    const std::vector<cv::KeyPoint>& points_left,
    const std::vector<cv::KeyPoint>& points_right,
    const std::vector<std::vector<size_t>>& row_indices,
    cv::Mat& mask) const {
  mask = cv::Mat::zeros(points_left.size(), points_right.size(), CV_8U);
  int row_range = std::ceil(max_epipolar_diff_);
  #pragma omp parallel for
  for (size_t i = 0; i < points_left.size(); i++) {
    //for (size_t j = 0; j < points_right.size(); j++) {
    //  const cv::KeyPoint& left = points_left[i];
    //  const cv::KeyPoint& right = points_right[j];
    //  double disp = left.pt.x - right.pt.x;
    //  if (std::abs(left.pt.y - right.pt.y) <= max_epipolar_diff_ && disp >= 0 && disp <= max_disp_)
    //    mask.at<uint8_t>(i,j) = 1;
    //}
    int row = points_left[i].pt.y;
    int start_row = std::max(0, row - row_range);
    int end_row = std::min((int)row_indices.size()-1, row + row_range);
    const cv::KeyPoint& left = points_left[i];
    for (int j = start_row; j <= end_row; j++) {
      for (size_t right_idx : row_indices[j]) {
        const cv::KeyPoint& right = points_right[right_idx];
        double disp = left.pt.x - right.pt.x;
        if (disp < 0)
          continue;
        if (std::abs(left.pt.y - right.pt.y) <= max_epipolar_diff_ && disp <= max_disp_)
          mask.at<uint8_t>(i,right_idx) = 1;
      }
    }
  }
}

void StereoTrackerORB::ApplyTemporalConstraint(
    const std::vector<cv::KeyPoint>& points_curr,
    cv::Mat& mask) const {
  mask = cv::Mat::zeros(tracks_lp_.size(), points_curr.size(), CV_8U);
  std::vector<size_t> active;
  // take all active (age > 0) and potential matches (age == 0)
  for (size_t i = 0; i < age_.size(); i++)
    if (age_[i] >= 0)
      active.push_back(i);
  #pragma omp parallel for
  for (size_t i = 0; i < active.size(); i++) {
    const cv::KeyPoint& prev = tracks_lp_[active[i]];
    for (size_t j = 0; j < points_curr.size(); j++) {
      const cv::KeyPoint& curr = points_curr[j];
      if (std::abs(prev.pt.y - curr.pt.y) <= max_ydiff_ &&
          std::abs(prev.pt.x - curr.pt.x) <= max_xdiff_)
        mask.at<uint8_t>(active[i],j) = 1;
    }
  }
}

}   // namespace track
