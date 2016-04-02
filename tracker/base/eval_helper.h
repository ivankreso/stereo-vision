#ifndef CORE_EVAL_HELPER_H_
#define CORE_EVAL_HELPER_H_

#include <vector>
#include <iostream>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "../../core/types.h"
#include "../stereo/stereo_tracker_base.h"

namespace track {

class EvalHelper
{
public:
  static void DrawDeformationFieldParams(
      const int bin_rows, const int bin_cols, const double* left_dx, const double* left_dy,
      const double* right_dx, const double* right_dy, const uint64_t* left_num_points,
      const uint64_t* right_num_points, cv::Mat& img);

  static void SaveTrackerEvaluation(
    track::StereoTrackerBase& tracker, const double* cam_params,
    const cv::Mat& cvRt, double max_error);
  static size_t DrawTracksWithErrors(
    track::StereoTrackerBase& tracker, const double* cam_params, const cv::Mat& cvRt,
    const cv::Mat& img_lp, const cv::Mat& img_rp, const cv::Mat& img_lc, const cv::Mat& img_rc,
    double min_error);

  static void DrawTracksWithBigErrors(track::StereoTrackerBase& tracker, const double* cam_params,
                                      const cv::Mat& cvRt, const double min_error, const double max_error,
                                      const int num, const cv::Mat& img_lp, const cv::Mat& img_rp,
                                      const cv::Mat& img_lc, const cv::Mat& img_rc,
                                      const cv::Scalar& color, bool redraw_image,
                                      cv::Mat& lp_draw, cv::Mat& rp_draw, cv::Mat& lc_draw, cv::Mat& rc_draw);

  static void GetStereoReprojErrors(const core::Point& pt_left_prev, const core::Point& pt_right_prev,
                                    const core::Point& pt_left_curr, const core::Point& pt_right_curr,
                                    const Eigen::Matrix4d& Rt, const double* cam_params,
                                    double& left_error, double& right_error,
                                    Eigen::Vector2d& left_vec_error, Eigen::Vector2d& right_vec_error);

  static void GetStereoReprojErrors(const core::Point& pt_left_prev, const core::Point& pt_right_prev,
                                    const core::Point& pt_left_curr, const core::Point& pt_right_curr,
                                    const Eigen::Matrix4d& Rt, const double* cam_params,
                                    double& error_3d, double& left_error, double& right_error);

  static bool FilterOutliersWithGroundtruth(track::StereoTrackerBase& tracker, const double* cam_params,
                                            const cv::Mat& cvRt, const double error_thr);

  static void DrawStereoTrack(track::StereoTrackerBase& tracker, int i, cv::Mat& img_lp, cv::Mat& img_rp,
                              cv::Mat& img_lc, cv::Mat& img_rc, const int pt_size, cv::Scalar color);
  static void DrawTrackPatches(track::StereoTrackerBase& tracker, int track_num, int win_size,
                               const cv::Mat& img_lp, const cv::Mat& img_rp,
                               const cv::Mat& img_lc, const cv::Mat& img_rc);
  static void DrawPointPatch(const core::Point& pt, const cv::Mat& img, const int size,
                             size_t resolution, bool save_on_disk, const std::string filename);

  static void CalculateErrorStatistics(const std::vector<std::vector<double>>& reproj_errors,
                                       const std::vector<std::vector<Eigen::Vector2d>>& error_vectors,
                                       std::vector<double>& means, std::vector<double>& variances,
                                       std::vector<Eigen::Vector2d>& vec_means,
                                       std::vector<Eigen::Vector2d>& vec_variances);

  static void CalculateErrorStatistics(const std::vector<std::vector<double>>& left_reproj_errors,
                                       const std::vector<std::vector<double>>& right_reproj_errors,
                                       const std::vector<std::vector<Eigen::Vector2d>>& left_error_vectors,
                                       const std::vector<std::vector<Eigen::Vector2d>>& right_error_vectors,
                                       std::vector<double>& means, std::vector<double>& variances,
                                       std::vector<Eigen::Vector2d>& vec_means,
                                       std::vector<Eigen::Vector2d>& vec_variances);

  static void CalculateReprojectionErrors(track::StereoTrackerBase& tracker, const double* cam_params,
                                          const cv::Mat& cvRt, const int img_rows, const int img_cols,
                                          std::vector<std::vector<double>>& reproj_errors_left,
                                          std::vector<std::vector<double>>& reproj_errors_right,
                                          std::vector<std::vector<Eigen::Vector2d>>& left_error_vectors,
                                          std::vector<std::vector<Eigen::Vector2d>>& right_error_vectors,
                                          int h_bins, int v_bins);

  static void SaveErrorStatistics(const int rows, const int cols,
                                  const std::vector<std::vector<double>>& reproj_errors,
                                  const std::vector<double>& means, const std::vector<double>& variances,
                                  const std::vector<Eigen::Vector2d>& vec_means,
                                  const std::vector<Eigen::Vector2d>& vec_variances,
                                  const std::string file_name);

  static void DrawErrorStatistics(const int bin_rows, const int bin_cols,
                                  const std::vector<double>& means, const std::vector<double>& variances,
                                  const std::vector<Eigen::Vector2d>& vec_means,
                                  const std::vector<Eigen::Vector2d>& vec_variances,
                                  const std::vector<std::vector<double>>& reproj_errors,
                                  const bool draw_stats, cv::Mat& img);

  static int CountFilterStoreBadTracks(track::StereoTrackerBase& tracker, const double* cam_params,
                                       const cv::Mat& cvRt, bool save_patches,
                                       const std::string& good_folder, const std::string& bad_folder,
                                       std::vector<size_t>& track_index, std::vector<size_t>& track_cnt,
                                       size_t& all_tracks_cnt, bool filter_bad,
                                       double reproj_error_thr, double max_remove_ratio);

  static void DrawTracksAndErrors(track::StereoTrackerBase& tracker, const double* cam_params,
                                  const cv::Mat& cvRt, const cv::Mat& img_lp, const cv::Mat& img_rp,
                                  const std::string& left_name,
                                  bool save_errors,
                                  std::vector<std::vector<double>>& reproj_errors_left,
                                  std::vector<std::vector<double>>& reproj_errors_right,
                                  int h_bins, int v_bins, bool draw_on = false,
                                  double thr = std::numeric_limits<double>::max(),
                                  bool filter_bad = false);

  static void DrawErrorDistribution(int h_bins, int v_bins,
                                    std::vector<std::vector<double>>& reproj_errors,
                                    cv::Mat& img, bool draw_stats, bool save_stats);

  static void DrawStereoTracks(StereoTrackerBase& tracker,
                               const cv::Mat& img_left,
                               const cv::Mat& img_right,
                               const std::string& name_left,
                               const std::string& name_right);

  static double GetStereoReprojError(const track::StereoTrackerBase& tracker,
                                     const double* cam_params, cv::Mat& Rt);

  static double getStereoReprojError(std::vector<core::Point>& points_lp, std::vector<core::Point>& points_rp,
                                     std::vector<core::Point>& points_lc, std::vector<core::Point>& points_rc,
                                     cv::Mat& C, cv::Mat& Rt, double baseline);

  static double getStereoRefinerDepthError(track::StereoTrackerBase& tracker_base,
                                           track::StereoTrackerBase& tracker,
                                           const double (&cam_params)[5], const cv::Mat& depth_mat,
                                           const cv::Mat lp_img, const cv::Mat rp_img,
                                           const cv::Mat lc_img, const cv::Mat rc_img);

  static double getStereoDepthError(track::StereoTrackerBase& tracker, const double (&cam_params)[5],
                             const cv::Mat& depth_mat);

  static double getStereoDepthError(track::StereoTrackerBase& tracker, const double (&cam_params)[5],
                                       const cv::Mat& depth_mat, cv::Mat lp_img, cv::Mat rp_img,
                                       cv::Mat lc_img, cv::Mat rc_img);

  //static double getStereoDepthError(std::vector<core::Point>& pts_l, std::vector<core::Point>& pts_r,
  //                                  const double (&cam_params)[5], const cv::Mat& depth_mat,
  //                                  const cv::Mat lp_img, const cv::Mat rp_img,
  //                                  const cv::Mat lc_img, const cv::Mat rc_img);


  static void drawStereoTrack(const core::Point& pt_l, const core::Point& pt_r,
                              cv::Mat& img_l, cv::Mat& img_r);

  static void voPoint2cvMat(core::Point& pt, cv::Mat& mat);
};

}

#endif
