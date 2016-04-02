#ifndef _FEATURE_HELPER_
#define _FEATURE_HELPER_

#include <vector>

#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "../tracker/base/types.h"
#include "../tracker/stereo/stereo_tracker_base.h"
#include "extern/libviso2/src/viso_stereo.h"

namespace vo {

class FeatureHelper
{
 public:
  static void FilterRansacOutliers(track::StereoTrackerBase& tracker,
                                   const std::vector<int>& active_tracks,
                                   const std::vector<int>& inliers);

  static void FilterTracksWithPriorOld(track::StereoTrackerBase& tracker,
                                       double max_disp_diff, double min_disp);

  static void FilterRansacOutliersWithPrior(track::StereoTrackerBase& tracker,
                                            const std::vector<int>& active_tracks,
                                            const std::vector<int>& inliers,
                                            double max_disp_diff, double min_disp);

  static void LibvisoToTrackerBase(std::vector<libviso::Matcher::p_match>& matches,
      std::vector<track::FeatureInfo>& feats_left,
      std::vector<track::FeatureInfo>& feats_right);

  static void TrackerBaseToLibviso(track::StereoTrackerBase* tracker,
                                   std::vector<libviso::Matcher::p_match>& matches,
                                   std::vector<int>& active_tracks);

  // old stuff
  static void TrackerBaseToLibviso(track::StereoTrackerBase* tracker,
      std::vector<libviso::Matcher::p_match>& matches);

  //static void FilterOutlierTracks(track::StereoTrackerBase& tracker, const std::vector<int>& active_tracks,
  //                                const std::vector<int>& inliers);

  static void TrackerBaseToLibvisoStratified(const track::StereoTrackerBase* tracker,
      std::vector<libviso::Matcher::p_match>& matches_strat,
      int max_features, double block_width, double block_height,
      core::Size img_size, std::vector<int>& active_tracks);

  static void TrackerBaseToLibvisoUniform(const track::StereoTrackerBase* tracker,
    std::vector<libviso::Matcher::p_match>& matches_strat, int max_features, int rows, int cols,
    core::Size img_size, std::vector<int>& active_tracks);

  static void LibvisoInliersToPoints(std::vector<libviso::Matcher::p_match>& matches, std::vector<int>& inliers,
      std::vector<core::Point>& points_lp, std::vector<core::Point>& points_rp,
      std::vector<core::Point>& points_lc, std::vector<core::Point>& points_rc);

  static void getActiveTracks(track::StereoTrackerBase* tracker, std::vector<int>& active_tracks);


  //static void filterOutlierTracks(track::StereoTrackerBase& tracker, const std::vector<int>& active_tracks,
  //                                const std::vector<int>& inliers, std::vector<int>& outliers);
  static void filterOutlierTracks(track::StereoTrackerBase& tracker, const cv::Mat& Rt,
                                  const double (&cam_params)[5], std::vector<int>& outliers, const double eps);


  static int filterBadTracks(track::StereoTrackerBase& tracker);

  static void drawStereoRefinerTracks(track::StereoTrackerBase& tracker,
                                      track::StereoTrackerBase& tracker_refiner,
                                      cv::Mat& img_lp, cv::Mat& img_rp);

  static void drawStereoTracks(track::StereoTrackerBase& tracker, const std::vector<int>& tracks,
                               cv::Mat& img_lc, cv::Mat& img_rc);
  static void drawFeatures(const std::vector<core::Point>& features, const cv::Mat& image);
  
 private:
  FeatureHelper() {}
};

}

#endif
