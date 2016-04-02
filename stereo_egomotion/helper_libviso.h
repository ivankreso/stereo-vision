#ifndef _TRACKER_HELPER_LIBVISO_H_
#define _TRACKER_HELPER_LIBVISO_H_

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "../core/image.h"
#include "extern/libviso2/src/matcher.h"
#include "extern/libviso2/src/viso_stereo.h"

namespace vo {

class HelperLibviso
{
  public:
    static void convertAllMatchesToKeys(std::vector<libviso::Matcher::p_match>& matches,
        std::vector<std::vector<cv::KeyPoint>>& keypoints);

    static void convertInlierMatchesToKeys(std::vector<libviso::Matcher::p_match>& matches, std::vector<int32_t>& inliers,
        std::vector<std::vector<cv::KeyPoint>>& keypoints);

    static cv::Mat getCameraMatrix(libviso::VisualOdometryStereo::parameters& param);

    static void drawOpticalFlow(cv::Mat& img, const std::vector<cv::KeyPoint>& points_prev,
        const std::vector<cv::KeyPoint>& points_next, const std::vector<uchar>& track_status, const cv::Scalar& color);

    static void LibvisoInliersToPoints(std::vector<libviso::Matcher::p_match>& matches,
                                       std::vector<int32_t>& inliers,
                                       std::vector<core::Point>& pts_lp, std::vector<core::Point>& pts_rp,
                                       std::vector<core::Point>& pts_lc, std::vector<core::Point>& pts_rc);
};

}

#endif
