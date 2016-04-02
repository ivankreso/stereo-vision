#include "helper_libviso.h"

namespace vo {

using namespace std;
using namespace libviso;

void HelperLibviso::convertAllMatchesToKeys(vector<Matcher::p_match>& matches, vector<vector<cv::KeyPoint>>& keypoints)
{
  keypoints.resize(4);
  cv::KeyPoint kp;
  for(size_t i = 0; i < matches.size(); i++) {
    //cout << "match " << i << endl;
    kp.pt.x = matches[i].u1p;
    kp.pt.y = matches[i].v1p;
    keypoints[0].push_back(kp);
    kp.pt.x = matches[i].u2p;
    kp.pt.y = matches[i].v2p;
    keypoints[1].push_back(kp);
    kp.pt.x = matches[i].u1c;
    kp.pt.y = matches[i].v1c;
    keypoints[2].push_back(kp);
    kp.pt.x = matches[i].u2c;
    kp.pt.y = matches[i].v2c;
    keypoints[3].push_back(kp);
  }
}

void HelperLibviso::convertInlierMatchesToKeys(vector<Matcher::p_match>& matches, vector<int32_t>& inliers, 
    vector<vector<cv::KeyPoint>>& keypoints)
{
  keypoints.resize(4);
  cv::KeyPoint kp;
  for(size_t i = 0; i < inliers.size(); i++) {
    //cout << "match " << i << endl;
    int32_t idx = inliers[i];
    kp.pt.x = matches[idx].u1p;
    kp.pt.y = matches[idx].v1p;
    keypoints[0].push_back(kp);
    kp.pt.x = matches[idx].u2p;
    kp.pt.y = matches[idx].v2p;
    keypoints[1].push_back(kp);
    kp.pt.x = matches[idx].u1c;
    kp.pt.y = matches[idx].v1c;
    keypoints[2].push_back(kp);
    kp.pt.x = matches[idx].u2c;
    kp.pt.y = matches[idx].v2c;
    keypoints[3].push_back(kp);
  }
}

void HelperLibviso::LibvisoInliersToPoints(std::vector<libviso::Matcher::p_match>& matches,
    std::vector<int32_t>& inliers,
    std::vector<core::Point>& pts_lp, std::vector<core::Point>& pts_rp,
    std::vector<core::Point>& pts_lc, std::vector<core::Point>& pts_rc)
{
  pts_lp.clear();
  pts_rp.clear();
  pts_lc.clear();
  pts_rc.clear();

  core::Point pt;
  for(size_t i = 0; i < inliers.size(); i++) {
    //cout << "match " << i << endl;
    int32_t idx = inliers[i];
    pt.x_ = matches[idx].u1p;
    pt.y_ = matches[idx].v1p;
    pts_lp.push_back(pt);

    pt.x_ = matches[idx].u2p;
    pt.y_ = matches[idx].v2p;
    pts_rp.push_back(pt);

    pt.x_ = matches[idx].u1c;
    pt.y_ = matches[idx].v1c;
    pts_lc.push_back(pt);

    pt.x_ = matches[idx].u2c;
    pt.y_ = matches[idx].v2c;
    pts_rc.push_back(pt);
  }
}
cv::Mat HelperLibviso::getCameraMatrix(VisualOdometryStereo::parameters& param)
{
  cv::Mat C = cv::Mat::zeros(3, 4, CV_64F);
  C.at<double>(0,0) = param.calib.f;
  C.at<double>(1,1) = param.calib.f;
  C.at<double>(2,2) = 1.0;
  C.at<double>(0,2) = param.calib.cu;
  C.at<double>(1,2) = param.calib.cv;
  return C;
}

// function draws optical flow ie. coresponding features
void HelperLibviso::drawOpticalFlow(cv::Mat& img, const std::vector<cv::KeyPoint>& points_prev, 
    const std::vector<cv::KeyPoint>& points_next, const std::vector<uchar>& track_status, const cv::Scalar& color)
{
  cv::Point pt1, pt2;
  assert(points_prev.size() == points_next.size());
  for(size_t i = 0; i < points_prev.size(); i++) {
    if(track_status[i] == 1) {
      pt1.x = points_prev[i].pt.x;
      pt1.y = points_prev[i].pt.y;
      pt2.x = points_next[i].pt.x;
      pt2.y = points_next[i].pt.y;
      cv::line(img, pt1, pt2, color, 2, 8);
      cv::circle(img, pt1, 2, cv::Scalar(255, 0, 0), -1, 8);
    }
  }
}
}

