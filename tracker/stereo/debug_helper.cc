#include "debug_helper.h"
#include "../../core/math_helper.h"
#include "../base/helper_opencv.h"
#include "../base/eval_helper.h"
#include "../../core/math_helper.h"

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace track {

namespace
{
void analyze_track(const core::Point& orig_pt, const cv::Mat& orig_desc,
                   const refiner::FeatureData& rfdata, double scale_sz,
                   double max_dist)
{
  if(rfdata.residue_ > rfdata.first_residue_)
    printf("[Warn] Residue increase detected: %f -> %f", rfdata.first_residue_, rfdata.residue_);
  const core::Point ref_pt = rfdata.pt();
  std::cout << "Orig pt:\n" << orig_pt << "\n" << "Refined pt:\n" << ref_pt << '\n';
  double dist2d = core::MathHelper::getDist2D(orig_pt, ref_pt);
  printf("pixel dist = %f\n", dist2d);
  if(dist2d < max_dist)
    return;
  cv::Point pt1, pt2;
  pt1.x = orig_pt.x_;
  pt1.y = orig_pt.y_;
  pt2.x = ref_pt.x_;
  pt2.y = ref_pt.y_;
  //cv::line(img_lp2, pt1, pt2, color_prev, 2, 8);
  //cv::circle(img_lp2, pt1, 2, color_prev, -1, 8);
  //cv::circle(img_lp2, pt2, 2, color_curr, -1, 8);
  //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
  //cv::putText(img_rp2, std::to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);
  //track::FeatureData t_fdata = tracker_->getLeftFeatureData(i);
  // draw patches
  cv::Mat r_patch_ref, r_patch_track, t_patch_track;
  track::HelperOpencv::FloatImageToMat(rfdata.ref_, r_patch_ref);
  track::HelperOpencv::FloatImageToMat(rfdata.warped_, r_patch_track);
  cv::resize(r_patch_ref, r_patch_ref, cv::Size(), scale_sz, scale_sz, cv::INTER_NEAREST);
  cv::resize(r_patch_track, r_patch_track, cv::Size(), scale_sz, scale_sz, cv::INTER_NEAREST);
  cv::resize(orig_desc, t_patch_track, cv::Size(), scale_sz, scale_sz, cv::INTER_NEAREST);
  cv::imshow("refiner_ref", r_patch_ref);
  cv::imshow("refiner_track", r_patch_track);
  cv::imshow("orig_track", t_patch_track);
  cv::waitKey(0);
}

}

void DebugHelper::DebugStereoRefiner(const cv::Mat& img_lp, const cv::Mat& img_rp, const cv::Mat& img_lc,
    const cv::Mat& img_rc, StereoTrackerRefiner& refiner, const cv::Mat& cvRt, const double* cam_params)
{
  Eigen::Matrix4d Rt;
  cv::cv2eigen(cvRt, Rt);

  cv::Mat draw_lp, draw_lc, draw_rp, draw_rc;
  double error_3d, left_error, right_error, ref_left_error, ref_right_error;
  //double reproj_error_thr = 1.0;
  std::cout << "GT motion:\n" << Rt << "\n";
  StereoTrackerBase& tracker = *refiner.tracker_;
  double scale_sz = 13.0;
  cv::Mat orig_patch_ref;
  int bad_left_refinemets = 0;
  int bad_right_refinemets = 0;
  for(int i = 0; i < refiner.countFeatures(); i++) {
    track::FeatureInfo feat_left = tracker.featureLeft(i);
    track::FeatureInfo feat_right = tracker.featureRight(i);
    if(feat_left.age_ < 1) continue;

    track::FeatureData left_data = tracker.getLeftFeatureData(i);
    track::FeatureData right_data = tracker.getRightFeatureData(i);
    EvalHelper::GetStereoReprojErrors(feat_left.prev_, feat_right.prev_, feat_left.curr_, feat_right.curr_,
                                      Rt, cam_params, error_3d, left_error, right_error);
    //if(left_error < reproj_error_thr && right_error < reproj_error_thr) continue;
    EvalHelper::GetStereoReprojErrors(refiner.fdata_lp_[i].pt(), refiner.fdata_rp_[i].pt(),
        refiner.fdata_lc_[i].pt(), refiner.fdata_rc_[i].pt(), Rt, cam_params, error_3d, 
        ref_left_error, ref_right_error);

    if(left_error > ref_left_error && right_error > ref_right_error)
      continue;
    if(left_error < ref_left_error)
      bad_left_refinemets++;
    if(right_error < ref_right_error)
      bad_right_refinemets++;

    // dont draw
    continue;

    printf("\nPoint idx = %d\n", i);
    //std::cout << "[Tracker] Curr 3D euclid error = " << error_3d << "\n";
    std::cout << "[Tracker] Curr 2D reproj error left = " << left_error << "\n";
    std::cout << "[Tracker] Curr 2D reproj error right = " << right_error << "\n";
    //std::cout << "[Refiner] Curr 3D euclid error = " << ref_error_3d << "\n";
    std::cout << "[Refiner] Curr 2D reproj error left = " << ref_left_error << "\n";
    std::cout << "[Refiner] Curr 2D reproj error right = " << ref_right_error << "\n";

    cv::cvtColor(img_lp, draw_lp, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img_lc, draw_lc, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img_rp, draw_rp, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img_rc, draw_rc, cv::COLOR_GRAY2RGB);
    cv::Scalar color(0,0,255);
    drawPoint(feat_left.prev_, color, draw_lp);
    drawPoint(feat_left.curr_, color, draw_lc);
    drawPoint(feat_right.prev_, color, draw_rp);
    drawPoint(feat_right.curr_, color, draw_rc);
    //cv::resize(img_lp, img_lp, cv::Size(), ssz, ssz);
    //cv::resize(img_rp, img_rp, cv::Size(), ssz, ssz);
    //cv::resize(img_lc, img_lc, cv::Size(), ssz, ssz);
    //cv::resize(img_rc, img_rc, cv::Size(), ssz, ssz);
    cv::imshow("left_prev", draw_lp);
    cv::imshow("left_curr", draw_lc);
    cv::imshow("right_prev", draw_rp);
    cv::imshow("right_curr", draw_rc);

    cv::resize(left_data.desc_prev_, orig_patch_ref, cv::Size(), scale_sz, scale_sz, cv::INTER_NEAREST);
    cv::imshow("orig_ref", orig_patch_ref);
    // we skip the left previous points - no need to look them
    printf("Right prev:\n");
    analyze_track(feat_right.prev_, right_data.desc_prev_, refiner.fdata_rp_[i], scale_sz, 1.0);
    printf("Left curr:\n");
    analyze_track(feat_left.curr_, left_data.desc_curr_, refiner.fdata_lc_[i], scale_sz, 1.0);
    printf("Right curr:\n");
    analyze_track(feat_right.curr_, right_data.desc_curr_, refiner.fdata_rc_[i], scale_sz, 1.0);
    //analyze_track(feat_left.curr_, points_lc_[i], fdata_lc_[i]);
    //refiner::FeatureData r_data = refiner_->getFeature(i);
    //if(std::get<1>(fdata_rp[i]) > std::get<0>(fdata_rp[i])
    //   || std::get<1>(fdata_lc_[i]) > std::get<0>(fdata_lc_[i])
    //   || std::get<1>(fdata_rc_[i]) > std::get<0>(fdata_rc_[i]))
  }
  int num_active = refiner.countActiveTracks();
  printf("[Left]: num of bad refinements = %d / %d = %.2f%%\n", bad_left_refinemets, num_active,
      ((double)bad_left_refinemets / num_active) * 100.0);
  printf("[Right]: num of bad refinements = %d / %d = %.2f%%\n", bad_right_refinemets, num_active,
      ((double)bad_right_refinemets / num_active) * 100.0);
}

void DebugHelper::renderPatch(const FeaturePatch& patch, cv::Mat& img)
{
  cv::Size imgsz = cv::Size(200, 200);
  img = patch.mat_.clone();
  img = img.reshape(0, 11);
  cv::resize(img, img, imgsz, 0, 0, cv::INTER_NEAREST);
}

void DebugHelper::drawFeatures(const std::vector<core::Point>& feats, const cv::Scalar& color, cv::Mat& img)
{
  cv::Point2f pt;
  for(size_t i = 0; i < feats.size(); i++) {
    pt.x = feats[i].x_;
    pt.y = feats[i].y_;
    cv::circle(img, pt, 2, color, -1, 8);
  }
}

void DebugHelper::drawPoint(const core::Point& pt, const cv::Scalar& color, cv::Mat& img) {
  cv::Point2f cvpt;
  cvpt.x = pt.x_;
  cvpt.y = pt.y_;
  cv::circle(img, cvpt, 2, color, -1, 8);
}



}
