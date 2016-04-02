#include "stereo_tracker_bfm.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


//#define TRACK_LEFT      90  // 90 for kitti, 30 - 60 for bb2
//#define TRACK_RIGHT     90
//#define TRACK_UP        90
//#define TRACK_DOWN      90
#define EPIMATCH_LEFT   1400
#define EPIMATCH_RIGHT  0
#define EPIMATCH_UP     1
#define EPIMATCH_DOWN   1


//#define CORRELATION_THR    0.9

namespace
{

void smoothImage(core::Image& img, core::Image& img_smooth)
{
  cv::Mat cvimg_tmp, cvimg_smooth;
  track::HelperOpencv::ImageToMat(img, cvimg_tmp);
  //cv::imshow("img_nosmooth", cvimg_tmp);
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

namespace track {

using namespace std;
using namespace cv;

StereoTrackerBFM::StereoTrackerBFM(FeatureDetectorBase* detector, int max_features, double min_crosscorr,
                                   int patch_size, int window_size) :
                                   detector_(detector), max_feats_(max_features),
                                   min_crosscorr_(min_crosscorr)
{
  cbw_ = cbh_ = patch_size;
  matches_lp_.resize(max_feats_);
  matches_rp_.resize(max_feats_);
  matches_lc_.resize(max_feats_);
  matches_rc_.resize(max_feats_);
  patches_lp_.resize(max_feats_);
  patches_rp_.resize(max_feats_);
  patches_lc_.resize(max_feats_);
  patches_rc_.resize(max_feats_);
  status_.resize(max_feats_);
  age_.assign(max_feats_, -1);
  age_acc_ = 0;
  death_count_ = 0;

  wsize_left_ = window_size / 2;
  wsize_right_ = window_size / 2;
  wsize_up_ = window_size / 2;
  wsize_down_ = window_size / 2;
  use_smoothing_ = true;
}

StereoTrackerBFM::~StereoTrackerBFM()
{
}

void StereoTrackerBFM::init(const cv::Mat& img_left, const cv::Mat& img_right)
{
  if(use_smoothing_) {
    cv::GaussianBlur(img_left, cvimg_lc_, cv::Size(3,3), 0.7);
    cv::GaussianBlur(img_right, cvimg_rc_, cv::Size(3,3), 0.7);
  }
  else {
    cvimg_lc_ = img_left.clone();
    cvimg_rc_ = img_right.clone();
  }

  // detect new features
  std::vector<core::Point> feats_left, feats_right;
  detector_->detect(cvimg_lc_, feats_left);
  detector_->detect(cvimg_rc_, feats_right);

  // lay out patches to vectors
  vector<FeaturePatch> patches_left;
  vector<FeaturePatch> patches_right;
  copyPatches(cvimg_lc_, feats_left, patches_left);
  copyPatches(cvimg_rc_, feats_right, patches_right);

  // match left-right stereo features
  vector<int> match_index;
  //vector<core::Point> mbest_left, mbest_right;
  cout << "matching init left-right:\n";
  matchFeatures(cvimg_lc_, cvimg_rc_, feats_left, feats_right, patches_left, patches_right, match_index,
               EPIMATCH_LEFT, EPIMATCH_RIGHT, EPIMATCH_UP, EPIMATCH_DOWN, false, vector<int>(), false);
  // initialize matches
  initMatches(feats_left, feats_right, patches_left, patches_right, match_index,
             matches_lc_, matches_rc_, patches_lc_, patches_rc_);
}

void StereoTrackerBFM::initMatches(const std::vector<core::Point>& feats1,
                                   const std::vector<core::Point>& feats2,
                                   const std::vector<FeaturePatch>& in_patches1,
                                   const std::vector<FeaturePatch>& in_patches2,
                                   const std::vector<int>& match_index,
                                   std::vector<core::Point>& matches1,
                                   std::vector<core::Point>& matches2,
                                   std::vector<FeaturePatch>& out_patches1,
                                   std::vector<FeaturePatch>& out_patches2)
{
   assert(feats1.size() == match_index.size());
   size_t idx = 0;
   for(size_t i = 0; i < feats1.size(); i++) {
      if(idx >= matches1.size())
         break;
      if(match_index[i] > -1) {
         //if(idx == 259) std::cout << i << "\n";
         matches1[idx] = feats1[i];
         out_patches1[idx] = in_patches1[i];
         matches2[idx] = feats2[match_index[i]];
         out_patches2[idx] = in_patches2[match_index[i]];
         //status_[i] = 1;
         age_[idx] = 0;    // 0 means that it is a newly added feature
         idx++;
      }
   }
   cout << "num of init matches: " << idx << endl;
   for(; idx < matches1.size(); idx++)
      age_[idx] = -1;      // -1 means the feature is dead
}

void StereoTrackerBFM::track(const cv::Mat& img_left, const cv::Mat& img_right)
{
  cv::swap(cvimg_lp_, cvimg_lc_);
  cv::swap(cvimg_rp_, cvimg_rc_);
  if(use_smoothing_) {
    cv::GaussianBlur(img_left, cvimg_lc_, cv::Size(3,3), 0.7);
    cv::GaussianBlur(img_right, cvimg_rc_, cv::Size(3,3), 0.7);
  }
  else {
    cvimg_lc_ = img_left.clone();
    cvimg_rc_ = img_right.clone();
  }
   // status = 0 if feature is dead
   // age = -1 if feature is dead
   // age = 0 if feature is added just now (only curr matters)

   matches_lp_ = matches_lc_;
   matches_rp_ = matches_rc_;
   patches_lp_ = patches_lc_;
   patches_rp_ = patches_rc_;

   std::vector<core::Point> feats_left, feats_right;
   detector_->detect(cvimg_lc_, feats_left);
   detector_->detect(cvimg_rc_, feats_right);
   vector<FeaturePatch> patches_left;
   vector<FeaturePatch> patches_right;
   copyPatches(cvimg_lc_, feats_left, patches_left);
   copyPatches(cvimg_rc_, feats_right, patches_right);
   vector<int> match_index_epi, match_index_left, match_index_right;
   std::vector<bool> unused_features;

   // yes yes, firt temporal because of lost feats otherwise
   // TODO: maybe first match temporal and then spatial
   // match spatial
   //cout << "matching current left-right\n";
   matchFeatures(cvimg_lc_, cvimg_rc_, feats_left, feats_right, patches_left, patches_right, match_index_epi,
                 EPIMATCH_LEFT, EPIMATCH_RIGHT, EPIMATCH_UP, EPIMATCH_DOWN, false, vector<int>(), false);
   // filter unmatched features so that they are not used again in temporal matching unnecesseary - wrong, this is biased
   // we wont filter anything so that the temporal reference patch would never lose his real match in current set
   //filterUnmatched(feats_left_, feats_right_, patches_left, patches_right, match_index);

   // debug begin
   //Mat disp_matches_lc, disp_matches_rc, cvimg_left, cvimg_right;
   //HelperOpencv::ImageToMat(img_lc_, cvimg_left);
   //HelperOpencv::ImageToMat(img_rc_, cvimg_right);
   //cvtColor(cvimg_left, disp_matches_lc, COLOR_GRAY2RGB);
   //cvtColor(cvimg_right, disp_matches_rc, COLOR_GRAY2RGB);
   //DebugHelper::drawFeatures(feats_left_, Scalar(255,0,0), disp_matches_lc);
   //DebugHelper::drawFeatures(feats_right_, Scalar(255,0,0), disp_matches_rc);
   //imshow("left_curr_matches", disp_matches_lc);
   //imshow("right_curr_matches", disp_matches_rc);
   // debug end

   // match temporal
   //cout << "matching left prev-curr\n";
   matchFeatures(cvimg_lp_, cvimg_lc_, matches_lp_, feats_left, patches_lp_, patches_left, match_index_left,
                 wsize_left_, wsize_right_, wsize_up_, wsize_down_, true, age_, false);
   //cout << "matching right prev-curr\n";
   matchFeatures(cvimg_rp_, cvimg_rc_, matches_rp_, feats_right, patches_rp_, patches_right, match_index_right,
                 wsize_left_, wsize_right_, wsize_up_, wsize_down_, true, age_, false);
   updateMatches(feats_left, feats_right, patches_left, patches_right, match_index_left, match_index_right,
                 match_index_epi, unused_features);

   replaceDeadFeatures(feats_left, feats_right, patches_left, patches_right, match_index_epi, unused_features);
}

void StereoTrackerBFM::replaceDeadFeatures(const std::vector<core::Point>& feats_left,
                                           const std::vector<core::Point>& feats_right,
                                           const std::vector<FeaturePatch>& patches_left,
                                           const std::vector<FeaturePatch>& patches_right,
                                           const std::vector<int>& match_index_epi,
                                           std::vector<bool>& unused_features)
{
   size_t j = 0;
   for(size_t i = 0; i < age_.size(); i++) {
      if(age_[i] < 0) {
         // first find an unused feature in matches set
         while(j < match_index_epi.size()) {
            if(match_index_epi[j] >= 0 && unused_features[j] == true) {
               unused_features[j] = false;
               break;
            }
            j++;
         }
         // if no more unused feats
         if(j == match_index_epi.size())
            break;
         // replace the dead one with unused new feature
         matches_lc_[i] = feats_left[j];
         patches_lc_[i] = patches_left[j];
         matches_rc_[i] = feats_right[match_index_epi[j]];
         patches_rc_[i] = patches_right[match_index_epi[j]];
         age_[i] = 0;
      }
   }
}


void StereoTrackerBFM::updateMatches(const std::vector<core::Point>& feats_left,
                                     const std::vector<core::Point>& feats_right,
                                     const std::vector<FeaturePatch>& patches_left,
                                     const std::vector<FeaturePatch>& patches_right,
                                     const std::vector<int>& match_index_left,
                                     const std::vector<int>& match_index_right,
                                     const std::vector<int>& match_index_epi,
                                     std::vector<bool>& unused_features)
{
   assert(match_index_left.size() == match_index_right.size());
   unused_features.assign(feats_left.size(), true);
   int mil, mir, mie;
   int num_match = 0;
   for(size_t i = 0; i < match_index_left.size(); i++) {
      // if it is already dead from before, skip it so it can be replaced
      if(age_[i] < 0)
         continue;
      mil = match_index_left[i];
      mir = match_index_right[i];
      // check for cycle match
      // if left and right tracks exist
      if(mil >= 0 && mir >= 0) {
         mie = match_index_epi[mil];
         // if the temporal tracks match with epipolar match - we have a match! :)
         if(mie == mir) {
            matches_lc_[i] = feats_left[mil];
            matches_rc_[i] = feats_right[mir];
            patches_lc_[i] = patches_left[mil];
            patches_rc_[i] = patches_right[mir];
            age_[i]++;
            unused_features[mil] = false;
            num_match++;
            continue;
         }
      }
      // else we have a dead feature :(
      death_count_++;
      age_acc_ += age_[i];
      age_[i] = -1;
   }
   cout << "[StereoTrackerBFM] Number of final cyclic matches: " << num_match << "\n";
}

void StereoTrackerBFM::printStats()
{
  std::cout << "[StereoTrackerBFM] Active tracks: " << countActiveTracks() << "\n";
  std::cout << "[StereoTrackerBFM] Average track age: " << (double) age_acc_ / death_count_ << "\n";
}

void StereoTrackerBFM::matchFeatures(const cv::Mat& cvimg_1, const cv::Mat& cvimg_2,
                                     const std::vector<core::Point>& feats1,
                                     const std::vector<core::Point>& feats2,
                                     const std::vector<FeaturePatch>& patches1,
                                     const std::vector<FeaturePatch>& patches2,
                                     std::vector<int>& match_index, double dxl,
                                     double dxr, double dyu, double dyd, bool is_temporal,
                                     const std::vector<int>& ages, bool debug)
{
   // debug:
   Mat disp_1_track, disp_2_track;
   Mat disp_left_patch = Mat::zeros(cbh_, cbw_, CV_8U);
   Mat disp_right_patch = Mat::zeros(cbh_, cbw_, CV_8U);

   match_index.assign(feats1.size(), -1);
   vector<int> matches_1to2, matches_2to1;
   matches_1to2.resize(feats1.size());
   matches_2to1.resize(feats2.size());
   vector<double> crosscorrs;
   crosscorrs.resize(feats1.size());
   double corr, dx, dy;
   // match 1 to 2
   for(size_t i = 0; i < feats1.size(); i++) {
      // dont track if the temporal reference feature is dead
      if(is_temporal && ages[i] < 0) {
         matches_1to2[i] = -1;
         continue;
      }

      //if(i == 151) {
      //   try {throw 0;} catch(int) {}
      //}
      int ind_best = -1;
      double corr_best = 0.0;
      for(size_t j = 0; j < feats2.size(); j++) {
         dy = feats1[i].y_ - feats2[j].y_;
         dx = feats1[i].x_ - feats2[j].x_;
         // ignore features outside
         if(dy < 0.0 && dy < -dyd) continue;
         if(dy > 0.0 && dy > dyu) continue;
         if(dx < 0.0 && dx < -dxr) continue;
         if(dx > 0.0 && dx > dxl) continue;

         //cout << "match 1-2: " << i << " - " << j << endl;
         corr = getCorrelation(patches1[i], patches2[j]);

         // debug: draw on images
         //if(debug) {
         if(debug && i == 471) {
         //if(debug && i == 259) {
         //if(debug && j == 485) {
            cv::Point2f pt;
            cvtColor(cvimg_1, disp_1_track, COLOR_GRAY2RGB);
            pt.x = feats1[i].x_;
            pt.y = feats1[i].y_;
            cv::circle(disp_1_track, pt, 2, Scalar(255,0,0), 2, 8);
            Point pt_rec1, pt_rec2;
            Rect rect;
            rect.x = pt.x - dxl;
            rect.y = pt.y - dyd;
            rect.width = dxl + dxr + 1;
            rect.height = dyd + dyu + 1;
            //pt_rec1.x_ = pt.x - dxl;
            //pt_rec1.y_ = pt.y - dyd;
            //pt_rec2.x_ = pt.x + dxr;
            //pt_rec2.y_ = pt.y + dyd;
            rectangle(disp_1_track, rect, Scalar(255,0,0), 1, 8);
            //cout << "size: " << disp_left_track.size();
            imshow("image_1", disp_1_track);

            cv::Point2f pt1, pt2, pt_best;
            cout << i << " - " << j << ":\n";
            cout << feats1[i] << " - " << feats2[j] << "\n";
            cout << "correlation: " << corr << "  (the best: " << corr_best << ")\n";
            cvtColor(cvimg_2, disp_2_track, COLOR_GRAY2RGB);
            pt2.x = feats2[j].x_;
            pt2.y = feats2[j].y_;
            if(ind_best >= 0) {
               pt_best.x = feats2[ind_best].x_;
               pt_best.y = feats2[ind_best].y_;
               cv::circle(disp_2_track, pt_best, 2, Scalar(0,255,0), 2, 8);
            }
            //cout << pt1 << " <--> " << pt2 << "\n--------------------------------\n";
            cv::circle(disp_2_track, pt2, 2, Scalar(255,0,0), 2, 8);
            cv::rectangle(disp_2_track, rect, Scalar(255,0,0), 1, 8);
            cv::imshow("image_2", disp_2_track);
            cv::Mat disp_patch_1, disp_patch_2, disp_patch_best;
            DebugHelper::renderPatch(patches1[i], disp_patch_1);
            DebugHelper::renderPatch(patches2[j], disp_patch_2);
            if(ind_best >= 0)
               DebugHelper::renderPatch(patches2[ind_best], disp_patch_best);
            //cout << patches1[i].mat_ << endl;
            //cout << patches2[j].mat_ << endl;
            imshow("patch_1", disp_patch_1);
            imshow("patch_2", disp_patch_2);
            if(ind_best >= 0)
               imshow("patch_2_best", disp_patch_best);
            waitKey(0);
         }
         // end debug
         if(corr > corr_best) {
            corr_best = corr;
            ind_best = j;
         }
      }

      if(debug)
         destroyWindow("patch_2_best");

      crosscorrs[i] = corr_best;
      //cout << corr_best << endl;
      if(corr_best > min_crosscorr_)
         matches_1to2[i] = ind_best;
      else
         matches_1to2[i] = -1;
   }

   // match 2 to 1
   for(size_t i = 0; i < feats2.size(); i++) {
      int ind_best = -1;
      double corr_best = 0.0;
      for(size_t j = 0; j < feats1.size(); j++) {
         // dont track if the temporal reference feature is dead
         if(is_temporal && ages[j] < 0) {
            continue;
         }
         dy = feats1[j].y_ - feats2[i].y_;
         dx = feats1[j].x_ - feats2[i].x_;
         if(dy < 0.0 && dy < -dyd) continue;
         if(dy > 0.0 && dy > dyu) continue;
         if(dx < 0.0 && dx < -dxr) continue;
         if(dx > 0.0 && dx > dxl) continue;

         // TODO - we can optimize this and put corrs in a map during the first match
         //cout << "match 2-1: " << i << " - " << j << endl;
         corr = getCorrelation(patches1[j], patches2[i]);
         if(corr > corr_best) {
            corr_best = corr;
            ind_best = j;
         }
         //cout << corr << endl;
      }
      //cout << corr_best << endl;
      if(corr_best > min_crosscorr_) {
         matches_2to1[i] = ind_best;
      }
      else
         matches_2to1[i] = -1;
   }

   // filter only the married features
   for(int i = 0; i < feats1.size(); i++) {
      int m_1to2 = matches_1to2[i];
      // if two features were matced to each other then accept the match
      if(m_1to2 >= 0) {
         if(matches_2to1[m_1to2] == i)
            match_index[i] = m_1to2;
      }
   }
}

std::vector<size_t> StereoTrackerBFM::getSortedIndices(std::vector<double> const& values) {
   std::vector<size_t> indices(values.size());
   std::iota(begin(indices), end(indices), static_cast<size_t>(0));
   std::sort(begin(indices), end(indices), [&](size_t a, size_t b) { return values[a] > values[b]; } );
   return indices;
}

double StereoTrackerBFM::getCorrelation(const FeaturePatch& p1, const FeaturePatch& p2)
{
   //cout << p1.mat_.size() << " -- " << p2.mat_.size() << endl;
   assert(p1.mat_.rows == p2.mat_.rows);
   double n = p1.mat_.rows;
   double D = p1.mat_.dot(p2.mat_);
   return (n * D - (p1.A_ * p2.A_)) * p1.C_ * p2.C_;
}

void StereoTrackerBFM::copyPatches(const cv::Mat& img, std::vector<core::Point>& features,
                                   std::vector<FeaturePatch>& patches)
{
   int cx, cy;
   int patchsz = cbw_*cbh_;
   int radx = (cbw_ - 1) / 2;
   int rady = (cbh_ - 1) / 2;
   for(size_t k = 0; k < features.size(); k++) {
      FeaturePatch patch;
      //patch.mat_ = Mat::zeros(patchsz, 1, CV_8U);
      patch.mat_ = Mat(patchsz, 1, CV_8U);
      cx = features[k].x_;
      cy = features[k].y_;
      //if(k == 1496)
         //cout << k << ": " << img.cols_ << "x" << img.rows_ << "\ncent: " << cx << " " << cy << "\n";
      assert(cx >= radx && cx < (img.cols-radx));
      assert(cy >= rady && cy < (img.rows-rady));
      int vpos = 0;
      for(int i = cy - rady; i <= cy + rady; i++) {
         for(int j = cx - radx; j <= cx + radx; j++) {
            //cout << k << " -> " << i << "-" << j << " = " << endl;
            //cout << (int)img(i,j) << endl;
            patch.mat_.at<uint8_t>(vpos,0) = img.at<uint8_t>(i,j);
            patch.A_ += img.at<uint8_t>(i,j);
            patch.B_ += (img.at<uint8_t>(i,j) * img.at<uint8_t>(i,j));
            vpos++;
         }
      }
      patch.C_ = 1.0 / sqrt((patchsz * patch.B_) - (patch.A_ * patch.A_));
      patches.push_back(patch);
   }
}


int StereoTrackerBFM::countFeatures() const
{
   return matches_lp_.size();
}

int StereoTrackerBFM::countActiveTracks() const
{
  int cnt = 0;
  for(size_t i = 0; i < age_.size(); i++) {
    if(age_[i] > 0)
      cnt++;
  }
  return cnt;
}

FeatureInfo StereoTrackerBFM::featureLeft(int i) const
{
   FeatureInfo feat;
   feat.prev_ = matches_lp_[i];
   feat.curr_ = matches_lc_[i];
   feat.age_ = age_[i];
   feat.status_ = age_[i] + 1;
   return std::move(feat);
}

FeatureInfo StereoTrackerBFM::featureRight(int i) const
{
   FeatureInfo feat;
   feat.prev_ = matches_rp_[i];
   feat.curr_ = matches_rc_[i];
   feat.age_ = age_[i];
   feat.status_ = age_[i] + 1;
   return std::move(feat);
}

void StereoTrackerBFM::removeTrack(int id)
{
  if(age_[id] > 0) {
    age_acc_ += age_[id];
    death_count_++;
  }
  age_[id] = -1;
}

FeatureData StereoTrackerBFM::getLeftFeatureData(int i)
{
  FeatureData fdata;
  fdata.feat_ = featureLeft(i);
  //fdata.patch_prev_ = okokoookkkkkkoooooookkkkk

  return fdata;
}

FeatureData StereoTrackerBFM::getRightFeatureData(int i)
{
  FeatureData fdata;

  return fdata;
}

void StereoTrackerBFM::filterUnmatched(std::vector<core::Point>& feats1, std::vector<core::Point>& feats2,
                                       std::vector<FeaturePatch>& patches1, std::vector<FeaturePatch>& patches2,
                                       std::vector<int>& match_index)
{
   vector<core::Point> feats1_new, feats2_new;
   vector<FeaturePatch> patches1_new, patches2_new;
   for(size_t i = 0; i < feats1.size(); i++) {
      if(match_index[i] > -1) {
         feats1_new.push_back(feats1[i]);
         feats2_new.push_back(feats2[match_index[i]]);
         patches1_new.push_back(patches1[i]);
         patches2_new.push_back(patches2[match_index[i]]);
      }
   }
   feats1 = std::move(feats1_new);
   feats2 = std::move(feats2_new);
   patches1 = std::move(patches1_new);
   patches2 = std::move(patches2_new);
}


void StereoTrackerBFM::showTrack(int i) const
{
  cv::Mat img_lp, img_lc, img_rp, img_rc;
  cv::cvtColor(cvimg_lp_, img_lp, cv::COLOR_GRAY2RGB);
  cv::cvtColor(cvimg_lc_, img_lc, cv::COLOR_GRAY2RGB);
  cv::cvtColor(cvimg_rp_, img_rp, cv::COLOR_GRAY2RGB);
  cv::cvtColor(cvimg_rp_, img_rc, cv::COLOR_GRAY2RGB);
  FeatureInfo feat_left = featureLeft(i);
  FeatureInfo feat_right = featureRight(i);
  if(feat_left.age_ <= 0) throw "Error\n";
  cv::Scalar color(0,255,0);
  DrawFeature(feat_left.prev_, color, img_lp);
  DrawFeature(feat_left.curr_, color, img_lc);
  DrawFeature(feat_right.prev_, color, img_rp);
  DrawFeature(feat_right.curr_, color, img_rc);
  
  cv::resize(img_lp, img_lp, cv::Size(), 2.0, 2.0);
  cv::resize(img_rp, img_rp, cv::Size(), 2.0, 2.0);
  cv::resize(img_lc, img_lc, cv::Size(), 2.0, 2.0);
  cv::resize(img_rc, img_rc, cv::Size(), 2.0, 2.0);
  cv::imshow("left_prev", img_lp);
  cv::imshow("left_curr", img_lc);
  cv::imshow("right_prev", img_rp);
  cv::imshow("right_curr", img_rc);
  cv::waitKey(0);
}


} // end namespace
