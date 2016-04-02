#include "stereo_tracker_refiner.h"

#include "stereo_tracker_bfm.h"
#include "../../tracker/stereo/stereo_tracker_libviso.h"
#include "../../core/math_helper.h"



namespace track {

using namespace track::refiner;

StereoTrackerRefiner::StereoTrackerRefiner(
    StereoTrackerBase* tracker,
    FeatureRefinerBase* refiner,
    bool debug_on)
  : tracker_(tracker), refiner_(refiner), debug_on_(debug_on)
{
  max_feats_ = tracker->countFeatures();
  points_lp_.resize(max_feats_);
  points_rp_.resize(max_feats_);
  points_lc_.resize(max_feats_);
  points_rc_.resize(max_feats_);
  age_.resize(max_feats_);
  if(debug_on_) {
    fdata_lp_.resize(max_feats_);
    fdata_rp_.resize(max_feats_);
    fdata_lc_.resize(max_feats_);
    fdata_rc_.resize(max_feats_);
  }
}

StereoTrackerRefiner::~StereoTrackerRefiner()
{
  delete tracker_;
  delete refiner_;
}

void StereoTrackerRefiner::init(const cv::Mat& img_left, const cv::Mat& img_right)
{
  core::Image left, right;
  HelperOpencv::MatToImage(img_left, left);
  HelperOpencv::MatToImage(img_right, right);
  
  tracker_->init(img_left, img_right);
  init(left, right);
  img_lp_ = img_left.clone();
  img_rp_ = img_right.clone();
}

void StereoTrackerRefiner::track(const cv::Mat& img_left, const cv::Mat& img_right)
{
  core::Image left, right;
  HelperOpencv::MatToImage(img_left, left);
  HelperOpencv::MatToImage(img_right, right);

  tracker_->track(img_left, img_right);
  track(left, right);
  img_lp_ = img_left.clone();
  img_rp_ = img_right.clone();
}

void StereoTrackerRefiner::init(core::Image& img_left, core::Image& img_right)
{
  imgset_left_prev_.compute(img_left);
  imgset_right_prev_.compute(img_right);
}

void StereoTrackerRefiner::track(core::Image& img_left, core::Image& img_right)
{
  // copy current to prev
  points_lp_ = points_lc_;
  points_rp_ = points_rc_;
  fdata_lp_ = fdata_lc_;
  fdata_rp_ = fdata_rc_;

  std::map<int, core::Point> new_references;
  std::map<int, core::Point> refined_left, refined_right, refined_new_rp;

  assert(tracker_->countFeatures() <= max_feats_);
  int i;
  for(i = 0; i < tracker_->countFeatures(); i++)
  {
    FeatureInfo f_left = tracker_->featureLeft(i);
    FeatureInfo f_right = tracker_->featureRight(i);
    int age = f_left.age_;
    age_[i] = age;

    // if dead or newly added - remove it from refiner
    if(age < 2) {
      if(refiner_->featureExists(i))
        refiner_->removeFeature(i);
    }
    if(age >= 1) {
      // if it is a first track then add the feature reference point for refinement
      if(age == 1) {
        points_lp_[i] = f_left.prev_;

        if(debug_on_)
          fdata_lp_[i].setpos(f_left.prev_.x_, f_left.prev_.y_);

        new_references.insert(std::pair<int,core::Point>(i, f_left.prev_));
        // add it for refinement in right image also
        refined_new_rp.insert(std::pair<int,core::Point>(i, f_right.prev_));
        //std::cout << f_left.prev_ << "\n";
        //std::cout << f_right.prev_ << "\n";
        // dont need this
        //points_lp_[i] = f_left.prev_;
        //points_rp_[i] = f_right.prev_;
      }

      // add feature track points for refinement
      refined_left.insert(std::pair<int,core::Point>(i, f_left.curr_));
      refined_right.insert(std::pair<int,core::Point>(i, f_right.curr_));
    }
  }

  // add new track reference points to refiner using prev imgset
  refiner_->addFeatures(imgset_left_prev_, new_references);

  // first refine new tracks (age == 1) in previous right image
  refiner_->refineFeatures(imgset_right_prev_, refined_new_rp);
  // update points positions
  int converged_cnt = 0;
  for(auto pt : refined_new_rp) {
    refiner::FeatureData fdata = refiner_->getFeature(pt.first);
    core::Point rpt = fdata.pt();
    auto status = fdata.status_;
    if(status == refiner::FeatureData::OK) {
      points_rp_[pt.first] = rpt;
      converged_cnt++;
      //fdata_rp[pt.first] = std::make_tuple(fdata.first_residue_, fdata.residue_);
      if(debug_on_)
        fdata_rp_[pt.first] = fdata;
    }
    else {
      tracker_->removeTrack(pt.first);
      age_[pt.first] = -1;
    }
  }
  std::cout << "[StereoTrackerRefiner]: Converged new tracks in prev right: " << converged_cnt << " / "
            << refined_new_rp.size() << "\n";
  
  converged_cnt = 0;
  // compute new images
  imgset_left_.compute(img_left);
  // refine the points in current frames
  refiner_->refineFeatures(imgset_left_, refined_left);
  // update current points positions
  // TODO - check refiner::FeatureData
  // refiner_->getFeature(id);
  for(auto pt : refined_left) {
    refiner::FeatureData fdata = refiner_->getFeature(pt.first);
    core::Point rpt = fdata.pt();
    auto status = fdata.status_;
    if(status == refiner::FeatureData::OK) {
      points_lc_[pt.first] = rpt;
      //fdata_lc_[pt.first] = std::make_tuple(fdata.first_residue_, fdata.residue_);
      if(debug_on_)
        fdata_lc_[pt.first] = fdata;
      converged_cnt++;
    }
    else {
      tracker_->removeTrack(pt.first);
      age_[pt.first] = -1;
      //std::cout << status << ": " << pt.second << " --> " << rpt << "\n";
      //points_lc_[pt.first] = pt.second;
    }
  }
  std::cout << "[StereoTrackerRefiner]: Converged in next left: " << converged_cnt << " / "
            << refined_left.size() << "\n";

  // refine in next right image
  converged_cnt = 0;
  imgset_right_.compute(img_right);
  refiner_->refineFeatures(imgset_right_, refined_right);
  for(auto pt : refined_right) {
    refiner::FeatureData fdata = refiner_->getFeature(pt.first);
    core::Point rpt = fdata.pt();
    auto status = fdata.status_;
    if(status == refiner::FeatureData::OK) {
      points_rc_[pt.first] = rpt;
      //fdata_rc_[pt.first] = std::make_tuple(fdata.first_residue_, fdata.residue_);
      if(debug_on_)
        fdata_rc_[pt.first] = fdata;
      converged_cnt++;
    }
    else {
      tracker_->removeTrack(pt.first);
      age_[pt.first] = -1;
      //std::cout << status << ": " << pt.second << " --> " << rpt << "\n";
      //points_rc_[pt.first] = pt.second;
    }
  }
  std::cout << "[StereoTrackerRefiner]: Converged in next right: " << converged_cnt << " / "
            << refined_right.size() << "\n";

  //filterBadTracks();

  imgset_left_prev_ = imgset_left_;
  imgset_right_prev_ = imgset_right_;
  // TODO: replace this by = imgset_right;
  //imgset_left_prev_.compute(img_left);
  //imgset_right_prev_.compute(img_right);
}

int StereoTrackerRefiner::countFeatures() const
{
  return tracker_->countFeatures();
}

int StereoTrackerRefiner::countActiveTracks() const
{
  return tracker_->countActiveTracks();
}

FeatureInfo StereoTrackerRefiner::featureLeft(int i) const
{
  FeatureInfo feat;
  feat.prev_ = points_lp_[i];
  feat.curr_ = points_lc_[i];
  feat.age_ = age_[i];
  feat.status_ = 0;
  return feat;
}

FeatureInfo StereoTrackerRefiner::featureRight(int i) const
{
  FeatureInfo feat;
  feat.prev_ = points_rp_[i];
  feat.curr_ = points_rc_[i];
  feat.age_ = age_[i];
  feat.status_ = 0;
  return feat;
}

void StereoTrackerRefiner::removeTrack(int id)
{
  tracker_->removeTrack(id);
  age_[id] = -1;
  if(refiner_->featureExists(id))
    refiner_->removeFeature(id);
  //else std::cerr << "[StereoTrackerRefiner]: track doesn't exist\n";
}

void StereoTrackerRefiner::printStats() const
{
  ((StereoTrackerBFM*)tracker_)->printStats();
}

}
