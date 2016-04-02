#include "stereo_tracker_artificial.h"

namespace track {

StereoTrackerArtificial::StereoTrackerArtificial(std::string src_folder, int max_feats) :
                                                 src_folder_(src_folder), frame_cnt_(0), max_feats_(max_feats)
{
  matches_lp_.resize(max_feats);
  matches_rp_.resize(max_feats);
  matches_lc_.resize(max_feats);
  matches_rc_.resize(max_feats);
  age_.assign(max_feats, -1);
}

void StereoTrackerArtificial::init(core::Image& img_left, core::Image& img_right)
{
  readNextFrame();
}

void StereoTrackerArtificial::track(core::Image& img_left, core::Image& img_right)
{
  matches_lp_ = matches_lc_;
  matches_rp_ = matches_rc_;
  readNextFrame();
}

void StereoTrackerArtificial::readNextFrame()
{
  // read point proj data from next file
  std::stringstream filename;
  filename << std::setw(6) << std::setfill('0') << frame_cnt_ << ".txt";
  std::string path = src_folder_ + "/" + filename.str();
  std::cout << "Reading frame: " << path << "\n";
  std::ifstream infile(path);
  std::string val;
  int i = 0;
  int feat_num;
  while(!infile.eof()) {
    //for(int i = 0; i < 3; i++) {
    //  infile >> val;
    //  if(infile.eof())
    //    goto file_end;
    //  //std::cout << val << std::endl;
    //  pt3d.at<double>(i,0) = std::stod(val);
    //}
    infile >> val;
    if(infile.eof())
      break;
    feat_num = std::stod(val);
    //std::cout << val << std::endl;
    //assert(std::stoi(val) == 2);
    infile >> val;
    //assert(stoi(val) == 0);
    age_[i] = std::stoi(val);
    infile >> val;
    matches_lc_[i].x_ = std::stod(val);
    infile >> val;
    matches_lc_[i].y_ = std::stod(val);
    infile >> val;
    matches_rc_[i].x_ = std::stod(val);
    infile >> val;
    matches_rc_[i].y_ = std::stod(val);
    i++;
  }
  frame_cnt_++;

  if(i != max_feats_)
    throw "[StereoTrackerArtificial]: wrong max_feats size!\n";
  else if(i != (feat_num+1))
    throw "[StereoTrackerArtificial]: error in input file!\n";
}

int StereoTrackerArtificial::countFeatures() const
{
  return max_feats_;
}

FeatureInfo StereoTrackerArtificial::featureLeft(int i) const
{
  FeatureInfo feat;
  feat.prev_ = matches_lp_[i];
  feat.curr_ = matches_lc_[i];
  feat.age_ = age_[i];
  feat.status_ = age_[i] + 1;
  return std::move(feat);
}

FeatureInfo StereoTrackerArtificial::featureRight(int i) const
{
  FeatureInfo feat;
  feat.prev_ = matches_rp_[i];
  feat.curr_ = matches_rc_[i];
  feat.age_ = age_[i];
  feat.status_ = age_[i] + 1;
  return std::move(feat);
}

}

