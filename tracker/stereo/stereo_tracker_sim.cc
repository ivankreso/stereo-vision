#include "stereo_tracker_sim.h"

namespace track {

StereoTrackerSim::StereoTrackerSim(std::string src_folder, std::string filelist_name) :
   src_folder_(src_folder), frame_cnt_(0)
{
   bool ok = readStringList(filelist_name, filelist_);
   if(!ok || filelist_.empty())
   {
      std::cout << "can not open " << filelist_name << " or the string list is empty\n";
      exit(-1);
   }
}

void StereoTrackerSim::init(core::Image& img_left, core::Image& img_right)
{
}

void StereoTrackerSim::track(core::Image& img_left, core::Image& img_right)
{
   // if first frame - simulate init func
   if(frame_cnt_ == 0) {
      frame_cnt_++;
      return;
   }

   // fill feature data
   // clear previous data
   feats_left_.clear();
   feats_right_.clear();
   pts3d_.clear();

   // read point proj data from next file   
   std::ifstream infile(src_folder_ + filelist_[frame_cnt_-1]);
   cv::Mat pt3d(3, 1, CV_64F);
   std::string val;
   while(!infile.eof()) {
      for(int i = 0; i < 3; i++) {
         infile >> val;
         if(infile.eof())
            goto file_end;
         //std::cout << val << std::endl;
         pt3d.at<double>(i,0) = std::stod(val);
      }
      pts3d_.push_back(pt3d.clone());

      infile >> val;
      //std::cout << val << std::endl;
      assert(std::stoi(val) == 2);
      infile >> val;
      assert(stoi(val) == 0);
      FeatureInfo fleft;
      FeatureInfo fright;
      infile >> val;
      fleft.prev_.x_ = stod(val);
      infile >> val;
      fleft.prev_.y_ = stod(val);
      infile >> val;
      fright.prev_.x_ = stod(val);
      infile >> val;
      fright.prev_.y_ = stod(val);
      fright.status_ = 1;
      fleft.status_ = 1;
      fleft.age_ = 1;
      fright.age_ = 1;
      
      infile >> val;
      assert(stoi(val) == 1);

      infile >> val;
      fleft.curr_.x_ = stod(val);
      infile >> val;
      fleft.curr_.y_ = stod(val);
      infile >> val;
      fright.curr_.x_ = stod(val);
      infile >> val;
      fright.curr_.y_ = stod(val);

      feats_left_.push_back(fleft);
      feats_right_.push_back(fright);
   }

file_end:
   frame_cnt_++;
}

int StereoTrackerSim::countFeatures() const
{
   assert(feats_left_.size() == feats_right_.size());
   return feats_left_.size();
}

track::FeatureInfo StereoTrackerSim::featureLeft(int i) const
{
   return feats_left_.at(i);
}

track::FeatureInfo StereoTrackerSim::featureRight(int i) const
{
   return feats_right_.at(i);
}


bool StereoTrackerSim::readStringList(const std::string& filename, std::vector<std::string>& strlist)
{
   strlist.resize(0);
   cv::FileStorage fs(filename, cv::FileStorage::READ);
   if( !fs.isOpened() )
      return false;
   cv::FileNode n = fs.getFirstTopLevelNode();
   if( n.type() != cv::FileNode::SEQ )
      return false;
   cv::FileNodeIterator it = n.begin(), it_end = n.end();
   for( ; it != it_end; ++it )
      strlist.push_back((std::string)*it);
   return true;

}

}

