#ifndef _STEREO_TRACKER_SIM_
#define _STEREO_TRACKER_SIM_

#include <string>
#include <vector>
#include <fstream>

#include <opencv2/core/core.hpp>

#include "stereo_tracker_base.h"

namespace track {

class StereoTrackerSim : public StereoTrackerBase
{
public:
   StereoTrackerSim(std::string src_folder, std::string filelist);
   ~StereoTrackerSim() {}
   virtual void init(core::Image& img_left, core::Image& img_right);
   virtual void track(core::Image& img_left, core::Image& img_right);
   virtual int countFeatures() const;
   virtual FeatureInfo featureLeft(int i) const;
   virtual FeatureInfo featureRight(int i) const;
   virtual void removeTrack(int id) {}

protected:
   std::vector<FeatureInfo> feats_left_;
   std::vector<FeatureInfo> feats_right_;
   std::vector<cv::Mat> pts3d_;
   std::string src_folder_;
   std::vector<std::string> filelist_;
   uint32_t frame_cnt_ = 0;

   bool readStringList(const std::string& filename, std::vector<std::string>& strlist);
   void readPointProjs();
};

}

#endif
