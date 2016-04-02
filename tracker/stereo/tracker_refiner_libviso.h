#ifndef __TRACKER_REFINER_LIBVISO__
#define __TRACKER_REFINER_LIBVISO__

#include "stereo_tracker_base.h"
#include "../refiner/feature_refiner_base.h"

namespace track {

class TrackerRefinerLibviso : public StereoTrackerBase
{
public:
   TrackerRefinerLibviso(StereoTrackerBase* tracker);
   //TrackerRefinerLibviso(StereoTrackerBase* tracker, FeatureRefinerBase* refiner);
   ~TrackerRefinerLibviso();
   virtual void init(core::Image& img_left, core::Image& img_right);
   virtual void track(core::Image& img_left, core::Image& img_right);
   virtual int countFeatures() const;
   virtual FeatureInfo featureLeft(int i) const;
   virtual FeatureInfo featureRight(int i) const;
   virtual void removeTrack(int id);
   virtual int countActiveTracks() const;

private:
   StereoTrackerBase* tracker_;
   //FeatureRefinerBase* refiner_;
   core::ImageSetExact imgset_left_;
   core::ImageSetExact imgset_right_;
   std::vector<core::Point> points_lp_, points_rp_, points_lc_, points_rc_;
   std::vector<int> age_;
};

}

#endif
