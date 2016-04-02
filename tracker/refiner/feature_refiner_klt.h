#ifndef __TRACKER_FEATURE_REFINER_KLT_H__
#define __TRACKER_FEATURE_REFINER_KLT_H__

#include "feature_refiner_base.h"

namespace track {
namespace refiner {

class FeatureRefinerKLT:
  public FeatureRefinerBase
{
  const int    warpModel_ = 2;          // 2, 5, 8: the best - 2
  const double thMaxResidue_ = 10;      // orig - 10, tried: 5 - 20
  const int    thMaxIterations_ = 15;   // try to increase: default 15, try: 30 - 100
  const double optimizationFactor_ = 0.5; // try to decrese, default: 0.5
  const double thDisplacementConvergence_ = 0.01;   // try to decrease, default: 0.01
  const double thDisplacementDivergence_ = 5;
  const bool   verbose_=false;

std::vector<core::Image> gwimg_;
public:
  virtual void config(const std::string& conf);

  virtual void addFeatures(
    const core::ImageSet& src,
    const std::map<int, core::Point>& pts);

  virtual void refineFeatures(
    const core::ImageSet& src,
    const std::map<int, core::Point>& pts);
};

}
}

#endif

