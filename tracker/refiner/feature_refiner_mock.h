#ifndef __TRACKER_FEATURE_REFINER_MOCK_H__
#define __TRACKER_FEATURE_REFINER_MOCK_H__

#include "feature_refiner_base.h"

namespace track {
namespace refiner {

class FeatureRefinerMock:
  public FeatureRefinerBase
{
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
