#include "feature_refiner_mock.h"

namespace track {
namespace refiner {

void FeatureRefinerMock::config(
  const std::string& conf)
{
}

void FeatureRefinerMock::addFeatures(
  const core::ImageSet& src,
  const std::map<int, core::Point>& pts)
{
  for (auto x : pts){
    map_.insert(std::make_pair(x.first, FeatureData(x.second.x_, x.second.y_)));
  }
}

void FeatureRefinerMock::refineFeatures(
  const core::ImageSet& src,
  const std::map<int, core::Point>& pts)
{
  // do nothing
}

}
}
