#include "feature_refiner_base.h"

#include <sstream>
#include <stdexcept>

namespace track {
namespace refiner {

FeatureRefinerBase::FeatureRefinerBase(){}
FeatureRefinerBase::~FeatureRefinerBase(){}

FeatureRefinerBase::Map::iterator
FeatureRefinerBase::find(int id)
{
  auto it=map_.find(id);
  if (it==map_.end()){
    std::ostringstream oss;
    oss <<"track::FeatureRefinerBase::find ";
    oss <<"- unknown id (" <<id <<")";
    throw std::runtime_error(oss.str());
  }
  return it;
}

void FeatureRefinerBase::removeFeature(int id)
{
  auto it=find(id);
  map_.erase(it);
}

bool FeatureRefinerBase::featureExists(int id)
{
  auto it = map_.find(id);
  if(it == map_.end())
     return false;
  return true;
}

const FeatureData& FeatureRefinerBase::getFeature(int id)
{
  auto it=find(id);
  return it->second;
}

std::vector<core::Point> FeatureData::bbox() const
{
  const int hw=width()/2;
  const int hh=height()/2;
  std::vector<core::Point> bounds;
  bounds.push_back(core::Point(warpx(-hw,-hh), warpy(-hw,-hh)));
  bounds.push_back(core::Point(warpx(-hw, hh), warpy(-hw, hh)));
  bounds.push_back(core::Point(warpx( hw,-hh), warpy( hw,-hh)));
  bounds.push_back(core::Point(warpx( hw, hh), warpy( hw, hh)));
  return bounds;
}

}
}
