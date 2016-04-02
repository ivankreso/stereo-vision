#ifndef __TRACKER_FEATURE_REFINER_BASE_H__
#define __TRACKER_FEATURE_REFINER_BASE_H__

#include <vector>
#include <map>

#include "../../core/types.h"
#include "../../core/image.h"


namespace track {
namespace refiner {

struct FeatureData;

class FeatureRefinerBase{
protected:
  typedef std::map<int,FeatureData> Map;
  Map map_;
  Map::iterator find(int id);
public:
  FeatureRefinerBase();
  virtual ~FeatureRefinerBase();
public:
  bool featureExists(int id);
  void removeFeature(int id);
  const FeatureData& getFeature(int id);
public:
  // acquires feature appearance
  virtual void addFeatures(
    const core::ImageSet& src,
    const std::map<int, core::Point>& pts) = 0;
  // improve positions of features with known appearance
  virtual void refineFeatures(
    const core::ImageSet& src,
    const std::map<int, core::Point>& pts) = 0;
public:
  virtual void config(const std::string& conf) = 0;
};


struct FeatureData
{
  core::Image ref_;    // reference appearance
  core::Image warped_; // current warped appearance
  core::Image error_;  // difference betwee the two
  double residue_;          // squared L2 norm of error_
  double first_residue_;    // squared L2 norm of error_
  int status_;              // whether the feature is tracked
  double warp_[8];          // feature warp
public:
  enum StatusCodes{ LargeResidue=-5, OutOfBounds, MaxIterations, SmallDet, NotFound, OK=0 };
public:
  FeatureData(double posx=0, double posy=0):
    ref_(width(),height(),4),
    warped_(width(),height(),4),
    error_(width(),height(),4),
    residue_(0),
    status_(OK)
  {
    warp_[0]=posx;
    warp_[1]=posy;
    warp_[2]=1;
    warp_[3]=0;
    warp_[4]=0;
    warp_[5]=1;
    warp_[6]=1;
    warp_[7]=0;
  }
public:
  static int width(){ return 15; }
  static int height(){ return 15; }
public:
  double posx() const {return warp_[0];}
  double posy() const {return warp_[1];}
  double axx() const {return warp_[2];}
  double axy() const {return warp_[3];}
  double ayx() const {return warp_[4];}
  double ayy() const {return warp_[5];}
  double lambda() const {return warp_[6];}
  double delta() const {return warp_[7];}
public:
  core::Point pt() const {return core::Point(posx(),posy());}
  double mag() const {return (axx()+ayy())/2;}
  std::vector<core::Point> bbox() const;
public:
  void setpos(double posx, double posy) {
    warp_[0]=posx;
    warp_[1]=posy;
  }
public:
  double warpx(double x, double y) const{
    return posx() + axx()*x + axy()*y;
  }
  double warpy(double x, double y) const{
    return posy() + ayx()*x + ayy()*y;
  }
};

}

}

#endif
