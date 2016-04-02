#ifndef TRACKER_TRACKER_BASE_H_
#define TRACKER_TRACKER_BASE_H_

#include <string>

#include "../../core/image.h"
#include "../../core/types.h"
#include "../base/types.h"

namespace track {

class TrackerBase {
 public:
  virtual ~TrackerBase() {}

  virtual void config(std::string conf) { throw "Error\n"; }
  virtual std::string getConfig() { throw "Error\n"; }
  virtual std::string getConfigDocs() { throw "Error\n"; }

  virtual int init(const core::Image& img) { throw "Error\n"; }
  virtual int track(const core::Image& img) { throw "Error\n"; }
  virtual int init(const cv::Mat& img) { throw "Error\n"; }
  virtual int track(const cv::Mat& img) { throw "Error\n"; }

  virtual int countTracked() = 0;
  virtual int countFeatures() = 0;
  virtual FeatureInfo feature(int i) = 0;
  virtual void removeTrack(int i) = 0;
  
  virtual void showTrack(int i) { throw "Error\n"; }

  virtual int getAge(int idx) const { throw "Error\n"; }
  virtual bool isAlive(int idx) const { throw "Error\n"; }
  virtual FeatureData getFeatureData(int i) { throw "[TrackerBase]: Calling stub!\n"; }
};

}

#endif
