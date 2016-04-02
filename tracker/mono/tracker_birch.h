#ifndef TRACKER_BIRCH_H_
#define TRACKER_BIRCH_H_

#include "tracker_base.h"
#include "../../core/types.h"

#include "birch/klt.h"

#include "config_parser.h"

namespace track {

class TrackerBirch: public TrackerBase {
public:
   TrackerBirch();
   ~TrackerBirch();

   virtual std::string getConfigDocs();
   virtual void config(std::string conf);
   virtual std::string getConfig();

   virtual int init(const core::Image& img);
   virtual int track(const core::Image& img);

   virtual int countFeatures();
   virtual int countTracked();
   virtual FeatureInfo feature(int i);
   
   virtual void removeTrack(int id) {}

private:
  ConfigParserMap config_;
  ConfigParserMap configPyr_;

  std::vector<FeatureInfo> feats_;

  KLT_TrackingContext kltTC_;
  KLT_FeatureList     kltFL_;

private:
  void reset();

  int warpMode();
  double thWarp();
  int replaceLost();
  int szWinWarp();
  int szWinTrans();
  int pyrSize();
  int pyrSubsampling();
  double pyrFactorSigma();

};

}

#endif
