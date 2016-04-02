#include "feature_refiner_klt.h"

#include "../../core/image.h"

/*
 g++ -g -std=c++11 ../base/tracker_image.cc *cc
 g++ -c -g -std=c++11 ../base/tracker_image.cc *cc
 g++ -c -g feature_refiner_test.cc
 g++ -g *.o
*/

int main(){
  core::Image img=track::maketest(80, 80, 20);
  track::savepgm("img.pgm", img);

  core::Image img2=track::equalize(img);
  track::savepgm("img2.pgm", img2);

  core::ImageSetExact tis;
  tis.compute(img);
  
  track::savepgm("smooth.pgm", track::equalize(tis.smooth_, 0, 255));
  track::savepgm("gx.pgm", track::equalize(tis.gx_));
  track::savepgm("gy.pgm", track::equalize(tis.gy_));
  
  track::FeatureRefinerKLT klt;
  std::map<int, core::Point> pts={{0,core::Point(40,40)}};
  klt.addFeatures(tis, pts);
  track::savepgm("refeq.pgm", track::equalize(klt.getFeature(0).ref_));
  track::savepgm("ref.pgm", track::equalize(klt.getFeature(0).ref_, 0, 255));
  //std::cerr <<track::print(klt.getFeature(0).ref_);

  pts[0]+=core::Point(.15,.05);
  klt.refineFeatures(tis, pts);
}

