#include "stereo_tracker_helper.h"

namespace track {

void StereoTrackerHelper::printTrackerStats(StereoTrackerBase& tracker)
{
  int max_age = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    FeatureInfo fl = tracker.featureLeft(i);
    if(fl.age_ > max_age)
      max_age = fl.age_;
  }
  int age_sum = 0;
  int tracks_num = 0;
  for(int i = 1; i <= max_age; i++) {
    int cnt = 0;
    for(int j = 0; j < tracker.countFeatures(); j++) {
      FeatureInfo fl = tracker.featureLeft(j);
      if(fl.age_ == i)
        cnt++;
    }
    age_sum += cnt * i;
    tracks_num += cnt;
    std::cout << i << ": " << cnt << " tracks\n";
  }
  //std::cout << "Tracks with age " << max_age << ": ";
  //for(int j = 0; j < tracker.countFeatures(); j++) {
  //  FeatureInfo fl = tracker.featureLeft(j);
  //  if(fl.age_ == max_age)
  //    std::cout << j << "  ";
  //}
  std::cout << "Avg. age: " << (double) age_sum / tracks_num << "\n";
}

}
