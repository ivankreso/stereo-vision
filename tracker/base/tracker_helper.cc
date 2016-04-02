#include "tracker_helper.h"

//#include <vector>
//#include <opencv2/core/core.hpp>
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

#include "types.h"

namespace track {
namespace TrackerHelper {
void FilterTracksWithPriorOld(track::StereoTrackerBase& tracker,
                              double max_disp_diff, double min_disp) {
  int filtered_cnt = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo left = tracker.featureLeft(i);
    track::FeatureInfo right = tracker.featureRight(i);
    if(left.age_ < 1) continue;
    double d_p = left.prev_.x_ - right.prev_.x_;
    double d_c = left.curr_.x_ - right.curr_.x_;
    //double max_disp_diff = 30.0; // 07
    //double min_disp = 0.1;
    //if(d_p < min_disp || d_c < min_disp)
    //  printf("\33[0;31m Small disp: %f -> %f !\33[0m\n", d_p, d_c);
    //if(std::abs(d_p - d_c) > max_disp_diff) {
    if(std::abs(d_p - d_c) > max_disp_diff || d_p < min_disp || d_c < min_disp) {
      //printf("\33[0;31m [Filtering]: Small disp or big disp difference: %f -> %f !\33[0m\n", d_p, d_c);
      //std::cout << "Feature index = " << i << '\n';
      //std::cout << "Previous:\n" << left.prev_ << '\n' << right.prev_ << '\n';
      //std::cout << "Current:\n" << left.curr_ << '\n' << right.curr_ << '\n';
      //tracker.showTrack(tracker_idx);

      tracker.removeTrack(i);
      filtered_cnt++;
    }
  }
  //printf("\33[0;31m [Filtering]: Num of filtered features = %d\33[0m\n", filtered_cnt);
}

}
}
