#ifndef TRACKER_BASE_TRACKER_HELPER_
#define TRACKER_BASE_TRACKER_HELPER_

#include "../stereo/stereo_tracker_base.h"

namespace track {

namespace TrackerHelper
{

void FilterTracksWithPriorOld(track::StereoTrackerBase& tracker,
                                     double max_disp_diff, double min_disp);
}

}

#endif
