#ifndef OPTIMIZATION_WEIGHT_LEARNER_
#define OPTIMIZATION_WEIGHT_LEARNER_


class WeightLearner
{
 public:
  void update_tracks(const track::StereoTrackerBase& tracker, const cv::Mat& Rt);
};

#endif
