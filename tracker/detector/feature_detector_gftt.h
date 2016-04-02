#ifndef FEATURE_DETECTOR_GFTT_
#define FEATURE_DETECTOR_GFTT_

#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "feature_detector_base.h"


namespace track {

class FeatureDetectorGFTT : public FeatureDetectorBase
{
public:
   FeatureDetectorGFTT(int block_size=3, int ksize=1, double eig_thr=0.0001, 
                       int margin_size=0, int max_corners=0);
   virtual void detect(const cv::Mat& img, std::vector<core::Point>& features);

private:
   cv::Mat mask_;
   int block_size_, ksize_, margin_size_, max_corners_;
   double eig_thr_;
};

}

#endif
