#ifndef __FEATURE_DETECTOR_HARRIS_CV__
#define __FEATURE_DETECTOR_HARRIS_CV__

#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "feature_detector_base.h"


namespace track {

class FeatureDetectorHarrisCV : public FeatureDetectorBase
{
public:
   FeatureDetectorHarrisCV(int block_size=3, int ksize=1, double k=0.04, double eig_thr=0.0001, 
                           int margin_size=0, int max_corners=0);
   virtual void detect(const cv::Mat& img, std::vector<core::Point>& features);
   virtual void detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features);

private:
   cv::Mat mask_;
   int block_size_, ksize_, margin_size_, max_corners_;
   double k_, eig_thr_;
};

}

#endif
