#ifndef __FEATURE_DETECTOR_GFTT_CV__
#define __FEATURE_DETECTOR_GFTT_CV__

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "feature_detector_base.h"
#include "../base/helper_opencv.h"


namespace track {

class FeatureDetectorGFTTCV : public FeatureDetectorBase
{
public:
   FeatureDetectorGFTTCV(int maxCorners=1000, double qualityLevel=0.01,
                         double minDistance=1, int blockSize=3, bool useHarrisDetector=true, double k=0.04, 
                         cv::Mat mask = cv::Mat(), int hbins=10, int vbins=10, int fpb=20);
   virtual void detect(const core::Image& img, std::vector<core::Point>& features);

private:
   cv::Mat cvimg_, mask_;
   int x1_, y1_, x2_, y2_;
   int h_bins_, v_bins_, fpb_;
   std::shared_ptr<cv::FeatureDetector> detector_;
   std::vector<cv::KeyPoint> keypoints_;
};

}

#endif
