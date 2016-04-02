#include "feature_detector_harris_cv.h"
#include "../base/helper_opencv.h"

namespace {

struct greaterThanPtr : public std::binary_function<const float*, const float*, bool>
{
    bool operator () (const float* a, const float* b) const
    { return *a > *b; }
};

}

namespace track {

FeatureDetectorHarrisCV::FeatureDetectorHarrisCV(int block_size, int ksize, double k, double eig_thr, 
                                                 int margin_size, int max_corners) {
  block_size_ = block_size;
  ksize_ = ksize;
  k_ = k;
  eig_thr_ = eig_thr;
  margin_size_ = margin_size;
  max_corners_ = max_corners;
}

//void FeatureDetectorHarrisCV::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features)
//{
//  features.clear();
//  std::vector<core::Point> feats;
//  detect(img, feats);
//  for(size_t i = 0; i < feats.size(); i++) {
//    cv::KeyPoint kp;
//    kp.pt.x = feats[i].x_;
//    kp.pt.y = feats[i].y_;
//    kp.size = 15.0;
//    features.push_back(kp);
//  }
//}

void FeatureDetectorHarrisCV::detect(const cv::Mat& img, std::vector<cv::KeyPoint>& features) {
  features.clear();
  //Mat img_harris = Mat::zeros(img.rows_, img.cols_, CV_32FC1);
  cv::Mat img_harris, img_dilate;
  cv::cornerHarris(img, img_harris, block_size_, ksize_, k_, cv::BORDER_DEFAULT);
  //threshold(img_harris, threshed, 0.0001, 255, THRESH_BINARY_INV); // just for visualisation

  //cv::Mat element = getStructuringElement(MORPH_RECT, cv::Size(2*dilation_size + 1, 2*dilation_size+1),
  // maybe not the same block_size_ used for harris?
  //cv::Mat struct_element = getStructuringElement(cv::MORPH_RECT, cv::Size(block_size_, block_size_));
  //cv::dilate(img_harris, img_dilate, struct_element);

  cv::threshold(img_harris, img_harris, eig_thr_, 0, cv::THRESH_TOZERO);
  // used for non-max supression
  cv::dilate(img_harris, img_dilate, cv::Mat());    // default block size - 3x3

  cv::Size imgsize = img.size();
  std::vector<const float*> responses;

  // collect list of pointers to features - put them into temporary image
  int marsz;
  if(margin_size_ > 0)
    marsz = margin_size_;
  else
    marsz = 1;

  for(int y = marsz; y < imgsize.height - marsz; y++) {
    const float* eig_data = (const float*)img_harris.ptr(y);
    const float* tmp_data = (const float*)img_dilate.ptr(y);

    for(int x = marsz; x < (imgsize.width - marsz); x++)
    {
      float val = eig_data[x];
      // supress the non-max area
      // in this case it is safe to compare floats directly for performance reasons
      if(val != 0 && val == tmp_data[x]) {
        //std::cout << val << " -- " << *(eig_data+x) << "\n";
        responses.push_back(eig_data + x);
      }
    }
  }

  // sort corners by response - best first
  std::sort(responses.begin(), responses.end(), greaterThanPtr());

  int total = responses.size(), ncorners = 0;
  for(int i = 0; i < total; i++) {
    int ofs = (int)((const uchar*)responses[i] - img_harris.data);
    int y = (int)(ofs / img_harris.step);
    int x = (int)((ofs - y*img_harris.step)/sizeof(float));

    cv::KeyPoint kpt;
    kpt.pt.x = x;
    kpt.pt.y = y;
    kpt.size = block_size_;
    kpt.response = *responses[i];
    features.push_back(kpt);
    ++ncorners;
    if(max_corners_ > 0 && (int)ncorners == max_corners_)
      break;
  }

  //std::cout << "[Harris]: Detected features = " << features.size() << "\n";
}

void FeatureDetectorHarrisCV::detect(const cv::Mat& img, std::vector<core::Point>& features) {
  throw 1;
  features.clear();
  //Mat img_harris = Mat::zeros(img.rows_, img.cols_, CV_32FC1);
  cv::Mat img_harris, img_dilate;
  cv::cornerHarris(img, img_harris, block_size_, ksize_, k_, cv::BORDER_DEFAULT);
  //threshold(img_harris, threshed, 0.0001, 255, THRESH_BINARY_INV); // just for visualisation

  //cv::Mat element = getStructuringElement(MORPH_RECT, cv::Size(2*dilation_size + 1, 2*dilation_size+1),
  // maybe not the same block_size_ used for harris?
  //cv::Mat struct_element = getStructuringElement(cv::MORPH_RECT, cv::Size(block_size_, block_size_));
  //cv::dilate(img_harris, img_dilate, struct_element);

  cv::threshold(img_harris, img_harris, eig_thr_, 0, cv::THRESH_TOZERO);
  // used for non-max supression
  cv::dilate(img_harris, img_dilate, cv::Mat());    // default block size - 3x3

  cv::Size imgsize = img.size();
  std::vector<const float*> tmpCorners;

  // collect list of pointers to features - put them into temporary image
  int marsz;
  if(margin_size_ > 0)
    marsz = margin_size_;
  else
    marsz = 1;

  for(int y = marsz; y < imgsize.height - marsz; y++)
  {
    const float* eig_data = (const float*)img_harris.ptr(y);
    const float* tmp_data = (const float*)img_dilate.ptr(y);

    for(int x = marsz; x < (imgsize.width - marsz); x++)
    {
      float val = eig_data[x];
      // supress the non-max area
      // in this case it is safe to compare floats directly for performance reasons
      if(val != 0 && val == tmp_data[x]) {
        //std::cout << val << " -- " << *(eig_data+x) << "\n";
        tmpCorners.push_back(eig_data + x);
      }
    }
  }

  // sort corners by response - best first
  std::sort(tmpCorners.begin(), tmpCorners.end(), greaterThanPtr());

  int total = tmpCorners.size(), ncorners = 0;
  for(int i = 0; i < total; i++)
  {
    int ofs = (int)((const uchar*)tmpCorners[i] - img_harris.data);
    int y = (int)(ofs / img_harris.step);
    int x = (int)((ofs - y*img_harris.step)/sizeof(float));

    features.push_back(core::Point((double)x, (double)y));
    ++ncorners;
    if(max_corners_ > 0 && (int)ncorners == max_corners_)
      break;
  }

  std::cout << "[Harris]: Detected features = " << features.size() << "\n";
  //for(size_t i = 0; i < features.size(); i++) {
  //  std::cout << i << " - " << features[i]  << " - response = " << *tmpCorners[i] << "\n";
  //  HelperOpencv::DrawPatch(features[i], img, 21);
  //  cv::waitKey(0);
  //}

  //for(size_t i = 0; i < features.size(); i++)
  //  std::cout << features[i] << " -> " << *tmpCorners[i] << "\n";

  //cv::Mat disp_img = img;
  //cv::cvtColor(img, disp_img, cv::COLOR_GRAY2RGB);
  //vo::FeatureHelper::drawFeatures(features, img);

  ////imshow("Harris response", img_harris);
  ////imshow("Harris response dilate", img_dilate);
  //cv::waitKey(0);
}

}
