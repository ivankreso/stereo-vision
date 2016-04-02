#include "helper_opencv.h"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace track {

using cv::Mat;

void HelperOpencv::MatToImage(const Mat& mat, core::Image& img)
{
   img.resize(mat.rows, mat.cols);
   for(int i = 0; i < mat.rows; i++) {
      for(int j = 0; j < mat.cols; j++)
         img(i,j) = mat.at<uint8_t>(i,j);
   }
}

void HelperOpencv::ImageToMat(const core::Image& img, Mat& mat)
{
   // first we need to decrese refcount in mat and release current data
   // otherwise data can be overwriten by Mat::zeros
   // but we dont really care about that right now... so comment out for speed
   // mat.release();

   // now allocate new memery if needed and copy data
   mat = Mat::zeros(img.rows_, img.cols_, CV_8U);
   for(int i = 0; i < img.rows_; i++) {
      for(int j = 0; j < img.cols_; j++)
         mat.at<uchar>(i,j) = img(i,j);
   }
}

void HelperOpencv::DrawDescriptor(const cv::Mat& desc, int rows, std::string name)
{
  cv::Mat patch = desc.reshape(0, rows);
  cv::resize(patch, patch, cv::Size(200,200), 0, 0, cv::INTER_NEAREST);
  cv::imshow(name, patch);
}

void HelperOpencv::DrawFloatDescriptor(const cv::Mat& desc, int rows, std::string name)
{
  cv::Mat float_patch = desc.reshape(0, rows);
  cv::Mat patch;
  float_patch.convertTo(patch, CV_8U);
  cv::resize(patch, patch, cv::Size(200,200), 0, 0, cv::INTER_NEAREST);
  cv::imshow(name, patch);
}
void HelperOpencv::DrawPoint(const core::Point& pt, const cv::Mat& img, std::string name)
{
  cv::Mat img_disp;
  cv::Point2f cvpt;
  cv::cvtColor(img, img_disp, cv::COLOR_GRAY2RGB);
  cvpt.x = pt.x_;
  cvpt.y = pt.y_;
  cv::circle(img_disp, cvpt, 3, cv::Scalar(0,255,0), -1, 8);
  cv::imshow(name, img_disp);
}

void HelperOpencv::Keypoint2Point(const cv::KeyPoint& kp, core::Point& pt)
{
  pt.x_ = kp.pt.x;
  pt.y_ = kp.pt.y;
}

void HelperOpencv::FloatImageToMat(const core::Image& img, Mat& mat)
{
   float* pmagdst = (float*)img.data_;
   mat = Mat::zeros(img.rows_, img.cols_, CV_8U);
   for(int i = 0; i < img.rows_; i++) {
      for(int j = 0; j < img.cols_; j++)
         mat.at<uint8_t>(i,j) = static_cast<uint8_t>(std::round(*pmagdst++));
   }
}
void HelperOpencv::PointsToCvPoints(const std::vector<core::Point>& in_feats,
                                    std::vector<cv::Point2f>& out_feats)
{
  out_feats.clear();
  for(auto pt : in_feats) {
    cv::Point2f cvpt;
    cvpt.x = pt.x_;
    cvpt.y = pt.y_;
    out_feats.push_back(cvpt);
  }
}

void HelperOpencv::PointsToCvKeypoints(const std::vector<core::Point>& in_feats,
                                      std::vector<cv::KeyPoint>& out_feats)
{
  out_feats.clear();
  for(auto pt : in_feats) {
    cv::KeyPoint cvkeypt;
    cvkeypt.pt.x = pt.x_;
    cvkeypt.pt.y = pt.y_;
    cvkeypt.size = 5;
    out_feats.push_back(cvkeypt);
  }
}
void HelperOpencv::DrawPatch(const core::Point& pt, const cv::Mat& img, int wsize)
{
  //cv::Mat patch = cv::Mat::zeros(wsize, wsize, CV_8U);
  int hsize = wsize / 2;
  int x_start = std::max(0, (int)pt.x_ - hsize);
  int x_end = std::min(img.cols-1, (int)pt.x_ + hsize);
  int y_start = std::max(0, (int)pt.y_ - hsize);
  int y_end = std::min(img.rows-1, (int)pt.y_ + hsize);
  int width = x_end - x_start;
  int height = y_end - y_start;
  cv::Mat patch = img(cv::Rect(x_start, y_start, width, height));
  cv::resize(patch, patch, cv::Size(200,200), 0, 0, cv::INTER_NEAREST);
  cv::imshow("patch", patch);
  cv::waitKey(0);
}

}

// not a god idea to use these because Opencv uses malloc, not c++ new like Image
//void HelperOpencv::MatToImage(Mat& mat, Image& img)
//{
//   (*img.refcount_)--;
//   if(*img.refcount_ == 0)
//      img.dealloc();
//   img.data_ = mat.data;
//   img.rows_ = mat.rows;
//   img.cols_ = mat.cols;
//   img.szBits_ = img.rows_ * img.cols_;
//   img.refcount_ = mat.refcount;
//   (*img.refcount_)++;
//}
//void HelperOpencv::ImageToMat(Image& img, Mat& mat)
//{
//   mat = Mat(Size(img.rows_, img.cols_), CV_8U, img.data_);
//   delete mat.refcount;
//   mat.refcount = img.refcount_;
//   (*mat.refcount)++;
//}
//

