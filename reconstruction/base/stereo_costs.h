#ifndef RECONSTRUCTION_BASE_STEREO_COSTS_
#define RECONSTRUCTION_BASE_STEREO_COSTS_

#include <iostream>
#include <unordered_map>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>

#include "../../core/types.h"

namespace recon
{

typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixCensusCosts;

namespace StereoCosts
{

void calcPatchMeans(const cv::Mat& img, cv::Mat& means, int wsz);


template<typename T>
uint8_t hamming_dist(T x, T y);

uint32_t get_cost_SAD(const cv::Mat& left_img, const cv::Mat& right_img, int wsz, int cx, int cy, int d);

float get_cost_ZSAD(const cv::Mat& left_img, const cv::Mat& right_img,
                    const cv::Mat& left_means, const cv::Mat& right_means,
                    int wsz, int cx, int cy, int d);

double get_cost_NCC(const core::DescriptorNCC& d1, const core::DescriptorNCC& d2);

void census_transform(const cv::Mat& img, int wsz, cv::Mat& census);
uint32_t census_transform_point(const core::Point& pt, const cv::Mat& img, int wsz);

template<typename T>
void compute_ncc_descriptor(const cv::Mat& img, const core::Point& feat, const int window_sz,
                            const int cv_type, core::DescriptorNCC& desc);

template<typename T>
void compute_ncc_descriptor(const cv::Mat& img, const int cx, const int cy, const int window_sz,
                            const int cv_type, core::DescriptorNCC& desc);

void compute_image_ncc_descriptors(const cv::Mat& img, int window_sz,
                                   std::vector<core::DescriptorNCC>& desciptors);

} // end namespace: StereoCosts

template<typename T>
inline
void StereoCosts::compute_ncc_descriptor(const cv::Mat& img, const int cx, const int cy, const int window_sz,
                                         const int cv_type, core::DescriptorNCC& desc)
{
  int desc_sz = window_sz * window_sz;
  int margin_sz = (window_sz - 1) / 2;
  //desc.vec.create(desc_sz, 1, CV_8U);
  //desc.vec.create(desc_sz, 1, CV_32F);
  desc.vec.create(desc_sz, 1, cv_type);
  desc.A = 0.0;
  desc.B = 0.0;
  desc.C = 0.0;
  assert(cx >= margin_sz && cx < (img.cols - margin_sz));
  assert(cy >= margin_sz && cy < (img.rows - margin_sz));
  int vpos = 0;
  for(int y = cy - margin_sz; y <= cy + margin_sz; y++) {
    for(int x = cx - margin_sz; x <= cx + margin_sz; x++) {
      T val = img.at<T>(y,x);
      desc.vec.at<T>(vpos) = val;
      double dval = static_cast<double>(val);
      desc.A += dval;
      desc.B += dval*dval;
      vpos++;
    }
  }
  // var - variance * N^2
  double var = std::sqrt((desc_sz * desc.B) - (desc.A * desc.A));
  // C = N^2 / variance
  if (var > 0.0)
    desc.C = 1.0 / var;
  // if variance equals 0 set negative C
  else
    desc.C = -1.0;
  assert(!std::isnan(desc.C));
  assert(!std::isinf(desc.C));
}

template<typename T>
inline
void StereoCosts::compute_ncc_descriptor(const cv::Mat& img, const core::Point& feat, const int window_sz,
                                         const int cv_type, core::DescriptorNCC& desc)
{
  int cx = static_cast<int>(feat.x_);
  int cy = static_cast<int>(feat.y_);
  compute_ncc_descriptor<T>(img, cx, cy, window_sz, cv_type, desc);
}

inline
void StereoCosts::compute_image_ncc_descriptors(const cv::Mat& img, int window_sz,
                                                std::vector<core::DescriptorNCC>& desciptors)
{
  int margin_sz = (window_sz-1) / 2;
  int width = img.cols - 2*margin_sz;
  int height = img.rows - 2*margin_sz;
  int num_of_desc = width * height;
  desciptors.resize(num_of_desc);

  #pragma omp parallel for
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      core::Point pt;
      pt.x_ = x + margin_sz;
      pt.y_ = y + margin_sz;
      compute_ncc_descriptor<uint8_t>(img, pt, window_sz, CV_8U, desciptors[y*width + x]);
    }
  }
}

template<typename T>
inline
uint8_t StereoCosts::hamming_dist(T x, T y)
{
  uint8_t dist = 0;
  T val = x ^ y; // XOR
  // Count the number of set bits
  while(val) {
    ++dist;
    val &= val - 1;
  }
  return dist;
}

inline
uint32_t StereoCosts::get_cost_SAD(const cv::Mat& left_img, const cv::Mat& right_img, int wsz, int cx, int cy, int d)
{
  int ssz = (wsz-1) / 2;
  int SAD = 0;
  for(int y = (cy - ssz); y <= (cy + ssz); y++) {
    for(int x = (cx - ssz); x <= (cx + ssz); x++) {
      int idiff = static_cast<int>(left_img.at<uint8_t>(y,x) - right_img.at<uint8_t>(y,x-d));
      SAD += std::abs(idiff);
   }
  }
  //std::cout << "SAD = " << sad << "\n";
  return SAD;
}

inline
float StereoCosts::get_cost_ZSAD(const cv::Mat& left_img, const cv::Mat& right_img,
                                 const cv::Mat& left_means, const cv::Mat& right_means,
                                 int wsz, int cx, int cy, int d)
{
  assert(left_img.rows == (left_means.rows + (wsz-1)));
  int ssz = (wsz-1) / 2;
  float zsad = 0.0f;
  int cx2 = cx - ssz;
  int cy2 = cy - ssz;
  float mean_diff = left_means.at<float>(cy2,cx2) - right_means.at<float>(cy2,cx2-d);
  for(int y = (cy - ssz); y <= (cy + ssz); y++) {
    for(int x = (cx - ssz); x <= (cx + ssz); x++) {
      float idiff = left_img.at<uint8_t>(y,x) - right_img.at<uint8_t>(y,x-d);
      zsad += std::abs(idiff - mean_diff);
   }
  }
  //std::cout << "ZSAD = " << zsad << "\n";
  return zsad;
}

inline
double StereoCosts::get_cost_NCC(const core::DescriptorNCC& d1, const core::DescriptorNCC& d2)
{
  assert(d1.vec.rows == d2.vec.rows);
  
  if (d1.C < 0.0 || d2.C < 0.0)
    return -1.0;
  double n = d1.vec.rows;
  double D = d1.vec.dot(d2.vec);
  double ncc = (n * D - (d1.A * d2.A)) * d1.C * d2.C;
  //if(std::isnan(ncc) || std::isinf(ncc)) {
  //  printf("TRUE! NCC = %f\n", ncc);
  //  printf("D = %f\n", D);
  //  ncc = 0.0;
  //  throw 1;
  //}
  return ncc;
}

}

#endif
