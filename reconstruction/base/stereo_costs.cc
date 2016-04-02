#include "stereo_costs.h"

#include <iostream>

#include "../../core/types.h"

namespace recon
{
namespace StereoCosts
{

void calcPatchMeans(const cv::Mat& img, cv::Mat& means, int wsz)
{
  int margin_sz = (wsz-1) / 2;
  float N = wsz * wsz;
  means = cv::Mat::zeros(img.rows-(2*margin_sz), img.cols-2*(margin_sz), CV_32F);
  for(int y = 0; y < means.rows; y++) {
    for(int x = 0; x < means.cols; x++) {
      int start_y = y;
      int end_y = y + wsz-1;
      int start_x = x;
      int end_x = x + wsz-1;
      //std::cout << "(" << y << ", " << x << ")\n";
      for(int py = start_y; py <= end_y; py++) {
        for(int px = start_x; px <= end_x; px++) {
          //std::cout << "\t(" << py << ", " << px << ") = " << (float)img.at<uint8_t>(py,px) << "\n";
          means.at<float>(y,x) += (float)img.at<uint8_t>(py,px);
        }
      }
      means.at<float>(y,x) /= N;
    }
  }
}


void census_transform(const cv::Mat& img, int wsz, cv::Mat& census)
{
  int margin_sz = (wsz-1) / 2;
  // no CV_32U but we can cast the signed type
  census = cv::Mat::zeros(img.rows - (2*margin_sz), img.cols - (2*margin_sz), CV_32S);
  for(int y = 0; y < census.rows; y++) {
    for(int x = 0; x < census.cols; x++) {
      int start_y = y;
      int end_y = y + wsz-1;
      int start_x = x;
      int end_x = x + wsz-1;
      int cx = x + margin_sz;
      int cy = y + margin_sz;
      uint8_t pix_val = img.at<uint8_t>(cy,cx);
      //std::cout << "(" << y << ", " << x << ")\n";
      for(int py = start_y; py <= end_y; py++) {
        for(int px = start_x; px <= end_x; px++) {
          if(cx == px && cy == py) continue;
          //std::cout << "\t(" << py << ", " << px << ") = " << (float)img.at<uint8_t>(py,px) << "\n";
          census.at<uint32_t>(y,x) <<= 1;
          census.at<uint32_t>(y,x) += (img.at<uint8_t>(py,px) > pix_val ? 1 : 0);
        }
      }
    }
  }
}

uint32_t census_transform_point(const core::Point& pt, const cv::Mat& img, int wsz)
{
  int margin_sz = (wsz-1) / 2;
  int start_x = pt.x_ - margin_sz;
  int end_x = pt.x_ + margin_sz;
  int start_y = pt.y_ - margin_sz;
  int end_y = pt.y_ + margin_sz;
  if(start_x < 0 || end_x >= img.cols || start_y < 0 || end_y >= img.rows) throw "Error\n";

  uint8_t pix_val = img.at<uint8_t>(pt.y_, pt.x_);
  uint32_t census = 0;
  //std::cout << "(" << y << ", " << x << ")\n";
  for(int py = start_y; py <= end_y; py++) {
    for(int px = start_x; px <= end_x; px++) {
      if(pt.x_ == px && pt.y_ == py) continue;
      //std::cout << "\t(" << py << ", " << px << ") = " << (float)img.at<uint8_t>(py,px) << "\n";
      census <<= 1;
      census += (img.at<uint8_t>(py,px) > pix_val ? 1 : 0);
    }
  }
  return census;
}

}
}   // end recon
