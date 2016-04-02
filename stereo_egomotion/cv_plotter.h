#ifndef STEREO_ODOMETRY_CV_PLOTTER_H_
#define STEREO_ODOMETRY_CV_PLOTTER_H_

#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core/core.hpp>

class CvPlotter
{
public:
   static const int kDefaultThickness = 2;
   static const int kDefaultLineType = 8;

   CvPlotter(unsigned w, unsigned h, double sf, double cx, double cy);
   CvPlotter(unsigned w, unsigned h);

   void drawLine(cv::Mat& pt1, cv::Mat& pt2, cv::Mat& img);
   void drawLine(cv::Point& pt1, cv::Point& pt2, cv::Mat& img);
   cv::Point matToPoint(cv::Mat& coord);
   void drawFirstFrames(std::string filename, cv::Mat& last_location, cv::Mat& disp);

private:
   double _scale_factor;
   double _cx, _cy;
   unsigned _window_height, _window_width;
   int _thickness, _line_type;
};

#endif
