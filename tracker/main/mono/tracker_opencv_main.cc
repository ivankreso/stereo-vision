// implementation of tracking using KLT method from opencv

#include <iostream>
#include <vector>
#include <string>
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/core/operations.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include "../../../core/types.h"
#include "../../../core/image.h"
#include "../../mono/tracker_base.h"
#include "../../mono/tracker_opencv.h"
#include "../../mono/tracker_birch.h"
#include "../../base/helper_opencv.h"

#define WINDOW_WIDTH		 680
#define WINDOW_HEIGHT		 600

void drawOpticalFlow(Mat& img, track::TrackerBase& tracker, const Scalar& color)
{
   Point pt1, pt2;
   for(int i = 0; i < tracker.countTracked(); i++) {
      track::FeatureInfo& feat = tracker.feature(i);
      if(feat.status_ > 0) {
         pt1.x = feat.prev_.x_;
         pt1.y = feat.prev_.y_;
         pt2.x = feat.curr_.x_;
         pt2.y = feat.curr_.y_;
         line(img, pt1, pt2, color, 2, 8);
         circle(img, pt1, 2, Scalar(255, 0, 0), -1, 8);
         cout << pt1.x << ", " << pt1.y << endl;
         cout << pt2.x << ", " << pt2.y << endl;
      }
   }
}

void runTracker(vector<string>& imglist, string& source_folder)
{
   Mat img_mat = imread(source_folder + imglist[0], CV_LOAD_IMAGE_GRAYSCALE);
   //   core::Image img2;
   //   img2 = move(img);
   //   img = move(img2);

   //HelperOpencv::moveImageToMat(img, img_mat);
   Mat disp_cvtracker; //= Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
   Mat img_rgb;
   cvtColor(img_mat, disp_cvtracker, COLOR_GRAY2RGB);
   imshow("Cv Tracker", disp_cvtracker);
   waitKey(0);

   core::Image img;
   track::HelperOpencv::MatToImage(img_mat, img);
   //img_mat.addref();
   track::TrackerBase* tracker = new track::TrackerOpencv;
   //TrackerBase* tracker = new TrackerBirch;
   //tracker.config("params");
   tracker->init(img);

   for(size_t i = 2; i < imglist.size(); i+=2) {
      img_mat = imread(source_folder + imglist[i], CV_LOAD_IMAGE_GRAYSCALE);
      cvtColor(img_mat, disp_cvtracker, COLOR_GRAY2RGB);
      track::HelperOpencv::MatToImage(img_mat, img);
      tracker->track(img);
      drawOpticalFlow(disp_cvtracker, *tracker, Scalar(0, 0, 255));
      imshow("Cv Tracker", disp_cvtracker);
      waitKey(10);
   }

   delete tracker;
}


inline static bool readStringList(const string& filename, vector<string>& strlist)
{
   strlist.resize(0);
   FileStorage fs(filename, FileStorage::READ);
   if(!fs.isOpened())
      return false;
   FileNode n = fs.getFirstTopLevelNode();
   if(n.type() != FileNode::SEQ)
      return false;
   FileNodeIterator it = n.begin(), it_end = n.end();
   for(; it != it_end; ++it)
      strlist.push_back((string)*it);
   return true;
}


int main()
{
   string imagelistfn("/home/kreso/projects/master_thesis/src/stereo-master/config_files/bb_step3_lst.xml");
   string source_folder("/home/kreso/projects/master_thesis/datasets/bumblebee/rectified_roi/");
   vector<string> imagelist;
   bool ok = readStringList(imagelistfn, imagelist);
   if(!ok || imagelist.empty())
   {
      cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
      return -1;
   }

   runTracker(imagelist, source_folder);

   return 0;
}
