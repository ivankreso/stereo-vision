// ./visual_odometry -i intrinsics.yml -e extrinsics.yml -s ../data/cropped/ -o ../data/out/ ../data/img_list.xml
#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <unistd.h>
#include <sys/wait.h>

#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

#include "opencv2/core/core.hpp"
#include "opencv2/core/operations.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

#include "../../../tracker/stereo/stereo_tracker_base.h"
#include "../../../tracker/stereo/stereo_tracker_sim.h"
#include "../../../tracker/stereo/stereo_tracker_libviso.h"
#include "../../../tracker/stereo/stereo_tracker_bfm.h"
#include "../../../tracker/detector/feature_detector_harris_cv.h"
#include "../../../tracker/detector/feature_detector_gftt_cv.h"
#include "../../../tracker/detector/feature_detector_base.h"
#include "../../extern/libviso2/src/viso_stereo.h"
#include "../../extern/libviso2/src/matrix.h"
#include "../../feature_filter.h"
#include "../../math_helper.h"
#include "../../eval_helper.h"
#include "../../format_helper.h"
#include "../../cv_plotter.h"
#include "../../matches.h"
using namespace vo;

// bumblebee dataset
#define CALIB_WIDTH		 640
#define CALIB_HEIGHT		 480
#define MAX_FEATURES		 500

// Lucas Kanade tracking params
// bumblebee dataset
//#define STEREO_BOX_W		 13		// 13
//#define STEREO_BOX_H		 5		// 5 or 3
//#define TRACK_BOX_W		 21 // 21
//#define TRACK_BOX_H		 21 // 21
//#define LK_PYRAMID_LEVEL	 3
//#define BORDER_FILTER		 5
//#define BORDER_MASK_SIZE	 10

#define STEREO_BOX_W		 21		// 13
#define STEREO_BOX_H		 21		// 5 or 3
#define TRACK_BOX_W		 21 // 21
#define TRACK_BOX_H		 21 // 21
#define LK_PYRAMID_LEVEL	 3
#define BORDER_FILTER		 5
#define BORDER_MASK_SIZE	 5

// detector params
#define QUALITY_LVL		 1	  // 0.15
#define MIN_DISTANCE		 4	  // 10
#define BLOCK_SIZE		 3	  // 3
#define USE_HARRIS		 true	  // false
#define HARRIS_K         0.04	  // 0.04

// libviso dataset
//#define STEREO_BOX_W		 11		 // 13
//#define STEREO_BOX_H		 5		// 5 or 3
//#define TRACK_BOX_W		 11 // 21
//#define TRACK_BOX_H		 11 // 21
//#define LK_PYRAMID_LEVEL	 3  // 5
//#define BORDER_FILTER		 5
//#define BORDER_MASK_SIZE	 60

//TODO hardcoded - in future add movement checking through optical flow
// feature tracking params
#define SBA_FRAME_STEP		      2  // 10

// plotting settings
#define SCALE_FACTOR             0.5	       // 0.5, zoom with 2.0
#define CENTER_X		            300
#define CENTER_Y		            200
#define WINDOW_WIDTH		         900
#define WINDOW_HEIGHT		      700
//#define WINDOW_CAMERA_WIDTH	   1000
//#define WINDOW_CAMERA_HEIGHT	   800

using namespace std;
//using namespace cv;

// functions
void getCalibParams(string& intrinsic_filename, string& extrinsic_filename, Mat& P_left, Mat& P_right, Mat& Q, 
                    Mat& C_left, Mat &D_left, Mat& C_right, Mat& D_right);

static int printHelp()
{
   cout << "Usage:\n ./visual_odometry -i intrinsics.yml -e extrinsics.yml -s source_folder -o output_folder <image "
           "list XML/YML file>\n" << endl;
   return 0;
}

void printPoint() {}
void printMatch() {}

void writeExtrinsicParams(deque<Mat>& Rt_params, fstream& file)
{
   // Rt for extr params (no cam pose)
   // TODO left or right coord system 
   double quat[4];
   //cout << extr_params[0] << "\n -> 0-Rt\n";
   //MathHelper::matToQuat(Rt_params[0], quat);
   //cout << quat << endl;

   std::setprecision(6);

   for(size_t i = 0; i < Rt_params.size(); i++) {
      MathHelper::matToQuat(Rt_params[i], quat);
      for(int j = 0; j < 4; j++) {
         file << quat[j] << " ";
      }
      Mat& Rt = Rt_params[i];
      file << Rt.at<double>(0,3) << " " << Rt.at<double>(1,3) << " " << Rt.at<double>(2,3)<< endl;
   }
   // TODO - y
}

void writeMatRt(Mat& Rt, ofstream& fp)
{
   std::setprecision(6);
   for(int i = 0; i < (Rt.rows-1); i++) {
      for(int j = 0; j < Rt.cols; j++)
         fp << Rt.at<double>(i,j) << " ";
   }
   fp << endl;
   //file << Rt.at<double>(0,3) << " " << Rt.at<double>(1,3) << " " << Rt.at<double>(2,3)<< endl;
}

void writePointProjections(fstream& file, deque<Matches>& features, deque<vector<uchar>>& track_status,
      deque<Mat>& ctw_params, VisualOdometryStereo::parameters& param)
{
   assert(features.size() == SBA_FRAME_STEP && track_status.size() == (features.size()-1));
   //unsigned start = features_left.size() - SBA_FRAME_STEP;
   Mat pt(4, 1, CV_64F);
   double d;
   //   double rerr_left = 0.0;
   //   double rerr_right = 0.0;
   Mat C = Mat::eye(3, 3, CV_64F);
   for(size_t k = 0; k < features.size(); k++) {
      for(size_t i = 0; i < features[k].left_.size(); i++) {
         if(k > 0 && track_status[k-1][i] == 1)
            continue;

         d = max(features[k].left_[i].pt.x - features[k].right_[i].pt.x, 1.f);
         pt.at<double>(0,0) = (features[k].left_[i].pt.x - param.calib.cu) * param.base / d;
         pt.at<double>(1,0) = (features[k].left_[i].pt.y - param.calib.cv) * param.base / d;
         pt.at<double>(2,0) = param.calib.f * param.base / d;
         pt.at<double>(3,0) = 1.0;

         for(int f = k; f >= 0; f--) {
            pt = ctw_params[f] * pt;
         }
         //	 pt = ctw_params[k] * pt;
         //TODO p.y = -p.y
         //assert(pt.at<double>(2,0) < 100.0);

         uint32_t cnt = 1;
         for(unsigned j = k;; j++) {
            if(j == track_status.size())
               break;
            if(track_status[j][i] == 1)
               cnt++;
            else
               break;
         }
         if(cnt == 1)
            continue;
         file << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << " ";
         //cout << p.x << " " << p.y << " " << p.z << " ";

         file << cnt;
         //cout << c;
         for(unsigned j = k; j < (k+cnt); j++) {
            file << " " << j << " " << features[j].left_[i].pt.x << " " << features[j].left_[i].pt.y;
            //cout << " " << j - start << " " << features_left[j][i].pt.x << " " << features_left[j][i].pt.y;
         }
         file << endl;
         //cout << endl;

         // test triangulation
         //	 cout << "start triang. test\n";
         //	 cout << "1. frame in file: " << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << "\n";
         //	 for(uint32_t j = k; j < (k+cnt); j++) {
         //	    d = max(features[j].left_[i].pt.x - features[j].right_[i].pt.x, 1.f);
         //	    pt.at<double>(0,0) = (features[j].left_[i].pt.x - param.calib.cu) * param.base / d;
         //	    pt.at<double>(1,0) = (features[j].left_[i].pt.y - param.calib.cv) * param.base / d;
         //	    pt.at<double>(2,0) = param.calib.f * param.base / d;
         //	    pt.at<double>(3,0) = 1.0;
         //	    cout << (j-k)+1 << ". frame: " << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << "  -->  ";
         //	    for(int32_t f = j; f >= 0; f--) {
         //	       pt = ctw_params[f] * pt;
         //	    }
         //	    cout << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << "\n";
         //	 }
         //	 cout << "end triang. test\n";
      }
   }
   //cout << "reproj error left: " << rerr_left << "\nreproj error right: " << rerr_right << "\n";
}


void writePointProjectionsFromFristFrame(fstream& file, deque<Matches>& features, deque<vector<int32_t>>& track_idx,
                                         deque<Mat>& ctw_params, VisualOdometryStereo::parameters& param)
{
   //   cout << features_left.size() << endl;
   //   assert(features_left.size() == SBA_FRAME_STEP && features_left.size() == features_right.size() && track_status.size() == (features_left.size()-1));
   //unsigned start = features_left.size() - SBA_FRAME_STEP;
   Mat pt(4, 1, CV_64F);
   double d;

   //   for(int32_t f = ctw_params.size()-1; f >= 0; f--) {
   //      cout << f << " frame: " << ctw_params[f] << endl;
   //   }

   Mat C = Mat::eye(3, 3, CV_64F);
   for(size_t k = 0; k <= 0; k++) {
      for(size_t i = 0; i < features[k].left_.size(); i++) {
         if(k > 0 && track_idx[k-1][i] >= 0)
            continue;
         // TODO racunat sa nenormaliziranom znacajkama zbog ciste rektifikacije...?
         d = max(features[k].left_[i].pt.x - features[k].right_[i].pt.x, 1.f);
         pt.at<double>(0,0) = (features[k].left_[i].pt.x - param.calib.cu) * param.base / d;
         pt.at<double>(1,0) = (features[k].left_[i].pt.y - param.calib.cv) * param.base / d;
         pt.at<double>(2,0) = param.calib.f * param.base / d;
         pt.at<double>(3,0) = 1.0;
         for(int32_t f = k; f >= 0; f--) {
            pt = ctw_params[f] * pt;
         }
         //	 pt = ctw_params[k] * pt;
         //TODO p.y = -p.y
         //assert(pt.at<double>(2,0) < 100.0);

         //	 Point3f pt_right(p.x-0.12, p.y, p.z);
         //	 rerr_left += MathHelper::getReprojError(features_left[k][i].pt, p, C);
         //	 rerr_right += MathHelper::getReprojError(features_right[k][i].pt, pt_right, C);

         // TODO
         //	 p.x += extr_params[k][4];
         //	 p.y += extr_params[k][5];
         //	 p.z += extr_params[k][6];
         //cout << "camera system: " << pt << endl;
         //pt = extr_params[k] * pt;
         //cout << "world system: " << pt << endl;
         uint32_t cnt = 1;
         int32_t index = i;
         for(uint32_t j = k;; j++) {
            if(j == track_idx.size())
               break;
            if(track_idx[j][index] >= 0) {
               cnt++;
               index = track_idx[j][index];
            }
            else
               break;
         }
         if(cnt == 1)
            continue;
         file << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << " ";
         //cout << p.x << " " << p.y << " " << p.z << " ";

         file << cnt;
         //cout << c;
         index = i;
         for(uint32_t j = k; j < (k+cnt); j++) {
            if(j > k)
               index = track_idx[j-1][index];
            file << " " << j << " " << features[j].left_[index].pt.x << " " << features[j].left_[index].pt.y;
            //cout << " " << j - start << " " << features_left[j][i].pt.x << " " << features_left[j][i].pt.y;
         }
         file << endl;
         //cout << endl;

         // test triangulation
         //	 index = i;
         //	 cout << "start triang. test\n";
         //	 cout << "1. frame in file: " << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << "\n";
         //	 for(uint32_t j = k; j < (k+cnt); j++) {
         //	    if(j > k)
         //	       index = track_idx[j-1][index];
         //
         //	    d = max(features[j].left_[index].pt.x - features[j].right_[index].pt.x, 1.f);
         //	    pt.at<double>(0,0) = (features[j].left_[index].pt.x - param.calib.cu) * param.base / d;
         //	    pt.at<double>(1,0) = (features[j].left_[index].pt.y - param.calib.cv) * param.base / d;
         //	    pt.at<double>(2,0) = param.calib.f * param.base / d;
         //	    pt.at<double>(3,0) = 1.0;
         //	    cout << (j-k)+1 << ". frame: " << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << "  -->  ";
         //	    for(int32_t f = j; f >= 0; f--) {
         //	       pt = ctw_params[f] * pt;
         //	    }
         //	    cout << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << "\n";
         //	 }
         //	 cout << "end triang. test\n";
      }
   }
   //cout << "reproj error left: " << rerr_left << "\nreproj error right: " << rerr_right << "\n";
}



void writePointProjections2Frames(fstream& file, deque<vector<KeyPoint>>& features_left, deque<vector<KeyPoint>>& features_right, 
      deque<Mat>& extr_inv, VisualOdometryStereo::parameters& param)
{
   assert(features_left.size() >= SBA_FRAME_STEP && features_left.size() == features_right.size());
   //assert(features_left.size() == extr_params.size());
   unsigned start = features_left.size() - SBA_FRAME_STEP;
   Mat pt(4, 1, CV_64F);
   double d;
   Mat C = Mat::eye(3, 3, CV_64F);
   for(unsigned k = start; k < (start+1); k++) {
      for(unsigned i = 0; i < features_left[k].size(); i++) {
         // TODO ovdje dodat sakupljeni pomak od prvog frejma
         // todo racunat sa nenormaliziranom znacajkama zbog ciste rektifikacije...?
         d = max(features_left[k][i].pt.x - features_right[k][i].pt.x, 1.f);
         pt.at<double>(0,0) = (features_left[k][i].pt.x - param.calib.cu) * param.base / d;
         pt.at<double>(1,0) = (features_left[k][i].pt.y - param.calib.cv) * param.base / d;
         pt.at<double>(2,0) = param.calib.f * param.base / d;
         pt.at<double>(3,0) = 1.0;
         //TODO p.y = -p.y
         //assert(pt.at<double>(2,0) < 100.0);

         //	 Point3f pt_right(p.x-0.12, p.y, p.z);
         //	 rerr_left += MathHelper::getReprojError(features_left[k][i].pt, p, C);
         //	 rerr_right += MathHelper::getReprojError(features_right[k][i].pt, pt_right, C);

         // TODO
         //cout << "camera system: " << pt << endl;
         //pt = extr_inv[k-start] * pt;
         //cout << "world system: " << pt << endl;

         file << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << " ";
         //cout << pt.at<double>(0,0) << " " << pt.at<double>(1,0) << " " << pt.at<double>(2,0) << " ";
         unsigned c = 2;
         file << c;
         //cout << c;
         for(unsigned j = k; j < (k+c); j++) {
            file << " " << j - start << " " << features_left[j][i].pt.x << " " << features_left[j][i].pt.y;
            //cout << " " << j - start << " " << features_left[j][i].pt.x << " " << features_left[j][i].pt.y;
         }
         file << endl;
         //cout << endl;
      }
   }
}


// TODO - not direct motion but inverse camera motion matrix?
void readMotion(string filename, double motion_params[7])
{
   ifstream file(filename);

   string line, tmp_line;

   while(std::getline(file, tmp_line)) {
      line = tmp_line;
   }
   stringstream stream(line);
   string val;
   //cout << "LINE: " << line << endl;
   for(int i = 0; i < 7; i++) {
      stream >> val;
      motion_params[i] = std::stod(val);
      //cout << "READ: " << motion_params[i] << endl << endl;
   }

   file.close();
}

void matToKeypoints(Mat& mat, vector<KeyPoint>& keys)
{
   assert(mat.cols == keys.size() && mat.rows == 1 && mat.channels() == 2);
   for(unsigned i = 0; i < keys.size(); i++) {
      keys[i].pt.x = mat.at<Vec2d>(0,i)[0];
      keys[i].pt.y = mat.at<Vec2d>(0,i)[1];
   }
}
void keypointsToMat(vector<KeyPoint>& keys, Mat& mat)
{
   assert(mat.cols == keys.size() && mat.rows == 1 && mat.channels() == 2);
   for(unsigned i = 0; i < keys.size(); i++) {
      mat.at<Vec2d>(0,i)[0] = keys[i].pt.x;
      mat.at<Vec2d>(0,i)[1] = keys[i].pt.y;
   }
}

void cvKeysToVisoMatches(deque<vector<KeyPoint>>& keys_left, deque<vector<KeyPoint>>& keys_right, 
      vector<uchar>& status, vector<Matcher::p_match>& matches)
{
   //assert(keys_left.back().size() == keys_right.back.size() && keys_left.back().size() == status.size());
   Matcher::p_match m;

   size_t last = keys_left.size() - 1;
   for(size_t i = 0; i < status.size(); i++) {
      if(status[i] == 1) {
         m.u1p = keys_left[last-1][i].pt.x;
         m.v1p = keys_left[last-1][i].pt.y;
         m.u1c = keys_left[last][i].pt.x;
         m.v1c = keys_left[last][i].pt.y;
         m.u2p = keys_right[last-1][i].pt.x;
         m.v2p = keys_right[last-1][i].pt.y;
         m.u2c = keys_right[last][i].pt.x;
         m.v2c = keys_right[last][i].pt.y;
         matches.push_back(m);
      }
   }
}

void cvKeysToVisoMatches(deque<Matches>& all_features, vector<uchar>& status, vector<Matcher::p_match>& matches)
{
   Matcher::p_match m;

   size_t last = all_features.size() - 1;
   assert(all_features[last-1].left_.size() == status.size());   
   for(size_t i = 0; i < status.size(); i++) {
      if(status[i] == 1) {
         m.u1p = all_features[last-1].left_[i].pt.x;
         m.v1p = all_features[last-1].left_[i].pt.y;
         m.u1c = all_features[last].left_[i].pt.x;
         m.v1c = all_features[last].left_[i].pt.y;
         m.u2p = all_features[last-1].right_[i].pt.x;
         m.v2p = all_features[last-1].right_[i].pt.y;
         m.u2c = all_features[last].right_[i].pt.x;
         m.v2c = all_features[last].right_[i].pt.y;
         //cout << m.u1p << " " << m.v1p << endl << m.u1c << " " << m.v1c << endl;
         matches.push_back(m);
      }
   }
}

void convertAllMatchesToKeys(vector<Matcher::p_match>& matches, vector<vector<KeyPoint>>& keypoints)
{
   keypoints.resize(4);
   KeyPoint kp;
   for(size_t i = 0; i < matches.size(); i++) {
      //cout << "match " << i << endl;
      kp.pt.x = matches[i].u1p;
      kp.pt.y = matches[i].v1p;
      keypoints[0].push_back(kp);
      kp.pt.x = matches[i].u2p;
      kp.pt.y = matches[i].v2p;
      keypoints[1].push_back(kp);
      kp.pt.x = matches[i].u1c;
      kp.pt.y = matches[i].v1c;
      keypoints[2].push_back(kp);
      kp.pt.x = matches[i].u2c;
      kp.pt.y = matches[i].v2c;
      keypoints[3].push_back(kp);
   }
}
void convertInlierMatchesToKeys(vector<Matcher::p_match>& matches, vector<int32_t>& inliers, vector<vector<KeyPoint>>& keypoints)
{
   keypoints.resize(4);
   KeyPoint kp;
   for(size_t i = 0; i < inliers.size(); i++) {
      //cout << "match " << i << endl;
      int32_t idx = inliers[i];
      kp.pt.x = matches[idx].u1p;
      kp.pt.y = matches[idx].v1p;
      keypoints[0].push_back(kp);
      kp.pt.x = matches[idx].u2p;
      kp.pt.y = matches[idx].v2p;
      keypoints[1].push_back(kp);
      kp.pt.x = matches[idx].u1c;
      kp.pt.y = matches[idx].v1c;
      keypoints[2].push_back(kp);
      kp.pt.x = matches[idx].u2c;
      kp.pt.y = matches[idx].v2c;
      keypoints[3].push_back(kp);
   }
}
Mat getCameraMatrix(VisualOdometryStereo::parameters& param)
{
   Mat C = Mat::zeros(3, 4, CV_64F);
   C.at<double>(0,0) = param.calib.f;
   C.at<double>(1,1) = param.calib.f;
   C.at<double>(2,2) = 1.0;
   C.at<double>(0,2) = param.calib.cu;
   C.at<double>(1,2) = param.calib.cv;
   return C;
}

void readGpsPoints(string filename, vector<Mat> points)
{
   ifstream file(filename);
   string val;
   points.clear();
   Mat pt(2, 1, CV_64F);
   while(!file.eof()) {
      file >> val;
      pt.at<double>(0,0) = std::stod(val);
      file >> val;
      pt.at<double>(1,0) = std::stod(val);
      cout << pt << endl;
      points.push_back(pt.clone());
      for(int i = 0; i < 9; i++) file >> val;
   }
}

void visualOdometry(vector<string>& imagelist, string& cparams_file, string& source_folder, string& output_folder)
{
   vector<KeyPoint> features_full;
   vector<KeyPoint> features_left;
   vector<KeyPoint> features_right;

   deque<vector<uchar>> track_status;
   deque<Matches> all_features;
   deque<vector<KeyPoint>> features_libviso_left;
   deque<vector<KeyPoint>> features_libviso_right;

   deque<Mat> extr_params; // pose mat
   deque<Mat> Rt_params;

   double cam_params[5];
   FormatHelper::readCameraParams(cparams_file, cam_params);
   VisualOdometryStereo::parameters param;
   param.calib.f = cam_params[0];	 // focal length in pixels
   param.calib.cu = cam_params[2];	 // principal point (u-coordinate) in pixels
   param.calib.cv = cam_params[3];	 // principal point (v-coordinate) in pixels
   param.base = cam_params[4];

   // bumblebee dataset - rectified - lcd calib
   //param.calib.f  = 491.22;	    // focal length 
   //param.calib.cu = 328.8657;	 // principal point (u-coordinate) in pixels
   //param.calib.cv = 249.43165;	 // principal point (v-coordinate) in pixels
   //param.base	  = 0.119744;
   ////param.base	  = 0.5;

   // old lcd
   //param.calib.f  = 491.598;	 // focal length 
   //param.calib.cu = 336.437;	 // principal point (u-coordinate) in pixels
   //param.calib.cv = 249.277;	 // principal point (v-coordinate) in pixels
   //param.base	  = 0.12;

   // bumblebee - alan calib rectified - with opencv
   //param.calib.f  = 492.2473;	 // focal length in pixels
   //param.calib.cu = 333.5488;	 // principal point (u-coordinate) in pixels
   //param.calib.cv = 240.6432;	 // principal point (v-coordinate) in pixels
   //param.base	  = 0.12043;      // 386.1448 / 718.856

   // bumblebee - alan calib rectified - original from pdf - i dont have calib data, only rectif data 
// param.calib.f = 425.3849;
// param.calib.cu = 325.9;
// param.calib.cv = 244.54275;
// param.base = 0.12;

   // libviso datasets: 00, 14
   //param.calib.f  = 718.856;	 // focal length in pixels
   //param.calib.cu = 607.1928;	 // principal point (u-coordinate) in pixels
   //param.calib.cv = 185.2157;	 // principal point (v-coordinate) in pixels
   //param.base	  = 0.53716;      // 386.1448 / 718.856

   // libviso 11 dataset
   //param.calib.f  = 707.0912;	 // focal length in pixels
   //param.calib.cu = 601.8873;	 // principal point (u-coordinate) in pixels
   //param.calib.cv = 183.1104;	 // principal point (v-coordinate) in pixels
   //param.base	  = 0.53715;      // 386.1448 / 718.856
   // optional
   //   param.inlier_threshold = 1.5;       // 1.5
   //   param.ransac_iters = 200;	       // 200
   //   param.bucket.bucket_height = 50;    // 50
   //   param.bucket.bucket_width = 50;     // 50
   //param.bucket.max_features = 6;      // 2

   //Image nullimg;
   string simdata_folder = "/home/kivan/projects/datasets/stereo_model/points_kitti_cam_nonoise_base_0.50/";
   string simdata_xml = "/home/kivan/projects/datasets/stereo_model/stereosim_viso00path.xml";
   Mat mask = imread("/home/kivan/Projects/project-vista/config_files/mask_kitti_rect.png", CV_LOAD_IMAGE_GRAYSCALE);
   FeatureDetectorBase* detector = new FeatureDetectorGFTTCV(5000, 0.0001, 1, 3, true, 0.04, mask);
   //FeatureDetectorBase* detector = new FeatureDetectorHarrisCV(10, 10, 10, 3, 1, 0.04);
   StereoTrackerBase* tracker = new StereoTrackerBFM(detector, 2000, 0.9);
   //StereoTrackerBase* tracker = new StereoTrackerSim(simdata_folder, simdata_xml);
   //StereoTrackerBase* tracker = new StereoTrackerLibviso(param);
   // init visual odometry
   VisualOdometryStereo viso(param, tracker);
   // LIBVISO end

   //Mat P_left, P_right, Q, C_left, D_left, C_right, D_right;
   //getCalibParams(intrinsic_filename, extrinsic_filename, P_left, P_right, Q, C_left, D_left, C_right, D_right);

   //	Mat descriptors_1, descriptors_2;
   Mat img_left_prev = imread(source_folder + imagelist[0], CV_LOAD_IMAGE_GRAYSCALE);
   Mat img_right_prev = imread(source_folder + imagelist[1], CV_LOAD_IMAGE_GRAYSCALE);

   //TODO try histo eq;

   Matrix pose = Matrix::eye(4);
   Matrix viso_cvtrack = Matrix::eye(4);
   //Matrix point_rt = Matrix::eye(4);
   Mat Rt_inv;
   Mat mat_I = Mat::eye(4, 4, CV_64F);
   mat_I.copyTo(Rt_inv);
   Vec<double,7> trans_vec;
   extr_params.push_back(mat_I.clone());
   Mat Rt(4, 4, CV_64F);
   Mat Rt_gt(4, 4, CV_64F);
   Mat Rt_gt_prev = Mat::eye(4, 4, CV_64F);
   Mat Rt_gt_curr = Mat::eye(4, 4, CV_64F);
   MathHelper::invTrans(Rt_inv, Rt);
   Rt_params.push_back(Rt.clone());
   cout << Rt << endl;
   cout << Rt_inv << endl;

   Mat prev_location_viso = Mat::zeros(4, 1, CV_64F);
   Mat prev_location_viso_cvtrack = Mat::zeros(4, 1, CV_64F);   
   Mat prev_location_sba = Mat::zeros(4, 1, CV_64F);

   int32_t dims[] = {img_left_prev.cols, img_left_prev.rows, img_left_prev.cols};
   // init the tracker
   if(!viso.process(img_left_prev.data, img_right_prev.data, dims))
      cout << "init frame - no estimation" << endl;

   Matches matches(features_left, features_right);

   Mat disp_libviso = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
   Mat disp_libviso_cvtracker = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
   Mat disp_libviso_birch = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
   Mat disp_sba = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
   Mat disp_camera = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
   Mat disp_camera_right = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

   imshow("libviso_orig", disp_libviso);
   //imshow("libviso_cvtracker", disp_libviso_cvtracker);
   //imshow("SBA", disp_sba);
   //moveWindow("libviso_orig", 0, 0);
   //moveWindow("libviso_cvtracker", 682, 0);

   // for SBA
   int frame = 2;
   string extr_filename = "cam_extr.txt";
   string points_filename = "cam_points.txt";
   fstream extr_file;
   fstream points_file;
   //   vector<Mat> gps_pts;
   //   readGpsPoints("gps_route.txt", gps_pts);

   Mat Rt_sba_path = Mat::eye(4, 4, CV_64F);	   // sba extrinsic params in whole scene path
   Mat Rt_sba = Mat::eye(4, 4, CV_64F);		   // sba output extrinsic params
   Mat pose_sba = Mat::eye(4, 4, CV_64F);
   Mat pose_libviso = Mat::eye(4, 4, CV_64F);   // matrix transforms points in current camera coordinate system to world coord system
   CvPlotter plotter(WINDOW_WIDTH, WINDOW_HEIGHT, SCALE_FACTOR , CENTER_X, CENTER_Y);

   ifstream groundtruth_file("/home/kivan/Projects/datasets/KITTI/poses/00.txt");
   ofstream reprojerr_file("stereo_reproj_error.txt");
   ofstream libviso_ptsfile("viso_points.txt");
   writeMatRt(pose_libviso, libviso_ptsfile);

   // -- demo start
   //string left_out_folder = "/home/kivan/projects/master_thesis/demo/libviso_demo/left_cam/";
   //string right_out_folder = "/home/kivan/projects/master_thesis/demo/libviso_demo/right_cam/";
   //std::ostringstream frame_num;
   //frame_num << std::setw(6) << std::setfill('0') << 0;
   //cout << frame_num.str() << "\n";   
   //imwrite(left_out_folder + "img_left_" + frame_num.str() + ".jpg", img_left_prev);
   //imwrite(right_out_folder + "img_right_" + frame_num.str() + ".jpg", img_right_prev);
   // -- demo end

   for(unsigned i = 2; i < imagelist.size(); i+=(2)) {
      cout << source_folder + imagelist[i] << endl;
      Mat img_left = imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
      Mat img_right = imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
      cout << i/2 << " frame:" << endl;

      if(viso.process(img_left.data, img_right.data, dims)) {
         vector<Matcher::p_match>& libviso_features = viso.getFeatures();
         vector<int32_t>& inliers = viso.getInliers();
//         double reproj_err = MathHelper::getReprojError(features_libviso_left, features_libviso_right, param.base, C, Rt);
//         cout << "reproj error: " << reproj_err << endl;
         
         ////	 cout << "libviso features size: " << viso.p_matched.size() << endl;
         vector<vector<KeyPoint>> matches, matches_all;
         convertInlierMatchesToKeys(libviso_features, inliers, matches);
         //convertAllMatchesToKeys(libviso_features, matches_all);

         // SBA begin
         //features_libviso_left.clear();
         //features_libviso_right.clear();
         //features_libviso_left.push_back(matches[0]);
         //features_libviso_right.push_back(matches[1]);
         //features_libviso_left.push_back(matches[2]);
         //features_libviso_right.push_back(matches[3]);
         // SBA end

         Mat disp_allfeats;
         vector<uchar> viso_status(matches[0].size(), 1);
         cvtColor(img_left_prev, disp_camera, COLOR_GRAY2RGB);
         cvtColor(img_left_prev, disp_allfeats, COLOR_GRAY2RGB);
         FeatureFilter::drawOpticalFlow(disp_camera, matches[0], matches[2], viso_status, Scalar(0,0,255));
         //FeatureFilter::drawOpticalFlow(disp_allfeats, matches_all[0], matches_all[2], viso_status, Scalar(0,0,255));
         imshow("camera_left", disp_camera);
         //imshow("camera_left_all", disp_allfeats);
         //moveWindow("camera left", 400, 200);

         // save imgs for demo
         // -- demo start
         //cvtColor(img_right, disp_camera_right, COLOR_GRAY2RGB);         
         //FeatureFilter::drawOpticalFlow(disp_camera_right, matches[1], matches[3], viso_status, Scalar(0,0,255));         
         //frame_num.str("");
         //frame_num << std::setw(6) << std::setfill('0') << (i/2);
         //imwrite(left_out_folder + "img_left_" + frame_num.str() + ".jpg", disp_camera);
         //imwrite(right_out_folder + "img_right_" + frame_num.str() + ".jpg", disp_camera_right);
         // -- demo end

         // on success, update current pose
         // TODO try without inverse
         pose = pose * Matrix::inv(viso.getMotion());
         //cout << pose << endl << endl;
         MathHelper::matrixToMat(pose, pose_libviso);         
         writeMatRt(pose_libviso, libviso_ptsfile);
         //viso_ptsfile << ~pose.extractCols(std::vector<int>(1,3)) << endl;

         //	 cout << "point_rt - rt * viso.getMotion:\n" << point_rt_mat << endl; 
         //TODO - why Rt and point_rt not excatly the same? because of Matrix::inv?
         Matrix Rt_inv_libviso = Matrix::inv(viso.getMotion());
         Matrix Rt_libviso = viso.getMotion();
         MathHelper::matrixToMat(Rt_inv_libviso, Rt_inv);
         MathHelper::matrixToMat(Rt_libviso, Rt);
         //MathHelper::invTrans(Rt, Rt_inv);    // better
         extr_params.push_back(Rt_inv.clone());

         // output some statistics
         vector<core::Point> points_lp, points_rp, points_lc, points_rc;
         FeatureHelper::LibvisoInliersToPoints(libviso_features, inliers, points_lp, points_rp, points_lc, points_rc);
         //cout << points_lp.size() << ", " << points_rp.size() << ", " << points_lc.size() << ", " << points_rc.size() << endl;
         Mat C = getCameraMatrix(param);
         //cout << Rt_inv << "\n";

         FormatHelper::readNextMatrixKitti(groundtruth_file, Rt_gt_curr);
         Mat Rt_gt_inv = Rt_gt_prev * Rt_gt_curr;
         //cout << Rt_gt_inv << endl;
         MathHelper::invTrans(Rt_gt_inv, Rt_gt);
         //Rt_gt = Rt_gt_inv;
         cout << "Groundtruth:\n" << Rt_gt << endl;
         cout << "Odometry:\n" << Rt << endl;
         cout << "-------------------------------\n";
         MathHelper::invTrans(Rt_gt_curr, Rt_gt_prev);
         double reproj_error = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc, C, 
                                                                Rt, param.base);
         double reproj_error_gt = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc, C, 
                                                                   Rt_gt, param.base);
         cout << "reprojection error (libviso): " << reproj_error << "\n";
         cout << "reprojection error (groundtruth): " << reproj_error_gt << "\n";
         cout << "-----------------------------------------------------------------------\n";
         reprojerr_file << reproj_error << "\n";
         if(reproj_error > reproj_error_gt) {
            cout << "Found better for GT!!!\n";
            //return;
         }

         double num_matches = viso.getNumberOfMatches();
         double num_inliers = viso.getNumberOfInliers();
         cout << "Matches: " << num_matches;
         cout << ", Inliers: " << num_inliers << " -> " << 100.0*num_inliers/num_matches << " %" << endl;
         Mat location_viso(pose_libviso, Range(0,4), Range(3,4)); // extract 4-th column
         //cout << location_viso << endl;
         //Matrix location = ~point_rt.extractCols(std::vector<int>(1,3));
         //for(int k = 0; k < 3; k++) location.val[0][k] = - location.val[0][k];
         //cout << point_rt * ~location << endl;
         //cout << location << endl << endl;

         // drawing
         //drawLine(disp_libviso, Point(100.5,100.5), Point(500,500));
         //Point pt1 = coordToPoint(prev_location, SCALE_FACTOR);
         //Point pt2 = coordToPoint(location, SCALE_FACTOR);
         plotter.drawLine(prev_location_viso, location_viso, disp_libviso);
         imshow("libviso_orig", disp_libviso);
         waitKey(10);
         //cout << pose << endl << endl;
         //cout << endl << location.val[0][3] << endl;

         location_viso.copyTo(prev_location_viso);
      } else {
         cout << "libviso ... failed!" << endl;
         waitKey(0);
         //writeMatRt(pose_libviso, libviso_ptsfile);
         //extr_params.push_back(Rt_inv.clone());
         exit(1);
      }


      // BEGIN SBA
      //Rt_params.push_back(Rt.clone());

      //if(frame >= SBA_FRAME_STEP) {
      //   extr_file.open(extr_filename, std::ios_base::out | std::fstream::trunc);
      //   writeExtrinsicParams(Rt_params, extr_file);
      //   extr_file.close();

      //   points_file.open(points_filename, std::ios_base::out | std::fstream::trunc);
      //   deque<vector<uchar>> status_viso;
      //   vector<uchar> s_viso(features_libviso_left[0].size(), 1);
      //   status_viso.push_back(s_viso);

      //   //writePointProjectionsFromFristFrame(points_file, all_features, track_idx, extr_params, param);
      //   //writePointProjections(points_file, all_features, track_status, extr_params, param);	 
      //   //writePointProjections(points_file, features_libviso_left, features_libviso_right, status_viso, extr_params, param);
      //   writePointProjections2Frames(points_file, features_libviso_left, features_libviso_right, extr_params, param);
      //   //writePointProjectionsDebug(points_file, all_features_left, all_features_right, track_status, extr_params, imagelist, source_folder);
      //   points_file.close();

      //   pid_t pid = fork();
      //   if(pid >= 0) {
      //      // child
      //      if(pid == 0) {
      //         //execl("eucsbademo", "eucsbademo", "cam_extr.txt", "cam_points.txt", "C_left_cam.txt", "output.txt", (char*)NULL);
      //         //const char* camera_mat = normalized_points ? "C_normal.txt" : "C_left_cam.txt";
      //         //const char* camera_mat = "C_libviso_00.txt";
      //         const char* camera_mat = "C_bb.txt"; // TODO - why not better
      //         cout << "Using camera matrix: " << camera_mat << endl;
      //         execl("eucsbademo_mot", "eucsbademo_mot", "cam_extr.txt", "cam_points.txt", camera_mat, "output.txt", (char*)NULL);
      //      }

      //      //execl("eucsbademo", "eucsbademo", "cam_extr.txt", "cam_points.txt", "cam_intr.txt", "output.txt", (char*)NULL);
      //      // parent
      //      else
      //         wait(NULL);
      //   }
      //   else {
      //      cout << "fork failed... exiting." << endl;
      //      exit(-1);
      //   }

      //   if(frame == SBA_FRAME_STEP)
      //      plotter.drawFirstFrames("output.txt", prev_location_sba, disp_sba);

      //   double motion_params[7];
      //   readMotion("output.txt", motion_params);
      //   // TODO ovo i za libviso
      //   MathHelper::quatToMat(motion_params, Rt_sba);
      //   Rt_sba_path = Rt_sba_path * Rt_sba;
      //   //	 cout << "viso Rt:\n" << Rt << endl;
      //   //	 cout << "sba Rt:\n" << Rt_sba << endl;

      //   //Mat pose_sba = Rt_sba_path.inv();
      //   Mat pose_sba_tmp;
      //   MathHelper::invTrans(Rt_sba, pose_sba_tmp);
      //   pose_sba = pose_sba * pose_sba_tmp;
      //   Mat location_sba(4, 1, CV_64F);
      //   location_sba.at<double>(0,0) = pose_sba.at<double>(0,3);
      //   location_sba.at<double>(1,0) = pose_sba.at<double>(1,3);
      //   location_sba.at<double>(2,0) = pose_sba.at<double>(2,3);

      //   // isto kao iznad sa pose_sba?
      //   //	 location_sba.at<double>(0,0) = Rt_sba_path.at<double>(0,3);
      //   //	 location_sba.at<double>(1,0) = -Rt_sba_path.at<double>(1,3);
      //   //	 location_sba.at<double>(2,0) = -Rt_sba_path.at<double>(2,3);

      //   //	 Point pt1 = coordToPoint(prev_location_sba, SCALE_FACTOR);
      //   //	 Point pt2 = coordToPoint(location_sba, SCALE_FACTOR);
      //   plotter.drawLine(prev_location_sba, location_sba, disp_sba);
      //   //imshow("Libviso", disp_libviso);
      //   imshow("SBA", disp_sba);
      //   waitKey(10);
      //   location_sba.copyTo(prev_location_sba);
      //   // calculate and print reproj errors
      //   Mat C = getCameraMatrix(param);
      //   double err_viso_l, err_viso_r, err_sba_l, err_sba_r;
      //   MathHelper::getReprojError(features_libviso_left, features_libviso_right, param.base, C, Rt, err_viso_l, err_viso_r);
      //   MathHelper::getReprojError(features_libviso_left, features_libviso_right, param.base, C, Rt_sba, err_sba_l, err_sba_r);
      //   cout << "Reprojection error (libviso):\n" << "left: " << err_viso_l << "\nright: " << err_viso_r << endl;
      //   cout << "Reprojection error (sba):\n" << "left: " << err_sba_l << "\nright: " << err_sba_r << endl;

      //   //all_features.pop_front();
      //   //track_status.pop_front();
      //   Rt_params.pop_front();
      //   Rt_params[0] = mat_I.clone();
      //   extr_params.pop_front();
      //   //extr_params.pop_front();
      //   //extr_params.push_front(mat_I.clone());
      //   extr_params[0] = mat_I.clone();
      //}
      // END SBA
      //

      frame++;

      cv::swap(img_left_prev, img_left);
      cv::swap(img_right_prev, img_right);
      //		img_left_prev.deallocate();
      //		img_left_prev = std::move(img_left);
      //		img_right_prev.deallocate();
      //		img_right_prev = std::move(img_right);
   }
   groundtruth_file.close();
   reprojerr_file.close();
   libviso_ptsfile.close();
   waitKey(0);
}


void getCalibParams(String& intrinsic_filename, String& extrinsic_filename, Mat& P_left,
                    Mat& P_right, Mat& Q, Mat& C_left, Mat& D_left, Mat& C_right, Mat& D_right)
{
   // reading intrinsic parameters
   FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
   if(!fs.isOpened())
   {
      cout << "Failed to open file " << intrinsic_filename << endl;
      return;
   }

   //Mat M1, D1, M2, D2;
   fs["M1"] >> C_left;
   fs["D1"] >> D_left;
   fs["M2"] >> C_right;
   fs["D2"] >> D_right;
   Mat R, T, R1, R2;
   fs["R"] >> R;
   fs["T"] >> T;

   cv::Size imageSize(CALIB_WIDTH, CALIB_HEIGHT);
   stereoRectify(C_left, D_left, C_right, D_right, imageSize, R, T, R1, R2, P_left, P_right, Q, CALIB_ZERO_DISPARITY, -1,
         imageSize);
   return;
}



inline static bool readStringList(const string& filename, vector<string>& l)
{
   l.resize(0);
   FileStorage fs(filename, FileStorage::READ);
   if( !fs.isOpened() )
      return false;
   FileNode n = fs.getFirstTopLevelNode();
   if( n.type() != FileNode::SEQ )
      return false;
   FileNodeIterator it = n.begin(), it_end = n.end();
   for( ; it != it_end; ++it )
      l.push_back((string)*it);
   return true;
}

int main(int argc, char	** argv)
{
   string config_file;
   string imagelistfn;
   string cam_params_file;
   string source_folder;
   string output_folder;

   try {
      po::options_description generic("Generic options");
      generic.add_options()
         ("help", "produce help message")
         ("config,c", po::value<string>(&config_file)->default_value("config.txt"), "config filename")
         ;
      po::options_description config("Config file options");
      config.add_options()
         ("camera_params,p", po::value<string>(&cam_params_file)->default_value("camera_params.txt"), "camera params file")
         ("source_folder,s", po::value<string>(&source_folder), "folder with source")
         ("output_folder,o", po::value<string>(&output_folder), "folder for output")
         ("imglist,l", po::value<string>(&imagelistfn), "file with image list")
         ;

      po::options_description cmdline_options;
      cmdline_options.add(generic).add(config);

      po::options_description config_file_options;
      config_file_options.add(config);
      po::variables_map vm;
      //po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
      po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
      notify(vm);
      if(vm.count("help")) {
         cout << generic;
         cout << config;
         return 0;
      }

      ifstream ifs(config_file.c_str());
      if (!ifs) {
         cout << "can not open config file: " << config_file << "\n";
         cout << generic;
         cout << config;
         return 0;
      }
      else {
         po::store(parse_config_file(ifs, config_file_options), vm);
         notify(vm);
      }
      cout << "Configuring done, using:" << endl;

      if(vm.count("camera_params")) {
         cout << "Camera params: ";
         cout << cam_params_file << endl;
      }
      if(vm.count("source_folder")) {
         cout << "Source folder: ";
         cout << source_folder << endl;
      }
      if(vm.count("output_folder")) {
         cout << "Output folder: ";
         cout << output_folder << endl;
      }
      if(vm.count("imglist")) {
         cout << "Image list file: ";
         cout << imagelistfn << endl;
      }
   }
   catch(std::exception& e) {
      cout << e.what() << "\n";
      return 1;
   }

   if(imagelistfn == "")
   {
      cout << "error: no xml image list given." << endl;
      return printHelp();
   }
   if(output_folder == "" || source_folder == "")
   {
      cout << "error: no output or source folder given." << endl;
      return printHelp();
   }

   vector<string> imagelist;
   bool ok = readStringList(imagelistfn, imagelist);
   if(!ok || imagelist.empty())
   {
      cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
      return printHelp();
   }

   visualOdometry(imagelist, cam_params_file, source_folder, output_folder);

   return 0;
}
