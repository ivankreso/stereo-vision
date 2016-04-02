#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;

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

#include "../../../core/image.h"
#include "../../../tracker/stereo/stereo_tracker_base.h"
#include "../../../tracker/stereo/stereo_tracker_sim.h"
#include "../../../tracker/stereo/stereo_tracker_libviso.h"
#include "../../../tracker/stereo/stereo_tracker_bfm.h"
#include "../../../tracker/stereo/stereo_tracker_refiner.h"
#include "../../../tracker/stereo/tracker_refiner_libviso.h"
#include "../../../tracker/refiner/feature_refiner_base.cc"
#include "../../../tracker/refiner/feature_refiner_klt.h"
#include "../../../tracker/detector/feature_detector_harris_cv.h"
#include "../../../tracker/detector/feature_detector_gftt_cv.h"
#include "../../../tracker/detector/feature_detector_base.h"
#include "../../extern/libviso2/src/viso_stereo.h"
#include "../../extern/libviso2/src/matrix.h"
#include "../../math_helper.h"
#include "../../eval_helper.h"
#include "../../format_helper.h"
#include "../../helper_libviso.h"
#include "../../cv_plotter.h"
#include "../../matches.h"

using namespace core;
using namespace vo;
using namespace track;

// Lucas Kanade tracking params
// bumblebee dataset
//#define STEREO_BOX_W		 13		// 13
//#define STEREO_BOX_H		 5		// 5 or 3
//#define TRACK_BOX_W		 21 // 21
//#define TRACK_BOX_H		 21 // 21
//#define LK_PYRAMID_LEVEL	 3
//#define BORDER_FILTER		 5
//#define BORDER_MASK_SIZE	 10

//#define STEREO_BOX_W		 21		// 13
//#define STEREO_BOX_H		 21		// 5 or 3
//#define TRACK_BOX_W		 21 // 21
//#define TRACK_BOX_H		 21 // 21
//#define LK_PYRAMID_LEVEL	 3
//#define BORDER_FILTER		 5
//#define BORDER_MASK_SIZE	 5

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
 
   // optional
   //   param.inlier_threshold = 1.5;       // 1.5
   //   param.ransac_iters = 200;	       // 200
   //param.bucket.bucket_height = 50;    // 50
   //param.bucket.bucket_width = 50;     // 50
   param.bucket.max_features = 3;      // 2

   // bumblebee dataset - rectified - lcd calib
   //param.calib.f  = 491.22;	    // focal length 
   //param.calib.cu = 328.8657;	 // principal point (u-coordinate) in pixels
   //param.calib.cv = 249.43165;	 // principal point (v-coordinate) in pixels
   //param.base	  = 0.119744;
   ////param.base	  = 0.5;

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

   //Image nullimg;
   //string simdata_folder = "/home/kivan/projects/datasets/stereo_model/points_kitti_cam_nonoise_base_0.50/";
   //string simdata_xml = "/home/kivan/projects/datasets/stereo_model/stereosim_viso00path.xml";
   Mat mask = imread("/home/kivan/Projects/project-vista/config_files/mask_kitti_rect.png", CV_LOAD_IMAGE_GRAYSCALE);

   //StereoTrackerBase* tracker = new StereoTrackerSim(simdata_folder, simdata_xml);

   FeatureRefinerKLT refiner;

   FeatureDetectorGFTTCV detector(5000, 0.0001, 1, 3, true, 0.04, mask);
   StereoTrackerBFM tracker_basic(&detector, 2000, 0.9);
   StereoTrackerRefiner tracker(&tracker_basic, &refiner, tracker_basic.countFeatures());

   //StereoTrackerLibviso tracker_basic(param);
   //TrackerRefinerLibviso tracker(&tracker_basic, &refiner);

   // init visual odometry
   VisualOdometryStereo viso(param, &tracker);
   //VisualOdometryStereo viso(param, &tracker_basic);
   // LIBVISO end

   int start_frame = 0 * 2;
   //	Mat descriptors_1, descriptors_2;
   Mat img_left_prev = imread(source_folder + imagelist[start_frame], CV_LOAD_IMAGE_GRAYSCALE);
   Mat img_right_prev = imread(source_folder + imagelist[start_frame+1], CV_LOAD_IMAGE_GRAYSCALE);

   Image tmp_img;
   HelperOpencv::MatToImage(img_left_prev, tmp_img);
   ImageSetExact imgset;
   imgset.compute(tmp_img);

   // first calc margins size of gradient images
   const int gradient_margin = (imgset.kernelGrad_.size() / 2) + 1;
   // calculate feature detector mask using feature size and extra gradient margin size
   const int fhw = (FeatureData::width() / 2) + gradient_margin;
   const int fhh = (FeatureData::height() / 2) + gradient_margin;
   detector.setMask(fhw, fhh, img_left_prev.cols - fhw, img_left_prev.rows - fhh);

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

   Mat pose_libviso = Mat::eye(4, 4, CV_64F);   // matrix transforms points in current camera coordinate system to world coord system
   CvPlotter plotter(WINDOW_WIDTH, WINDOW_HEIGHT, SCALE_FACTOR , CENTER_X, CENTER_Y);

   ifstream groundtruth_file("/home/kivan/Projects/datasets/KITTI/poses/07.txt");
   // skip first (identety) matrix
   FormatHelper::readNextMatrixKitti(groundtruth_file, Rt_gt_curr);
   ofstream reprojerr_file("stereo_reproj_error.txt");
   ofstream libviso_ptsfile("viso_points.txt");
   FormatHelper::writeMatRt(pose_libviso, libviso_ptsfile);

   // -- demo start
   //string left_out_folder = "/home/kivan/projects/master_thesis/demo/libviso_demo/left_cam/";
   //string right_out_folder = "/home/kivan/projects/master_thesis/demo/libviso_demo/right_cam/";
   //std::ostringstream frame_num;
   //frame_num << std::setw(6) << std::setfill('0') << 0;
   //cout << frame_num.str() << "\n";   
   //imwrite(left_out_folder + "img_left_" + frame_num.str() + ".jpg", img_left_prev);
   //imwrite(right_out_folder + "img_right_" + frame_num.str() + ".jpg", img_right_prev);
   // -- demo end

   for(unsigned i = start_frame + 2; i < imagelist.size(); i+=(2)) {
      cout << source_folder + imagelist[i] << endl;
      Mat img_left = imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
      Mat img_right = imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
      cout << i/2 << " frame:" << endl;

      //if(i/2 == 5)
      //   break;

      if(viso.process(img_left.data, img_right.data, dims)) {
         vector<Matcher::p_match>& libviso_features = viso.getFeatures();
         vector<int32_t>& inliers = viso.getInliers();

         //vector<vector<KeyPoint>> matches, matches_all;
         //HelperLibviso::convertInlierMatchesToKeys(libviso_features, inliers, matches);
         ////HelperLibviso::convertAllMatchesToKeys(libviso_features, matches_all);
         //Mat disp_allfeats;
         //vector<uchar> viso_status(matches[0].size(), 1);
         //cvtColor(img_left_prev, disp_camera, COLOR_GRAY2RGB);
         //cvtColor(img_left_prev, disp_allfeats, COLOR_GRAY2RGB);
         //HelperLibviso::drawOpticalFlow(disp_camera, matches[0], matches[2], viso_status, Scalar(0,0,255));
         ////FeatureFilter::drawOpticalFlow(disp_allfeats, matches_all[0], matches_all[2], viso_status, Scalar(0,0,255));
         //imshow("camera_left", disp_camera);
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
         FormatHelper::writeMatRt(pose_libviso, libviso_ptsfile);
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
         Mat C = HelperLibviso::getCameraMatrix(param);
         //cout << Rt_inv << "\n";

         FormatHelper::readNextMatrixKitti(groundtruth_file, Rt_gt_curr);
         Mat Rt_gt_inv = Rt_gt_prev * Rt_gt_curr;
         //cout << Rt_gt_inv << endl;
         MathHelper::invTrans(Rt_gt_inv, Rt_gt);
         //Rt_gt = Rt_gt_inv;
         //cout << "Groundtruth:\n" << Rt_gt << endl;
         //cout << "Odometry:\n" << Rt << endl;
         cout << "-------------------------------\n";
         MathHelper::invTrans(Rt_gt_curr, Rt_gt_prev);
         double reproj_error = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc, C, 
                                                                Rt, param.base);
         double reproj_error_gt = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc, C, 
                                                                   Rt_gt, param.base);
         cout << "reprojection error (libviso): " << reproj_error << "\n";
         cout << "reprojection error (groundtruth): " << reproj_error_gt << "\n";
         cout << "-----------------------------------------------------------------------\n";
         // reprojerr_file << reproj_error << "\n";
         if(reproj_error > reproj_error_gt) {
            cout << "Found better for GT!!!\n";
            waitKey(0);
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

      cv::swap(img_left_prev, img_left);
      cv::swap(img_right_prev, img_right);
   }
   
   destroyAllWindows();
   waitKey(0);
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
      return -1;
   }
   if(output_folder == "" || source_folder == "")
   {
      cout << "error: no output or source folder given." << endl;
      return -1;
   }

   vector<string> imagelist;
   bool ok = FormatHelper::readStringList(imagelistfn, imagelist);
   if(!ok || imagelist.empty())
   {
      cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
      return -1;
   }

   visualOdometry(imagelist, cam_params_file, source_folder, output_folder);

   return 0;
}
