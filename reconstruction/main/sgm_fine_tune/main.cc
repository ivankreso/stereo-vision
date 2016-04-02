#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "../../../core/format_helper.h"

//#include "../../base/stereo_cvsgbm.h"
#include "../../base/stereo_sgm.h"

void computeSGM();

//cv::StereoSGBM sgbm;
cv::Mat img_left, img_right;
cv::Mat img_disp, mat_disp, img_disp_save, mat_disp_save;

int sad_window_size = 5;                  // 3 - 11
int sad_window_size_max = 11;
int P1 = 10;                              // 70
int P1_max = 100;
int P2 = 50;                             // 800
int P2_max = 600;
//int frame_num = 391;
int frame_num = 102;
int frame_num_max = 1000;
int pre_filter_cap = 40;                  // 70
int pre_filter_cap_max = 200;
int min_disparity = 0;
int min_disparity_max = 20;
//int number_of_disparities = 128;          // or 256 max
int number_of_disparities = 200;          // or 256 max
int number_of_disparities_max = 512;
int uniqueness_ratio = 0;                 // 0 - 15
int uniqueness_ratio_max = 15;
int speckle_window_size = 200;            // 100
int speckle_window_size_max = 1000;
int speckle_range = 32;
int speckle_range_max = 128;
int disp12_max_diff = 1;                 // 1
int disp12_max_diff_max = 20;
int full_dp = 1;
int full_dp_max = 1;

void updateParameters()
{
  if(P2 <= P1) {
    P2 = P1 + 1;
    std::cout << "Warning: P2 too small! Using P2 = " << P2 << "\n";
  }
  int residue = number_of_disparities % 16;
  if(residue != 0) {
    number_of_disparities += (16 - residue);
    std::cout << "Warning: number_of_disparities \% 16 != 0! Using number_of_disparities = " << number_of_disparities << "\n";
  }
  //sgbm.preFilterCap = pre_filter_cap;
  //sgbm.SADWindowSize = sad_window_size;
  //sgbm.P1 = P1;
  //sgbm.P2 = P2;
  //sgbm.minDisparity = min_disparity;
  //sgbm.numberOfDisparities = number_of_disparities;
  //sgbm.uniquenessRatio = uniqueness_ratio;
  //sgbm.speckleWindowSize = speckle_window_size;
  //sgbm.speckleRange = speckle_range;
  //sgbm.disp12MaxDiff = disp12_max_diff;
  //sgbm.fullDP = full_dp;
}

void on_trackbar(int, void*)
{
  updateParameters();
}

void computeSGM()
{
  //recon::StereoSGBMParams params(min_disparity, number_of_disparities, sad_window_size, P1, P2, disp12_max_diff,
  //                               pre_filter_cap, uniqueness_ratio, speckle_window_size, speckle_range, full_dp);

  //cv::Mat buffer;
  //recon::computeDisparitySGBM(img_left, img_right, img_disp, params, buffer);
  //recon::computeSGM(img_left, img_right, params, img_disp);

  //double sigma = 0.7;
  //cv::GaussianBlur(img_left, img_left, cv::Size(3,3), sigma);
  //cv::GaussianBlur(img_right, img_right, cv::Size(3,3), sigma);

  // opencv
  //sgbm(img_left, img_right, img_disp);
  //img_disp.convertTo(img_disp_save, CV_8U, 1.0/16.0);
  //img_disp.convertTo(mat_disp_save, CV_32F, 1.0/16.0);
  //img_disp.convertTo(img_disp, CV_8U, (255.0/sgbm.numberOfDisparities) * (1.0/16.0));

  recon::StereoSGMParams sgm_params;
  sgm_params.disp_range = number_of_disparities;
  sgm_params.window_sz = sad_window_size;
  sgm_params.penalty1 = P1;
  sgm_params.penalty2 = P2;
  recon::StereoSGM sgm(sgm_params);
  sgm.compute(img_left, img_right, mat_disp);
  mat_disp.convertTo(img_disp, CV_8U);

  cv::equalizeHist(img_disp, img_disp);
  cv::imshow("Disparity image", img_disp);

  //cv::imwrite("disp_img.png", img_disp);
}

void fineTuneSGM(const vector<string>& imagelist, const string& source_folder)
{
  // http://www.cvlibs.net/datasets/kitti/eval_stereo_flow_detail.php?benchmark=stereo&error=3&eval=all&result=3ae300a3a3b3ed3e48a63ecb665dffcc127cf8ab
  //sgbm.preFilterCap = 63;
  //sgbm.SADWindowSize = 3;       // 3, 5
  //int cn = 1;                   // image channel size
  ////sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
  //sgbm.P1 = 4*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;       // 50
  //sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;      // 800
  //sgbm.minDisparity = 0;
  //sgbm.numberOfDisparities = 128;     // 128, 256
  //sgbm.uniquenessRatio = 10;          // 10, 0
  //sgbm.speckleWindowSize = 100;       // 100
  //sgbm.speckleRange = 32;             // 32
  //sgbm.disp12MaxDiff = 1;
  //sgbm.fullDP = 1;

  // better for KITTI
  // http://www.cvlibs.net/datasets/kitti/eval_stereo_flow_detail.php?benchmark=stereo&error=3&eval=all&result=0d8c42f3fb90ff87bf89c1f8f7977cf309a0f49d
  //sgbm.preFilterCap = 15;
  //sgbm.SADWindowSize = 5;
  //sgbm.P1 = 50;
  //sgbm.P2 = 800;
  //sgbm.minDisparity = 0;
  //sgbm.numberOfDisparities = 256;
  //sgbm.uniquenessRatio = 0;
  //sgbm.speckleWindowSize = 100;
  //sgbm.speckleRange = 32;
  //sgbm.disp12MaxDiff = 1;
  //sgbm.fullDP = 1;
  updateParameters();

  cv::namedWindow("Disparity image");
  cv::namedWindow("Left image");
  cv::namedWindow("Right image");
  cv::namedWindow("Parameters");
  cv::createTrackbar("Frame number", "Parameters", &frame_num, frame_num_max, on_trackbar);
  cv::createTrackbar("preFilterCap", "Parameters", &pre_filter_cap, pre_filter_cap_max, on_trackbar);
  cv::createTrackbar("SADWindowSize", "Parameters", &sad_window_size, sad_window_size_max, on_trackbar);
  cv::createTrackbar("P1", "Parameters", &P1, P1_max, on_trackbar);
  cv::createTrackbar("P2", "Parameters", &P2, P2_max, on_trackbar);
  cv::createTrackbar("minDisparity", "Parameters", &min_disparity, min_disparity_max, on_trackbar);
  cv::createTrackbar("numberOfDisparities", "Parameters", &number_of_disparities, number_of_disparities_max, on_trackbar);
  cv::createTrackbar("uniquenessRatio", "Parameters", &uniqueness_ratio, uniqueness_ratio_max, on_trackbar);
  cv::createTrackbar("speckleWindowSize", "Parameters", &speckle_window_size, speckle_window_size_max, on_trackbar);
  cv::createTrackbar("speckleRange", "Parameters", &speckle_range, speckle_range_max, on_trackbar);
  cv::createTrackbar("disp12MaxDiff", "Parameters", &disp12_max_diff, disp12_max_diff_max, on_trackbar);
  cv::createTrackbar("fullDP", "Parameters", &full_dp, full_dp_max, on_trackbar);
  cv::imshow("Parameters", cv::Mat::zeros(1, 1400, CV_8U));

  while(true) {
    cout << source_folder + imagelist[frame_num*2] << "\n";
    img_left = cv::imread(source_folder + imagelist[frame_num*2], CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(source_folder + imagelist[(frame_num*2)+1], CV_LOAD_IMAGE_GRAYSCALE);
    //img_left = cv::imread("/home/kivan/Projects/datasets/Middlebury/im2.png", CV_LOAD_IMAGE_GRAYSCALE);
    //img_right = cv::imread("/home/kivan/Projects/datasets/Middlebury/im6.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::imshow("Left image", img_left);
    cv::imshow("Right image", img_right);

    //cv::Mat img_left_eq, img_right_eq;
    //cv::equalizeHist(img_left, img_left_eq);
    //cv::equalizeHist(img_right, img_right_eq);
    //img_left = img_left_eq;
    //img_right = img_right_eq;
    //cv::imshow("left_eq", img_left_eq);
    //cv::imshow("right_eq", img_right_eq);

    computeSGM();
    while (cv::waitKey(0) != 10);
  }
  cv::waitKey(0);
}

int main(int argc, char** argv)
{
  string config_file;
  string imagelistfn;
  string source_folder;

  try {
    po::options_description generic("Generic options");
    generic.add_options()
      ("help", "produce help message")
      ("config,c", po::value<string>(&config_file)->default_value("config.txt"), "config filename")
      ;
    po::options_description config("Config file options");
    config.add_options()
      ("source_folder,s", po::value<string>(&source_folder), "folder with source")
      ("imglist,l", po::value<string>(&imagelistfn), "file with image list")
      ;

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config);

    po::options_description config_file_options;
    config_file_options.add(config);
    po::variables_map vm;
    //po::store(po::command_line_parser(argc, argv).options(cmdline_options).allow_unregistered().run(), vm);
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
      po::store(parse_config_file(ifs, config_file_options, true), vm);
      notify(vm);
    }
  }
  catch(std::exception& e) {
    cout << e.what() << "\n";
    return 1;
  }
  
  std::vector<std::string> imagelist;
  bool ok = core::FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
  {
    cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
    return -1;
  }

  fineTuneSGM(imagelist, source_folder);

  return 0;
}
