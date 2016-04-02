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
using namespace cv;

#include "../../../core/format_helper.h"
#include "../../base/stereo_sgm.h"

void runDenseStereo(const vector<string>& imagelist, const string& source_folder, const string& output_folder)
{
  StereoSGBM sgbm;

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

  // worse setting
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

  // hand tuned
  sgbm.preFilterCap = 40;
  sgbm.SADWindowSize = 3;
  //sgbm.P1 = 40;
  sgbm.P1 = 40; // 10 good for 01 without histeq
  //sgbm.P2 = 2500; //1800;
  sgbm.P2 = 2500;
  sgbm.minDisparity = 0;
  sgbm.numberOfDisparities = 128; // or 256
  sgbm.uniquenessRatio = 0;
  sgbm.speckleWindowSize = 200;
  sgbm.speckleRange = 32;
  sgbm.disp12MaxDiff = 1;
  sgbm.fullDP = 1;

  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  Mat img_left, img_right;
  Mat img_disp, img_disp_subpixel, mat_disp;
  for(size_t i = 0; i < imagelist.size(); i+=2) {
    cout << source_folder + imagelist[i] << "\n";
    img_left = imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
    img_right = imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
    //cv::imshow("img_left", img_left);
    //cv::waitKey(0);

    double sigma = 0.7;
    cv::GaussianBlur(img_left, img_left, cv::Size(3,3), sigma);
    cv::GaussianBlur(img_right, img_right, cv::Size(3,3), sigma);

    recon::StereoSGMParams sgm_params;
    //sgm_params.disp_range = 128;
    //sgm_params.window_sz = 1;
    sgm_params.window_sz = 5;
    //sgm_params.penalty1 = 15;
    //sgm_params.penalty1 = 3;         // Daimler - 10
    //sgm_params.penalty2 = 100;
    sgm_params.penalty2 = 60;        // Daimler - 50
    // Axel tractor
    sgm_params.disp_range = 270;
    sgm_params.penalty1 = 7;

    recon::StereoSGM sgm(sgm_params);
    sgm.compute(img_left, img_right, mat_disp);
    mat_disp.convertTo(img_disp, CV_8U);

    //sgbm(img_left, img_right, img_disp);
    //img_disp.convertTo(img_disp_subpixel, CV_8U, 1.0/16.0);
    //img_disp.convertTo(mat_disp, CV_32F, 1.0/16.0);
    //img_disp.convertTo(img_disp, CV_8U, (255.0/sgbm.numberOfDisparities) * (1.0/16.0));

    string img_fname = output_folder + "img/disp_" + imagelist[i].substr(9, imagelist[i].size()-13) + ".png";
    string mat_fname = output_folder + "mat/disp_" + imagelist[i].substr(9, imagelist[i].size()-13) + ".yml";
    cout << "Saving: " << img_fname << "\n" << mat_fname << "\n----------------------\n";
    cv::equalizeHist(img_disp, img_disp);
    cv::imwrite(img_fname, img_disp, compression_params);
    cv::FileStorage mat_file(mat_fname, cv::FileStorage::WRITE);
    mat_file << "disparity_matrix" << mat_disp;

    //cout << img_disp << "\n\n";
    //imshow("disparity", img_disp);
    //waitKey(0);
  }
}

int main(int argc, char** argv)
{
  string config_file;
  string imagelistfn;
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
      ("source_folder,s", po::value<string>(&source_folder), "folder with source")
      ("disp_folder,d", po::value<string>(&output_folder), "folder for output")
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
    cout << "Configuring done, using:" << endl;

    if(vm.count("source_folder")) {
      cout << "Source folder: ";
      cout << source_folder << endl;
    }
    if(vm.count("disp_folder")) {
      cout << "Disparity folder: ";
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
  bool ok = core::FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
  {
    cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
    return -1;
  }

  runDenseStereo(imagelist, source_folder, output_folder);

  return 0;
}
