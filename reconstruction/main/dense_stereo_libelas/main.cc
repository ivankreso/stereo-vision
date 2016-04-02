#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;


#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;

#include "../../../core/format_helper.h"

#include "elas.h"

void runDenseStereo(const vector<string>& imagelist, const string& source_folder, const string& output_folder)
{
  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  Mat img_left, img_right;
  Mat img_disp, img_disp_save, mat_disp_save;
  for(size_t i = 0; i < imagelist.size(); i+=2) {
    cout << source_folder + imagelist[i] << "\n";
    img_left = imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
    img_right = imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::GaussianBlur(img_left, img_left, cv::Size(3,3), 0.7);
    cv::GaussianBlur(img_right, img_right, cv::Size(3,3), 0.7);
  
    int width = img_left.cols;
    int height = img_left.rows;
    const int32_t dims[3] = {width,height,width}; // bytes per line = width
    float* D1_data = new float[width*height];
    float* D2_data = new float[width*height];
    // process
    Elas::parameters param;
    param.add_corners = 1; 
    param.match_texture = 0;
    //param.postprocess_only_left = false;
    Elas elas(param);
    elas.process(img_left.data, img_right.data, D1_data, D2_data, dims);
    // find maximum disparity for scaling output disparity images to [0..255]
    float disp_max = 0;
    for (int32_t i=0; i<width*height; i++) {
      //std::cout << D1_data[i] << "\n";
      if(D1_data[i] > disp_max) disp_max = D1_data[i];
    }
    std::cout << "Max disp = " << disp_max << "\n";
    // copy float to uchar
    img_disp = cv::Mat::zeros(height, width, CV_8U);
    mat_disp_save = cv::Mat::zeros(height, width, CV_32F);
    for(int i = 0; i < height; i++) {
      for(int j = 0; j < width; j++) {
        //img_disp.at<uint8_t>(i,j) = (uint8_t)max(255.0*D1_data[i*width + j]/disp_max, 0.0);
        img_disp.at<uint8_t>(i,j) = (uint8_t)std::max(D1_data[i*width + j], 0.0f);
        mat_disp_save.at<float>(i,j) = D1_data[i*width + j];
      }
    }

    string img_fname = output_folder + "img/" + imagelist[i].substr(9, imagelist[i].size()-13) + "_disp.png";
    string mat_fname = output_folder + "mat/" + imagelist[i].substr(9, imagelist[i].size()-13) + "_disp.yml";
    cout << "Saving: " << img_fname << "\n" << mat_fname << "\n----------------------\n";
    cv::equalizeHist(img_disp, img_disp);
    cv::imwrite(img_fname, img_disp, compression_params);
    cv::FileStorage mat_file(mat_fname, cv::FileStorage::WRITE);
    mat_file << "disparity_matrix" << mat_disp_save;

    //cout << img_disp << "\n\n";
    //imshow("disparity", img_disp);
    //waitKey(0);
    delete[] D1_data;
    delete[] D2_data;
  }
}

int main(int argc, char** argv)
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
      ("camera_params,p", po::value<string>(&cam_params_file)->default_value("camera_params.txt"),
       "camera params file")
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

    if(vm.count("camera_params")) {
      cout << "Camera params: ";
      cout << cam_params_file << endl;
    }
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
