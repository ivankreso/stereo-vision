#include "format_helper.h"

#include <iostream>
#include <iomanip>

#include "math_helper.h"

namespace core {

using namespace std;
using namespace cv;

namespace {
void GetCalibParams(const cv::Mat& P_left, const cv::Mat& P_right, double* calib) {
  //std::cout << P_left << "\n" << P_right;
  calib[0] = P_left.at<double>(0,0);
  calib[1] = P_left.at<double>(1,1);
  calib[2] = P_left.at<double>(0,2);
  calib[3] = P_left.at<double>(1,2);
  //calib[4] = std::abs(P_left.at<double>(0,3) - P_right.at<double>(0,3)) / calib[0];
  double x_diff = P_left.at<double>(0,3) - P_right.at<double>(0,3);
  double y_diff = P_left.at<double>(1,3) - P_right.at<double>(1,3);
  double z_diff = P_left.at<double>(2,3) - P_right.at<double>(2,3);
  calib[4] = std::sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff) / calib[0];
  for (int i = 0; i < 5; i++)
    std::cout << calib[i] << "\n";
}
}

void FormatHelper::ReadCalibKitti(const std::string path, double* grey_cam, double* color_cam)
{
  std::ifstream calib_file(path);
  std::vector<cv::Mat> P;
  P.resize(4);

  std::string word;
  for (int i = 0; i < 4; i++) {
    P[i].create(3, 4, CV_64F);
    calib_file >> word;
    for (int j = 0; j < 12; j++) {
      calib_file >> word;
      int row = j / 4;
      int col = j % 4;
      P[i].at<double>(row,col) = std::stod(word);
    }
    //std::cout << P[i] << "\n";
  }
  std::cout << "\nCalib mono:\n";
  GetCalibParams(P[0], P[1], grey_cam);
  std::cout << "\nCalib color:\n";
  GetCalibParams(P[2], P[3], color_cam);
}

// reads next matrix in a row
void FormatHelper::ReadNextRtMatrix(std::ifstream& file, cv::Mat& Rt)
{
   std::string line;
   getline(file, line);
   stringstream stream(line);
   string val;
   //cout << "LINE: " << line << endl;
   Rt = Mat::eye(4, 4, CV_64F);
   for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 4; j++) {
         stream >> val;
         Rt.at<double>(i,j) = std::stod(val);
         //cout << "READ: " << motion_params[i] << endl << endl;
      }
   }
}

void FormatHelper::Read2FrameMotionFromAccCameraMotion(const std::string filename, const int num_of_motions,
                                                       std::vector<cv::Mat>& world_motion,
                                                       std::vector<cv::Mat>& camera_motion) {
  // wrt - world motion
  // crt - camera motion
  std::ifstream file(filename);
  cv::Mat wrt, wrt_prev, wrt_curr;
  // read first (usually identety) matrix
  core::FormatHelper::ReadNextRtMatrix(file, wrt_curr);
  core::MathHelper::invTrans(wrt_curr, wrt_prev);

  for (int i = 0; i < num_of_motions; i++) {
    core::FormatHelper::ReadNextRtMatrix(file, wrt_curr);
    cv::Mat crt = wrt_prev * wrt_curr;
    core::MathHelper::invTrans(wrt_curr, wrt_prev);
    camera_motion.push_back(crt.clone());
    core::MathHelper::invTrans(crt, wrt);
    world_motion.push_back(wrt.clone());
    //std::cout << crt << "\n\n";
    //std::cout << wrt << "\n\n";
  }
}

void FormatHelper::WriteMatRt(const cv::Mat& Rt, std::ofstream& fp) {
   //std::setprecision(6);
   for(int i = 0; i < (Rt.rows-1); i++) {
      for(int j = 0; j < Rt.cols; j++) {
        double val = Rt.at<double>(i,j);
        fp << std::scientific << val << " ";
      }
   }
   fp << endl;
}

void FormatHelper::WriteMotionToFile(const Eigen::Matrix4d& Rt, std::ofstream& file) {
  //std::setprecision(6);
  for(int i = 0; i < Rt.rows() - 1; i++)
    for(int j = 0; j < Rt.cols(); j++)
      file << std::scientific << Rt(i,j) << " ";
  file << std::endl;
}

void FormatHelper::WriteMatRt(const cv::Mat& Rt, std::ofstream& fp, bool convert_to_cm)
{
   //std::setprecision(6);
   for(int i = 0; i < (Rt.rows-1); i++) {
      for(int j = 0; j < Rt.cols; j++) {
        double val = Rt.at<double>(i,j);
        if(convert_to_cm && j == 3)
          val *= 100.0;
        fp << std::scientific << val << " ";
      }
   }
   fp << endl;
}

// [fx fy cx cy tx=baseline]
void FormatHelper::readCameraParams(const std::string& filepath, double (&cam_params)[5])
{
   ifstream file(filepath);
   std::string line;
   getline(file, line);
   stringstream stream(line);
   string val;
   cout << "--using camera params: ";
   for(int i = 0; i < 5; i++) {
      stream >> val;
      cam_params[i] = std::stod(val);
      cout << cam_params[i] << "  ";
   }
   cout << endl;
   file.close();
}

void FormatHelper::readGpsPoints(string filename, vector<Mat> points)
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

void FormatHelper::getCalibParams(std::string& intrinsic_filename, std::string& extrinsic_filename, cv::Mat& P_left,
                    cv::Mat& P_right, cv::Mat& Q, cv::Mat& C_left, cv::Mat& D_left, cv::Mat& C_right, cv::Mat& D_right)
{
   // reading intrinsic parameters
   //FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
   FileStorage fs(intrinsic_filename, FileStorage::READ);
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

   // TODO - read this also from file
   //cv::Size imageSize(CALIB_WIDTH, CALIB_HEIGHT);
   //stereoRectify(C_left, D_left, C_right, D_right, imageSize, R, T, R1, R2, P_left, P_right, Q, CALIB_ZERO_DISPARITY, 
   //      -1, imageSize);

   return;
}

bool FormatHelper::readStringList(const std::string& filename, std::vector<std::string>& l)
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

}
