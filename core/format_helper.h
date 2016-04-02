#ifndef CORE_FORMAT_HELPER_H_
#define CORE_FORMAT_HELPER_H_

#include <vector>
#include <fstream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

namespace core {

class FormatHelper
{
 public:
  static void ParseKITTIDatasetConfig(
      const std::string config_file, std::string& cam_params_file, std::string& dataset_name,
      std::string& left_folder, std::string& right_folder, std::string& gt_filepath,
      std::size_t& start_num, std::size_t& end_num, std::size_t& num_width);
  static void ReadCalibKitti(const std::string path, double* grey_cam, double* color_cam);
  static void ReadNextRtMatrix(std::ifstream& file, cv::Mat& Rt);
  static void ReadMotionFromFile(const std::string filename, std::vector<cv::Mat> motion_data);
  static void Read2FrameMotionFromAccCameraMotion(const std::string filename, const int num_of_motions,
                                                  std::vector<cv::Mat>& world_motion,
                                                  std::vector<cv::Mat>& camera_motion);
  static void WriteMatRt(const cv::Mat& Rt, std::ofstream& fp);
  static void WriteMotionToFile(const Eigen::Matrix4d& Rt, std::ofstream& file);
  static void WriteMatRt(const cv::Mat& Rt, std::ofstream& fp, bool convert_to_cm);

  static void readCameraParams(const std::string& file, double (&cam_params)[5]);
  static void readGpsPoints(std::string filename, std::vector<cv::Mat> points);
  static void getCalibParams(std::string& intrinsic_filename, std::string& extrinsic_filename, cv::Mat& P_left,
             cv::Mat& P_right, cv::Mat& Q, cv::Mat& C_left, cv::Mat& D_left, cv::Mat& C_right, cv::Mat& D_right);
  static bool readStringList(const std::string& filename, std::vector<std::string>& l);
  static void GetFilesInFolder(const std::string& dir_path, std::vector<std::string>& files,
                               bool get_full_path);
};

}

#endif
