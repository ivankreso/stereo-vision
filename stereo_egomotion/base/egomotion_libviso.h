#ifndef STEREO_ODOMETRY_BASE_EGOMOTION_LIBVISO_
#define STEREO_ODOMETRY_BASE_EGOMOTION_LIBVISO_

#include "egomotion_base.h"
#include "../../tracker/stereo/stereo_tracker_base.h"

namespace egomotion
{

class EgomotionLibviso : public EgomotionBase {
 public:
  // camera parameters (all are mandatory / need to be supplied)
  struct calibration {
    double f;  // focal length (in pixels)
    double cu; // principal point (u-coordinate)
    double cv; // principal point (v-coordinate)
    calibration () {
      f  = 1;
      cu = 0;
      cv = 0;
    }
  };
  // general parameters
  struct parameters {
    calibration calib;            // camera calibration parameters
    // stereo-specific parameters (mandatory: base)
    double  base;             // baseline (meters)
    int32_t ransac_iters;     // number of RANSAC iterations
    double  inlier_threshold; // fundamental matrix inlier threshold
    bool    reweighting;      // lower border weights (more robust to calibration errors)
    parameters () {
      base             = 1.0;
      ransac_iters     = 200;
      inlier_threshold = 1.5;
      reweighting      = true;
    }
  };
  // structure for storing matches
  struct StereoMatch {
    float u1p, v1p; // u,v-coordinates in previous left  image
    float u2p, v2p; // u,v-coordinates in previous right image
    float u1c, v1c; // u,v-coordinates in current  left  image
    float u2c, v2c; // u,v-coordinates in current  right image
    StereoMatch() {}
    StereoMatch(float u1p, float v1p, float u2p, float v2p,
                float u1c, float v1c, float u2c, float v2c):
                u1p(u1p), v1p(v1p), u2p(u2p), v2p(v2p),
                u1c(u1c), v1c(v1c), u2c(u2c), v2c(v2c) {}
  };

  // constructor, takes as inpute a parameter structure
  EgomotionLibviso(parameters params);
  EgomotionLibviso(parameters params, std::string deformation_params_fname, int img_rows, int img_cols);

  //virtual bool GetMotion(track::StereoTrackerBase& tracker, cv::Mat& Rt);
  bool GetMotion(track::StereoTrackerBase& tracker, Eigen::Matrix4d& Rt) override;
  virtual std::vector<int> GetTrackerInliers() { return tracker_inliers_; }
  virtual std::vector<int> GetTrackerOutliers() { return tracker_outliers_; }
  std::vector<double> estimateMotion(std::vector<StereoMatch>& tracks);

  // ADDED
  void setLeftPrevImage(const cv::Mat& img) { img.copyTo(img_left_prev_); }

  // parameters
  parameters params_;

private:
  enum ResultState { UPDATED, FAILED, CONVERGED };  

  ResultState updateParameters(double* p_observe,
                               double* p_predict,
                               double* p_residual,
                               double* J,
                               std::vector<StereoMatch> &tracks,
                               std::vector<int32_t>& active,
                               std::vector<double>& tr,
                               double step_size, double eps);

  void computeObservations(double* p_observe,
                           std::vector<StereoMatch>& tracks,
                           std::vector<int> &active);

  void computeResidualsAndJacobian(double* p_observe,
                                   double* p_predict,
                                   double* p_residual,
                                   double* J,
                                   std::vector<double>& tr,
                                   std::vector<int>& active);


  std::vector<int> getInliers(double* p_observe,
                              double* p_predict,
                              double* p_residual,
                              double* J,
                              std::vector<StereoMatch> &tracks,
                              std::vector<double> &tr,
                              std::vector<int>& active);

  void updateTrackerInliers(const std::vector<int>& active_tracks);

  
  void GetTracksFromStereoTracker(track::StereoTrackerBase& tracker,
                                  std::vector<StereoMatch>& tracks,
                                  std::vector<int>& active_tracks);
  void GetPointCell(const core::Point& pt, int& row, int& col);

  void InterpolateBilinear(const cv::Mat& mat, const int row, const int col,
                           const double x, const double y, double& ival);
  void InterpolateLinear(const double val1, const double val2, const double x,
                         const double size, double& ival);

  void GetPointDeformation(const core::Point& pt, const cv::Mat& def_x,
                           const cv::Mat& def_y, double& dx, double& dy);

  void ComputeCellCenters();
  int GetBinNum(const core::Point& pt);

  double *X, *Y, *Z = nullptr;            // 3d points
  double *W = nullptr;

  std::vector<int> inliers_;            // ransac inlier set
  std::vector<int> tracker_inliers_;    // tracker inlier set
  std::vector<int> tracker_outliers_;

  cv::Mat left_dx_, left_dy_;
  cv::Mat right_dx_, right_dy_;
  cv::Mat cell_centers_x_;
  cv::Mat cell_centers_y_;
  bool use_deformation_map_ = false;

  cv::Mat weights_mat_;
  int img_rows_;
  int img_cols_;
  int cell_width_, cell_height_;
  //int bin_rows_, bin_cols_;

  // ADDED
  cv::Mat img_left_prev_;
};

inline
void EgomotionLibviso::InterpolateBilinear(const cv::Mat& mat, const int row, const int col,
                                               const double x, const double y, double& ival)
{
  double q1 = mat.at<double>(row, col);
  double q2 = mat.at<double>(row, col+1);
  double q3 = mat.at<double>(row+1, col);
  double q4 = mat.at<double>(row+1, col+1);

  double w = cell_width_;
  double h = cell_height_;
  double q12 = ((w-x) / w) * q1 + (x / w) * q2;
  double q34 = ((w-x) / w) * q3 + (x / w) * q4;
  ival = ((h-y) / h) * q12 + (y / h) * q34;
}

inline
void EgomotionLibviso::InterpolateLinear(const double val1, const double val2, const double x,
                                             const double size, double& ival)
{
  ival = ((size - x) / size) * val1 + (x / size) * val2;
}

inline
void EgomotionLibviso::GetPointCell(const core::Point& pt, int& row, int& col)
{
  col = pt.x_ / cell_width_;
  row = pt.y_ / cell_height_;
  //int bin_num = row * bin_cols_ + col;
}

}

#endif

