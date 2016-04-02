#ifndef OPTIMIZATION_BUNDLE_ADJUSTMENT_SOLVER_H_
#define OPTIMIZATION_BUNDLE_ADJUSTMENT_SOLVER_H_

#include <vector>
#include <cmath>
#include <array>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "../../core/types.h"

namespace optim {

struct TrackData
{
  int dist_from_cframe;
  std::vector<core::Point> left_tracks;
  std::vector<core::Point> right_tracks;
};

class BundleAdjustmentSolver
{
 public:
  BundleAdjustmentSolver(const std::string loss_type, const std::vector<double>& params,
                         bool use_weighting);
  void SetCameraParams(const Eigen::VectorXd& camera_params) {
    camera_params_ = camera_params;
  }

  void AddCameraMotion(const Eigen::Matrix4d& Rt);
  void AddTrackData(const TrackData& data);
  bool Solve();
  // get camera extrinsic transform from cam local coord to world coord (1-frame coord)
  Eigen::Matrix4d GetCameraMotion(int camera_index) const;

 private:
  ceres::Problem ceres_problem_;  
  ceres::LossFunction* loss_function_;
  Eigen::VectorXd camera_params_;
  std::vector<std::array<double,3>> translation_;
  std::vector<std::array<double,4>> rotation_;
  double *trans_params_, *rot_params_;
  double use_weighting_;

  std::vector<cv::Mat> pts3d_;
  std::vector<double> weights_;
  std::map<std::pair<int,int>, std::pair<core::Point,core::Point>> projs_;
  std::map<std::pair<int,int>, core::Point> mono_projs_;
};

namespace
{

template <typename T>
void Transform3DPoint(
  const T* const cam_trans,
  const T* const cam_rot,
  const T* const pt3d,
  T* trans_pt3d)
{
  // Apply the angle-axis camera rotation
  //ceres::AngleAxisRotatePoint(cam_rot, pt3d, trans_pt3d);
  ceres::UnitQuaternionRotatePoint(cam_rot, pt3d, trans_pt3d);
  // Apply the camera translation
  trans_pt3d[0] += cam_trans[0];
  trans_pt3d[1] += cam_trans[1];
  trans_pt3d[2] += cam_trans[2];
}

//template <typename T>
//void ComputeStereoResiduals(T* pos_proj,
//                            const Eigen::VectorXd& cam_intr,
//                            const double* const left_point,
//                            const double* const right_point,
//                            T* out_residuals)
//{
//
//  T f = T(cam_intr[0]);
//  T cx = T(cam_intr[2]);
//  T cy = T(cam_intr[3]);
//  T b = T(cam_intr[4]);
//
//  // Transform the point from homogeneous to euclidean
//  T xe = pos_proj[0] / pos_proj[2];   // x / z
//  T ye = pos_proj[1] / pos_proj[2];   // y / z
//
//  //// Apply the focal length
//  //const T& focal = cam_f[0];
//  T predicted_x_left = f * xe + cx;
//  T predicted_y_left = f * ye + cy;
//  //std::cout << predicted_x_left << " -- " << predicted_y_left << "\n";
//  //std::cout << cam_intr << "\n";
//
//  // now for right camera
//  // first move point in right cam coord system
//  //pos_proj[0] -= b; -- this could lead to a bug
//  xe = (pos_proj[0] - b) / pos_proj[2];   // x / z
//  ye = pos_proj[1] / pos_proj[2];   // y / z
//  T predicted_x_right = f * xe + cx;
//  T predicted_y_right = f * ye + cy;
//
//  out_residuals[0] = predicted_x_left - T(left_point[0]);
//  out_residuals[1] = predicted_y_left - T(left_point[1]);
//  out_residuals[2] = predicted_x_right - T(right_point[0]);
//  out_residuals[3] = predicted_y_right - T(right_point[1]);
//}
template <typename T>
void ComputeStereoResiduals(T* pos_proj,
                            const Eigen::VectorXd& cam_intr,
                            const double* const left_point,
                            const double* const right_point,
                            const bool use_weighting,
                            const double weight,
                            T* out_residuals)
{

  T f = T(cam_intr[0]);
  T cx = T(cam_intr[2]);
  T cy = T(cam_intr[3]);
  T b = T(cam_intr[4]);

  // Transform the point from homogeneous to euclidean
  T xe = pos_proj[0] / pos_proj[2];   // x / z
  T ye = pos_proj[1] / pos_proj[2];   // y / z

  //// Apply the focal length
  //const T& focal = cam_f[0];
  T predicted_x_left = f * xe + cx;
  T predicted_y_left = f * ye + cy;
  //std::cout << predicted_x_left << " -- " << predicted_y_left << "\n";
  //std::cout << cam_intr << "\n";

  // now for right camera
  // first move point in right cam coord system
  //pos_proj[0] -= b; -- this could lead to a bug
  xe = (pos_proj[0] - b) / pos_proj[2];   // x / z
  ye = pos_proj[1] / pos_proj[2];   // y / z
  T predicted_x_right = f * xe + cx;
  T predicted_y_right = f * ye + cy;

  if (!use_weighting) {
    out_residuals[0] = predicted_x_left - T(left_point[0]);
    out_residuals[1] = predicted_y_left - T(left_point[1]);
    out_residuals[2] = predicted_x_right - T(right_point[0]);
    out_residuals[3] = predicted_y_right - T(right_point[1]);
  }
  else {
    T w = T(weight);
    out_residuals[0] = w * (predicted_x_left - T(left_point[0]));
    out_residuals[1] = w * (predicted_y_left - T(left_point[1]));
    out_residuals[2] = w * (predicted_x_right - T(right_point[0]));
    out_residuals[3] = w * (predicted_y_right - T(right_point[1]));
  }
}

struct ReprojErrorStereo
{
  //ReprojErrorStereo(const Eigen::Vector4d& pt3d, const core::Point& left_pt, const core::Point& right_pt,
  //                  const Eigen::VectorXd& cam_intr) : cam_intr_(cam_intr)
  //{
  //  left_pt_[0] = left_pt.x_;
  //  left_pt_[1] = left_pt.y_;
  //  right_pt_[0] = right_pt.x_;
  //  right_pt_[1] = right_pt.y_;

  //  pt3d_[0] = pt3d[0];
  //  pt3d_[1] = pt3d[1];
  //  pt3d_[2] = pt3d[2];
  //  
  //  weight = -1.0;
  //}
  ReprojErrorStereo(const Eigen::Vector4d& pt3d, const core::Point& left_pt, const core::Point& right_pt,
                    const Eigen::VectorXd& cam_intr, bool use_weighting) : cam_intr_(cam_intr)
  {
    left_pt_[0] = left_pt.x_;
    left_pt_[1] = left_pt.y_;
    right_pt_[0] = right_pt.x_;
    right_pt_[1] = right_pt.y_;

    pt3d_[0] = pt3d[0];
    pt3d_[1] = pt3d[1];
    pt3d_[2] = pt3d[2];
    
    if (use_weighting) {
      double cx = cam_intr[2];
      weight = 1.0 / (std::fabs(left_pt.x_ - cx)/std::fabs(cx) + 0.05);
    }
    else {
      weight = -1.0;
    }

    //for(int i = 0; i < 5; i++)
    //  cam_intr_[i] = cam_intr(i);    // 5 params - fx fy cx cy b
  }

  template <typename T>
  bool operator()(const T* const cam_trans, const T* const cam_rot, T* out_residuals) const
  {
    T pt3d0[3];
    //WrapPoint3D(&pt3d0[0]);
    WrapPoint3D(pt3d0);
    T pt3d1[3];
    Transform3DPoint(cam_trans, cam_rot, pt3d0, pt3d1);
    ComputeStereoResiduals(pt3d1, cam_intr_, left_pt_, right_pt_, true, weight, out_residuals);
    return true;
  }

  template <typename T>
  bool operator()(const T* const cam_trans1, const T* const cam_rot1,
                  const T* const cam_trans2, const T* const cam_rot2, T* out_residuals) const
  {
    T pt3d0[3];
    //WrapPoint3D(&pt3d0[0]);
    WrapPoint3D(pt3d0);
    T pt3d1[3];
    Transform3DPoint(cam_trans1, cam_rot1, pt3d0, pt3d1);
    T pt3d2[3];
    Transform3DPoint(cam_trans2, cam_rot2, pt3d1, pt3d2);
    ComputeStereoResiduals(pt3d2, cam_intr_, left_pt_, right_pt_, true, weight, out_residuals);
    return true;
  }

  template <typename T>
  bool operator()(const T* const cam_trans1, const T* const cam_rot1,
                  const T* const cam_trans2, const T* const cam_rot2,
                  const T* const cam_trans3, const T* const cam_rot3, T* out_residuals) const
  {
    T pt3d0[3];
    //WrapPoint3D(&pt3d0[0]);
    WrapPoint3D(pt3d0);
    T pt3d1[3];
    Transform3DPoint(cam_trans1, cam_rot1, pt3d0, pt3d1);
    T pt3d2[3];
    Transform3DPoint(cam_trans2, cam_rot2, pt3d1, pt3d2);
    T pt3d3[3];
    Transform3DPoint(cam_trans3, cam_rot3, pt3d2, pt3d3);
    ComputeStereoResiduals(pt3d3, cam_intr_, left_pt_, right_pt_, true, weight, out_residuals);
    return true;
  }

  template <typename T>
  bool operator()(const T* const cam_trans1, const T* const cam_rot1,
                  const T* const cam_trans2, const T* const cam_rot2,
                  const T* const cam_trans3, const T* const cam_rot3,
                  const T* const cam_trans4, const T* const cam_rot4,
                  T* out_residuals) const
  {
    T pt3d0[3];
    //WrapPoint3D(&pt3d0[0]);
    WrapPoint3D(pt3d0);
    T pt3d1[3];
    Transform3DPoint(cam_trans1, cam_rot1, pt3d0, pt3d1);
    T pt3d2[3];
    Transform3DPoint(cam_trans2, cam_rot2, pt3d1, pt3d2);
    T pt3d3[3];
    Transform3DPoint(cam_trans3, cam_rot3, pt3d2, pt3d3);
    T pt3d4[3];
    Transform3DPoint(cam_trans4, cam_rot4, pt3d3, pt3d4);
    ComputeStereoResiduals(pt3d4, cam_intr_, left_pt_, right_pt_, true, weight, out_residuals);
    return true;
  }

  template <typename T>
  bool operator()(const T* const cam_trans1, const T* const cam_rot1,
                  const T* const cam_trans2, const T* const cam_rot2,
                  const T* const cam_trans3, const T* const cam_rot3,
                  const T* const cam_trans4, const T* const cam_rot4,
                  const T* const cam_trans5, const T* const cam_rot5,
                  T* out_residuals) const
  {
    T pt3d0[3];
    //WrapPoint3D(&pt3d0[0]);
    WrapPoint3D(pt3d0);
    T pt3d1[3];
    Transform3DPoint(cam_trans1, cam_rot1, pt3d0, pt3d1);
    T pt3d2[3];
    Transform3DPoint(cam_trans2, cam_rot2, pt3d1, pt3d2);
    T pt3d3[3];
    Transform3DPoint(cam_trans3, cam_rot3, pt3d2, pt3d3);
    T pt3d4[3];
    Transform3DPoint(cam_trans4, cam_rot4, pt3d3, pt3d4);
    T pt3d5[3];
    Transform3DPoint(cam_trans5, cam_rot5, pt3d4, pt3d5);
    ComputeStereoResiduals(pt3d5, cam_intr_, left_pt_, right_pt_, true, weight, out_residuals);
    return true;
  }


  template <typename T>
  void WrapPoint3D(T (&pt3d)[3]) const {
    for (int i = 0; i < 3; i++)
      pt3d[i] = T(pt3d_[i]);
  }

  double left_pt_[2];                 // The left 2D observation
  double right_pt_[2];                // The right 2D observation
  double pt3d_[3];                    // The 3D point in world coords (first frame coord system)
  // TODO
  //const core::Point& left_pt_;                 // The left 2D observation
  //const core::Point& right_pt_;                // The right 2D observation
  //Eigen::Vector4d& pt3d_;                    // The 3D point in world coords (first frame coord system)

  double weight;
  const Eigen::VectorXd& cam_intr_;   // Instrinsic params are fixed
  int num_motions_;
};

} // end unnamed namespace

} // end namespace optim

#endif
