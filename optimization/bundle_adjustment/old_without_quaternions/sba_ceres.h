#ifndef OPTIMIZATION_SBA_CERES_
#define OPTIMIZATION_SBA_CERES_

#include "sba_base.h"
#include "bal_problem_stereo.h"

#include <map>
#include <vector>
#include <cmath>
#include <array>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace optim {

class SBAceres : public SBAbase
{
public:
  SBAceres(BALProblemStereo* problem, BAType ba_type);
  virtual void setCameraIntrinsics(const cv::Mat& cam_params);
  virtual void addCameraMotion(const cv::Mat& Rt);
  virtual void addPoint(const cv::Mat& pt, double weight);
  virtual void addMonoProj(int ci, int pi, const core::Point& proj);
  virtual void addStereoProj(int ci, int pi, const core::Point& proj_left, const core::Point& proj_right);
  virtual void runSBA();

  // get camera extrinsic transform from cam local coord to world coord (1-frame coord)
  virtual cv::Mat getCameraRt(int ci) const;

protected:
  void setupSBAproblem();

  BALProblemStereo* sba_problem_;
  BAType ba_type_;
  std::vector<cv::Mat> pts3d_;
  std::vector<double> weights_;
  std::vector<std::array<double,3>> cam_extr_trans_;
  std::vector<std::array<double,3>> cam_extr_rot_;
  cv::Mat cam_intr_;
  std::map<std::pair<int,int>, std::pair<core::Point,core::Point>> projs_;
  std::map<std::pair<int,int>, core::Point> mono_projs_;
};

namespace
{
/**
 * @brief Compute the residual error after reprojection.
 * residual = observed - euclidean( f * [R|t] X)
 *
 * @warning Principal point is assumed being applied on observed points.
 *
 * @param[in] cam_R Angle-axis camera rotation
 * @param[in] cam_t (x, y, z) Camera translation
 * @param[in] cam_f (f) Focal length
 * @param[in] pos_3dpoint The observed 3D point
 * @param[in] pos_2dpoint The image plane observation
 * @param[out] out_residuals The residuals along the x and y axis
 */

template <typename T>
void ComputeStereoResidualStructureAndMotion(
  const T* const cam_R,
  const T* const cam_t,
  const T* const pt3d,
  const double weight,
  const double* const cam_intr,
  const double* const left_2dpoint,
  const double* const right_2dpoint,
  T* out_residuals)
{
  T pos_proj[3];
  // Apply the angle-axis camera rotation
  ceres::AngleAxisRotatePoint(cam_R, pt3d, pos_proj);

  // Apply the camera translation
  pos_proj[0] += cam_t[0];
  pos_proj[1] += cam_t[1];
  pos_proj[2] += cam_t[2];

  // camera matrix
  //Mat3 K = Mat3::Identity();
  //K(0,0) = cam_intr[0];
  //K(1,1) = cam_intr[1];
  //K(0,2) = cam_intr[2];
  //K(1,2) = cam_intr[3];
  //std::cout << K << "\n";
  //pos_proj = K * pos_proj;

  T fx = T(cam_intr[0]);
  T fy = T(cam_intr[1]);
  T cx = T(cam_intr[2]);
  T cy = T(cam_intr[3]);
  T b = T(cam_intr[4]);

  // Transform the point from homogeneous to euclidean
  T xe = pos_proj[0] / pos_proj[2];   // x / z
  T ye = pos_proj[1] / pos_proj[2];   // y / z

  //// Apply the focal length
  //const T& focal = cam_f[0];
  T predicted_x_left = fx * xe + cx;
  T predicted_y_left = fy * ye + cy;
  //std::cout << predicted_x_left << " -- " << predicted_y_left << "\n";
  //std::cout << cam_intr << "\n";

  // now for right camera
  // first move point in right cam coord system
  pos_proj[0] -= b;
  xe = pos_proj[0] / pos_proj[2];   // x / z
  ye = pos_proj[1] / pos_proj[2];   // y / z
  T predicted_x_right = fx * xe + cx;
  T predicted_y_right = fy * ye + cy;

  // without weighting
  //// Compute and return the error is the difference between the predicted and observed position
  //out_residuals[0] = predicted_x_left - T(left_2dpoint[0]);
  //out_residuals[1] = predicted_y_left - T(left_2dpoint[1]);
  //out_residuals[2] = predicted_x_right - T(right_2dpoint[0]);
  //out_residuals[3] = predicted_y_right - T(right_2dpoint[1]);

  //double cp_x = cam_intr[2];
  //double w = 1.0/(std::fabs(left_2dpoint[0] - cp_x)/std::fabs(cp_x) + 0.05);
  T w = T(weight);
  // Compute and return the error is the difference between the predicted and observed position
  out_residuals[0] = w * (predicted_x_left - T(left_2dpoint[0]));
  out_residuals[1] = w * (predicted_y_left - T(left_2dpoint[1]));
  out_residuals[2] = w * (predicted_x_right - T(right_2dpoint[0]));
  out_residuals[3] = w * (predicted_y_right - T(right_2dpoint[1]));
}


template <typename T>
void ComputeStereoResidualMotion(
  const T* const cam_R,
  const T* const cam_t,
  const double* const pt3d,
  const double weight,
  const double* const cam_intr,
  const double* const left_2dpoint,
  const double* const right_2dpoint,
  T* out_residuals)
{
  T pos_3dpoint[3];
  pos_3dpoint[0] = T(pt3d[0]);
  pos_3dpoint[1] = T(pt3d[1]);
  pos_3dpoint[2] = T(pt3d[2]);
  T pos_proj[3];

  // Apply the angle-axis camera rotation
  ceres::AngleAxisRotatePoint(cam_R, pos_3dpoint, pos_proj);
  // Apply the camera translation
  pos_proj[0] += cam_t[0];
  pos_proj[1] += cam_t[1];
  pos_proj[2] += cam_t[2];

  T fx = T(cam_intr[0]);
  T fy = T(cam_intr[1]);
  T cx = T(cam_intr[2]);
  T cy = T(cam_intr[3]);
  T b = T(cam_intr[4]);

  // Transform the point from homogeneous to euclidean
  T xe = pos_proj[0] / pos_proj[2];   // x / z
  T ye = pos_proj[1] / pos_proj[2];   // y / z

  //// Apply the focal length
  //const T& focal = cam_f[0];
  T predicted_x_left = fx * xe + cx;
  T predicted_y_left = fy * ye + cy;
  //std::cout << predicted_x_left << " -- " << predicted_y_left << "\n";
  //std::cout << cam_intr << "\n";

  // now for right camera
  // first move point in right cam coord system
  pos_proj[0] -= b;
  xe = pos_proj[0] / pos_proj[2];   // x / z
  ye = pos_proj[1] / pos_proj[2];   // y / z
  T predicted_x_right = fx * xe + cx;
  T predicted_y_right = fy * ye + cy;

  // without weighting
  //// Compute and return the error is the difference between the predicted and observed position
  //out_residuals[0] = predicted_x_left - T(left_2dpoint[0]);
  //out_residuals[1] = predicted_y_left - T(left_2dpoint[1]);
  //out_residuals[2] = predicted_x_right - T(right_2dpoint[0]);
  //out_residuals[3] = predicted_y_right - T(right_2dpoint[1]);

  //double cp_x = cam_intr[2];
  //double w = 1.0/(std::fabs(left_2dpoint[0] - cp_x)/std::fabs(cp_x) + 0.05);
  T w = T(weight); // Compute and return the error is the difference between the predicted and observed position
  out_residuals[0] = w * (predicted_x_left - T(left_2dpoint[0]));
  out_residuals[1] = w * (predicted_y_left - T(left_2dpoint[1]));
  out_residuals[2] = w * (predicted_x_right - T(right_2dpoint[0]));
  out_residuals[3] = w * (predicted_y_right - T(right_2dpoint[1]));
}

template <typename T>
void ComputeStereoResidualMotionBetter(
  const T* const cam_R,
  const T* const cam_t,
  const double* const pt3d,
  const int start_frame,
  const int end_frame,
  const double weight,
  const double* const cam_intr,
  const double* const left_2dpoint,
  const double* const right_2dpoint,
  T* out_residuals)
{
  //n = 3d point frame = pt3d_frame
  // if we keep multi-frame motions in Rt
  //Rt_real = Rt_inv_0_to_n * Rt;

  // do Rt params need to be independent of each other because of threads?
  // or if we keep 2-frame motions
  //Rt_real = Rt[start_frame] * ... * Rt[end_frame-1] * Rt

  // TODO: do this
  //Rt_before = Rt[start_frame] * ... * Rt[end_frame-1]
  //pt3d = Rt_before * pt3d;

  T pos_3dpoint[3];
  pos_3dpoint[0] = T(pt3d[0]);
  pos_3dpoint[1] = T(pt3d[1]);
  pos_3dpoint[2] = T(pt3d[2]);
  T pos_proj[3];

  // Apply the angle-axis camera rotation
  ceres::AngleAxisRotatePoint(cam_R, pos_3dpoint, pos_proj);
  // Apply the camera translation
  pos_proj[0] += cam_t[0];
  pos_proj[1] += cam_t[1];
  pos_proj[2] += cam_t[2];

  T fx = T(cam_intr[0]);
  T fy = T(cam_intr[1]);
  T cx = T(cam_intr[2]);
  T cy = T(cam_intr[3]);
  T b = T(cam_intr[4]);

  // Transform the point from homogeneous to euclidean
  T xe = pos_proj[0] / pos_proj[2];   // x / z
  T ye = pos_proj[1] / pos_proj[2];   // y / z

  //// Apply the focal length
  //const T& focal = cam_f[0];
  T predicted_x_left = fx * xe + cx;
  T predicted_y_left = fy * ye + cy;
  //std::cout << predicted_x_left << " -- " << predicted_y_left << "\n";
  //std::cout << cam_intr << "\n";

  // now for right camera
  // first move point in right cam coord system
  pos_proj[0] -= b;
  xe = pos_proj[0] / pos_proj[2];   // x / z
  ye = pos_proj[1] / pos_proj[2];   // y / z
  T predicted_x_right = fx * xe + cx;
  T predicted_y_right = fy * ye + cy;

  // without weighting
  //// Compute and return the error is the difference between the predicted and observed position
  //out_residuals[0] = predicted_x_left - T(left_2dpoint[0]);
  //out_residuals[1] = predicted_y_left - T(left_2dpoint[1]);
  //out_residuals[2] = predicted_x_right - T(right_2dpoint[0]);
  //out_residuals[3] = predicted_y_right - T(right_2dpoint[1]);

  //double cp_x = cam_intr[2];
  //double w = 1.0/(std::fabs(left_2dpoint[0] - cp_x)/std::fabs(cp_x) + 0.05);
  T w = T(weight); // Compute and return the error is the difference between the predicted and observed position
  out_residuals[0] = w * (predicted_x_left - T(left_2dpoint[0]));
  out_residuals[1] = w * (predicted_y_left - T(left_2dpoint[1]));
  out_residuals[2] = w * (predicted_x_right - T(right_2dpoint[0]));
  out_residuals[3] = w * (predicted_y_right - T(right_2dpoint[1]));
}
}

struct ReprojectionErrorStereoStructureAndMotion
{
  ReprojectionErrorStereoStructureAndMotion(const double* const stereo_projs,
                                            const double* const cam_intr,
                                            const double weight)
  {
    left_proj[0] = stereo_projs[0];
    left_proj[1] = stereo_projs[1];
    right_proj[0] = stereo_projs[2];
    right_proj[1] = stereo_projs[3];
    this->weight = weight;
    for(int i = 0; i < 5; i++)
      cam_intr_[i] = cam_intr[i];    // 5 params - fx fy cx cy b
  }

  /**
   * @param[in] cam_Rtf: Camera parameterized using one block of 7 parameters [R;t;f]:
   *   - 3 for rotation(angle axis), 3 for translation, 1 for the focal length.
   * @param[out] out_residuals
   */
  template <typename T>
  bool operator()(
      const T* const cam_params, // [R;t]
      const T* const pt3d, // 3D world point
      T* out_residuals) const
  {
    ComputeStereoResidualStructureAndMotion(
        cam_params, // => cam_R - rotation in Euler angles
        &cam_params[3], // => cam_t
        pt3d,
        weight,
        cam_intr_, // => cam intrinsics
        left_proj, right_proj, out_residuals);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* create(const double* const stereo_projs,
                                     const double* const cam_intr,
                                     const double weight)
  {
    // 4 resudual errors for stereo
    // 6 params for motion
    // 3 params for 3D point
    return (new ceres::AutoDiffCostFunction<ReprojectionErrorStereoStructureAndMotion, 4, 6, 3>(
            new ReprojectionErrorStereoStructureAndMotion(stereo_projs, cam_intr, weight)));
  }

  double left_proj[2];      // The left 2D observation
  double right_proj[2];     // The right 2D observation
  double cam_intr_[5];      // Instrinsic params are fixed
  double weight;
};

struct ReprojectionErrorStereoMotion
{
  ReprojectionErrorStereoMotion(const double* const stereo_projs, const double* const pos_3dpoint,
                                const double* const cam_intr, const double weight)
  {
    left_proj[0] = stereo_projs[0];
    left_proj[1] = stereo_projs[1];
    right_proj[0] = stereo_projs[2];
    right_proj[1] = stereo_projs[3];
    this->weight = weight;

    m_pos_3dpoint[0] = pos_3dpoint[0];
    m_pos_3dpoint[1] = pos_3dpoint[1];
    m_pos_3dpoint[2] = pos_3dpoint[2];

    for(int i = 0; i < 5; i++)
      cam_intr_[i] = cam_intr[i];    // 5 params - fx fy cx cy b
  }

  /**
   * @param[in] cam_Rtf: Camera parameterized using one block of 7 parameters [R;t;f]:
   *   - 3 for rotation(angle axis), 3 for translation, 1 for the focal length.
   * @param[out] out_residuals
   */
  template <typename T>
  bool operator()(
      const T* const cam_params, // [R;t]
      T* out_residuals) const
  {
    ComputeStereoResidualMotion(
        cam_params, // => cam_R - rotation in Euler angles
        &cam_params[3], // => cam_t
        m_pos_3dpoint,
        weight,
        cam_intr_, // => cam intrinsics
        left_proj,
        right_proj,
        out_residuals);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* create(const double* const stereo_projs,
                                     const double* const pos_3dpoint,
                                     const double* const cam_intr,
                                     const double weight)
  {
    return (new ceres::AutoDiffCostFunction<ReprojectionErrorStereoMotion, 4, 6>(
            new ReprojectionErrorStereoMotion(stereo_projs, pos_3dpoint, cam_intr, weight)));
  }

  double left_proj[2];      // The left 2D observation
  double right_proj[2];     // The right 2D observation
  double m_pos_3dpoint[3];  // The 3D point in world coords (first frame coord system)
  double cam_intr_[5];      // Instrinsic params are fixed
  double weight;
};


//struct ReprojErrorStereoAnalyticJacobian
//{
//  ReprojErrorStereoAnalyticJacobian(const double* const stereo_projs, const double* const pos_3dpoint,
//                          const double* const cam_intr)
//  {
//    left_proj[0] = stereo_projs[0];
//    left_proj[1] = stereo_projs[1];
//    right_proj[0] = stereo_projs[2];
//    right_proj[1] = stereo_projs[3];
//
//    m_pos_3dpoint[0] = pos_3dpoint[0];
//    m_pos_3dpoint[1] = pos_3dpoint[1];
//    m_pos_3dpoint[2] = pos_3dpoint[2];
//
//    for(int i = 0; i < 5; i++)
//      cam_intr_[i] = cam_intr[i];    // 5 params - fx fy cx cy b
//  }
//
//  /**
//   * @param[in] cam_Rtf: Camera parameterized using one block of 7 parameters [R;t;f]:
//   *   - 3 for rotation(angle axis), 3 for translation, 1 for the focal length.
//   * @param[out] out_residuals
//   */
//  template <typename T>
//    bool operator()(
//        const T* const cam_params, // [R;t]
//        T* out_residuals) const
//    {
//      T pos_3dpoint[3];
//      pos_3dpoint[0] = T(m_pos_3dpoint[0]);
//      pos_3dpoint[1] = T(m_pos_3dpoint[1]);
//      pos_3dpoint[2] = T(m_pos_3dpoint[2]);
//
//      computeStereoResidual(
//          cam_params, // => cam_R - rotation in Euler angles
//          &cam_params[3], // => cam_t
//          pos_3dpoint,
//          cam_intr_, // => cam intrinsics
//          left_proj,
//          right_proj,
//          out_residuals);
//      return true;
//    }
//
//  // Factory to hide the construction of the CostFunction object from
//  // the client code.
//  static ceres::CostFunction* create(const double* const stereo_projs,
//                                     const double* const pos_3dpoint,
//                                     const double* const cam_intr)
//  {
//    return (new ceres::AutoDiffCostFunction<ReprojErrorStereoAnalyticJacobian, 4, 6>(
//            new ReprojErrorStereoAnalyticJacobian(stereo_projs, pos_3dpoint, cam_intr)));
//  }
//
//
//  double left_proj[2];      // The left 2D observation
//  double right_proj[2];     // The right 2D observation
//  double m_pos_3dpoint[3];  // The 3D point in world coords (first frame coord system)
//  double cam_intr_[5];      // Instrinsic params are fixed
//};




// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionErrorMono {
  SnavelyReprojectionErrorMono(double observed_x, double observed_y)
    : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
    bool operator()(const T* const camera,
        const T* const point,
        T* residuals) const {
      // camera[0,1,2] are the angle-axis rotation.
      T p[3];
      ceres::AngleAxisRotatePoint(camera, point, p);

      // camera[3,4,5] are the translation.
      p[0] += camera[3];
      p[1] += camera[4];
      p[2] += camera[5];

      // Compute the center of distortion. The sign change comes from
      // the camera model that Noah Snavely's Bundler assumes, whereby
      // the camera coordinate system has a negative z axis.
      T xp = - p[0] / p[2];
      T yp = - p[1] / p[2];

      // Apply second and fourth order radial distortion.
      const T& l1 = camera[7];
      const T& l2 = camera[8];
      T r2 = xp*xp + yp*yp;
      T distortion = T(1.0) + r2  * (l1 + l2  * r2);

      // Compute final projected point position.
      const T& focal = camera[6];
      T predicted_x = focal * distortion * xp;
      T predicted_y = focal * distortion * yp;

      // The error is the difference between the predicted and observed position.
      residuals[0] = predicted_x - T(observed_x);
      residuals[1] = predicted_y - T(observed_y);

      return true;
    }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
      const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionErrorMono, 2, 9, 3>(
          new SnavelyReprojectionErrorMono(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};


}

#endif
