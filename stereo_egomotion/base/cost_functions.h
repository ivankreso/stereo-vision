// Copyright 2014, 2015 Ivan Kreso (kreso.ivan@gmail.com). All rights reserved.

#ifndef STEREO_EGOMOTION_BASE_COST_FUNCTIONS_H_
#define STEREO_EGOMOTION_BASE_COST_FUNCTIONS_H_

#include "../../core/types.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Templated pinhole camera model for used with Ceres.  The camera motion is
// parameterized using 7 parameters. 4 for rotation, 3 for translation.
struct ReprojectionErrorWithQuaternion {
  // (u, v): the position of the observation with respect to the image
  // center point.
  ReprojectionErrorWithQuaternion(const Eigen::Vector3d& point3d, const core::Point& obs_left,
                                  const core::Point& obs_right, const double* camera_intr)
      : observed_left(obs_left), observed_right(obs_right), pt3d(point3d), cam_intr(camera_intr) {}

  template <typename T>
  bool operator()(const T* const rotation,
                  const T* const translation,
                  T* residuals) const {
    T pt3d_prev[3];
    pt3d_prev[0] = T(pt3d[0]);
    pt3d_prev[1] = T(pt3d[1]);
    pt3d_prev[2] = T(pt3d[2]);

    T pt3d_curr[3];
    // Use a quaternion rotation that doesn't assume the quaternion is
    // normalized, since one of the ways to run the bundler is to let Ceres
    // optimize all 4 quaternion parameters unconstrained.
    //ceres::QuaternionRotatePoint(rotation, pt3d_prev, pt3d_curr);
    ceres::UnitQuaternionRotatePoint(rotation, pt3d_prev, pt3d_curr);

    pt3d_curr[0] += translation[0];
    pt3d_curr[1] += translation[1];
    pt3d_curr[2] += translation[2];

    T f = T(cam_intr[0]);
    T cx = T(cam_intr[2]);
    T cy = T(cam_intr[3]);
    T b = T(cam_intr[4]);

    // Transform the point from homogeneous to euclidean
    T xe = pt3d_curr[0] / pt3d_curr[2];   // x / z
    T ye = pt3d_curr[1] / pt3d_curr[2];   // y / z

    //// Apply the focal length
    //const T& focal = cam_f[0];
    T predicted_left_x = f * xe + cx;
    T predicted_left_y = f * ye + cy;
    //std::cout << predicted_x_left << " -- " << predicted_y_left << "\n";
    //std::cout << cam_intr << "\n";

    // now for right camera
    // first move point in right cam coord system
    pt3d_curr[0] -= b;
    xe = pt3d_curr[0] / pt3d_curr[2];   // x / z
    ye = pt3d_curr[1] / pt3d_curr[2];   // y / z
    T predicted_right_x = f * xe + cx;
    T predicted_right_y = f * ye + cy;

    // without weighting
    //// Compute and return the error is the difference between the predicted and observed position
    residuals[0] = predicted_left_x - T(observed_left.x_);
    residuals[1] = predicted_left_y - T(observed_left.y_);
    residuals[2] = predicted_right_x - T(observed_right.x_);
    residuals[3] = predicted_right_y - T(observed_right.y_);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector3d& pt3d, const core::Point& left_proj,
                                     const core::Point& right_proj, const double* camera_intr) {
    return (new ceres::AutoDiffCostFunction<ReprojectionErrorWithQuaternion,4,4,3>(
                new ReprojectionErrorWithQuaternion(pt3d, left_proj, right_proj, camera_intr)));
  }

  const Eigen::Vector3d& pt3d;
  const core::Point& observed_left;
  const core::Point& observed_right;
  const double* cam_intr;
};


struct WeightedReprojectionError {
  // (u, v): the position of the observation with respect to the image
  // center point.
  WeightedReprojectionError(const Eigen::Vector3d& point3d, const core::Point& obs_left,
                            const core::Point& obs_right, const double* camera_intr)
      : observed_left(obs_left), observed_right(obs_right), pt3d(point3d), cam_intr(camera_intr) {
    //weight = GetLibvisoWeight(obs_left, obs_right);
    double cx = camera_intr[2];
    weight = 1.0 / (std::fabs(obs_left.x_ - cx)/std::fabs(cx) + 0.05);
  }

  template <typename T>
  bool operator()(const T* const rotation,
                  const T* const translation,
                  T* residuals) const {
    T pt3d_prev[3];
    pt3d_prev[0] = T(pt3d[0]);
    pt3d_prev[1] = T(pt3d[1]);
    pt3d_prev[2] = T(pt3d[2]);

    T pt3d_curr[3];
    // Use a quaternion rotation that doesn't assume the quaternion is
    // normalized, since one of the ways to run the bundler is to let Ceres
    // optimize all 4 quaternion parameters unconstrained.
    //ceres::QuaternionRotatePoint(rotation, pt3d_prev, pt3d_curr);
    ceres::UnitQuaternionRotatePoint(rotation, pt3d_prev, pt3d_curr);

    pt3d_curr[0] += translation[0];
    pt3d_curr[1] += translation[1];
    pt3d_curr[2] += translation[2];

    T f = T(cam_intr[0]);
    T cx = T(cam_intr[2]);
    T cy = T(cam_intr[3]);
    T b = T(cam_intr[4]);

    // Transform the point from homogeneous to euclidean
    T xe = pt3d_curr[0] / pt3d_curr[2];   // x / z
    T ye = pt3d_curr[1] / pt3d_curr[2];   // y / z

    //// Apply the focal length
    //const T& focal = cam_f[0];
    T predicted_left_x = f * xe + cx;
    T predicted_left_y = f * ye + cy;
    //std::cout << predicted_x_left << " -- " << predicted_y_left << "\n";
    //std::cout << cam_intr << "\n";

    // now for right camera
    // first move point in right cam coord system
    pt3d_curr[0] -= b;
    xe = pt3d_curr[0] / pt3d_curr[2];   // x / z
    ye = pt3d_curr[1] / pt3d_curr[2];   // y / z
    T predicted_right_x = f * xe + cx;
    T predicted_right_y = f * ye + cy;

    T w = T(weight);
    // without weighting
    //// Compute and return the error is the difference between the predicted and observed position
    residuals[0] = w * (predicted_left_x - T(observed_left.x_));
    residuals[1] = w * (predicted_left_y - T(observed_left.y_));
    residuals[2] = w * (predicted_right_x - T(observed_right.x_));
    residuals[3] = w * (predicted_right_y - T(observed_right.y_));

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector3d& pt3d, const core::Point& left_proj,
                                     const core::Point& right_proj, const double* camera_intr) {
    return (new ceres::AutoDiffCostFunction<WeightedReprojectionError,4,4,3>(
            new WeightedReprojectionError(pt3d, left_proj, right_proj, camera_intr)));
  }

  const Eigen::Vector3d& pt3d;
  const core::Point& observed_left;
  const core::Point& observed_right;
  const double* cam_intr;
  double weight;
};

#endif
