#ifndef STEREO_EGOMOTION_BASE_EGOMOTION_SOLVER_H_
#define STEREO_EGOMOTION_BASE_EGOMOTION_SOLVER_H_

#include <vector>
#include <Eigen/Core>
#include <ceres/ceres.h>

#include "../../core/types.h"

namespace egomotion {

class EgomotionSolver {
 public:
  EgomotionSolver(const double* cam_params, const std::string loss_function_type,
                  double robust_loss_scale, bool use_weighting);
  ~EgomotionSolver();
  void InitializeParams(const Eigen::Matrix4d& Rt);
  void AddPoint(const Eigen::Vector3d& pt3d, const core::Point& left_proj,
                const core::Point& right_proj);
  void ClearPoints();
  bool Solve(Eigen::Matrix4d& Rt);
  //EgomotionSolver(int num_pts);
  //void Set3DPointsSparse(const std::vector<Eigen::Vector3d>& points, const std::vector<int>& active);
  //void SetProjectionsSparse(const std::vector<cv::Point>& left_curr, const std::vector<cv::Point>&right_curr,
  //                          const std::vector<int>& active);
 private:
  void ResetParams();
  //int num_pts_;
  //double *rotation_, *translation_;
  std::array<double,4> rotation_;
  std::array<double,3> translation_;
  const double* cam_params_;
  const std::string loss_function_type_;
  double robust_loss_scale_;
  bool params_initialized_ = false;
  bool use_weighting_;

  std::vector<Eigen::Vector3d> pts3d_;
  std::vector<core::Point> left_projs_;
  std::vector<core::Point> right_projs_;
  ceres::Solver::Summary summary_;  
  ceres::Solver::Options options_;
};

inline
void EgomotionSolver::ClearPoints() {
  pts3d_.clear();
  left_projs_.clear();
  right_projs_.clear();
}

}

#endif
