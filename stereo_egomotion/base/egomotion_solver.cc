#include "egomotion_solver.h"

#include "cost_functions.h"
#include "../../optimization/bundle_adjustment/ceres_helper.h"
#include "../../core/math_helper.h"

namespace egomotion {

EgomotionSolver::EgomotionSolver(const double* cam_params, const std::string loss_function_type,
                                 double robust_loss_scale, bool use_weighting) :
  cam_params_(cam_params), loss_function_type_(loss_function_type),
  robust_loss_scale_(robust_loss_scale), use_weighting_(use_weighting) {

  // Minimizer options_
  //options_.max_num_iterations = 5;
  options_.max_num_iterations = 10;
  //options_.minimizer_progress_to_stdort = true;
  options_.num_threads = 1;
  // Default value for eta. Eta determines the
  // accuracy of each linear solve of the truncated newton step.
  // Changing this parameter can affect solve performance.
  options_.eta = 1e-2;
  // Maximum solve time in seconds
  options_.max_solver_time_in_seconds = 1e32;
  options_.use_nonmonotonic_steps = false;
  options_.minimizer_type = ceres::TRUST_REGION;
  //options_->minimizer_type = ceres::LINE_SEARCH;
  //options_->trust_region_strategy_type = ;
  // Use inner iterations to non-linearly refine each successful trust region step.
  options_.use_inner_iterations = false;
  //options_.gradient_tolerance = 1e-16;
  //options_.function_tolerance = 1e-16;

  // Linear solver options_
  options_.linear_solver_type = ceres::DENSE_SCHUR;
  //CHECK(StringToPreconditionerType(FLAGS_preconditioner,
  //                                 &options_->preconditioner_type));
  //CHECK(StringToVisibilityClusteringType(FLAGS_visibility_clustering,
  //                                       &options_->visibility_clustering_type));
  //CHECK(StringToSparseLinearAlgebraLibraryType(
  //          FLAGS_sparse_linear_algebra_library,
  //          &options_->sparse_linear_algebra_library_type));
  //CHECK(StringToDenseLinearAlgebraLibraryType(
  //          FLAGS_dense_linear_algebra_library,
  //          &options_->dense_linear_algebra_library_type));
  //options_->num_linear_solver_threads = FLAGS_num_threads;
  //options_->use_explicit_schur_complement = FLAGS_explicit_schur_complement;
}

EgomotionSolver::~EgomotionSolver() {
}

void EgomotionSolver::ResetParams() {
  rotation_[0] = 1.0;
  for (int i = 1; i < 4; i++) {
    rotation_[i] = 0.0;
    translation_[i-1] = 0.0;
  }
}

void EgomotionSolver::AddPoint(const Eigen::Vector3d& pt3d, const core::Point& left_proj,
                               const core::Point& right_proj) {
  pts3d_.push_back(pt3d);
  left_projs_.push_back(left_proj);
  right_projs_.push_back(right_proj);
}

void EgomotionSolver::InitializeParams(const Eigen::Matrix4d& Rt) {
  core::MathHelper::MotionMatrixToParams(Rt, rotation_, translation_);
  params_initialized_ = true;
}

bool EgomotionSolver::Solve(Eigen::Matrix4d& Rt) {
  if (pts3d_.size() < 3) {
    std::cout << "Not enough points!1\n";
    return false;
  }
  ceres::Problem problem;
  if (!params_initialized_)
    ResetParams();

  std::vector<double> params = { robust_loss_scale_ };
  ceres::LossFunction* loss_function = optim::CeresHelper::CreateLoss(loss_function_type_, params);
  for (size_t i = 0; i < pts3d_.size(); i++) {
    // Each Residual block takes a point and a camera as input and
    // outputs a 2 dimensional residual.
    if (!use_weighting_) {
      ceres::CostFunction* cost_function = ReprojectionErrorWithQuaternion::
          Create(pts3d_[i], left_projs_[i], right_projs_[i], cam_params_);
      problem.AddResidualBlock(cost_function, loss_function, &rotation_[0], &translation_[0]);
    }
    else {
      //std::cout << "ALALL";
      ceres::CostFunction* cost_function = WeightedReprojectionError::
          Create(pts3d_[i], left_projs_[i], right_projs_[i], cam_params_);
      problem.AddResidualBlock(cost_function, loss_function, &rotation_[0], &translation_[0]);
    }
  }
  ceres::LocalParameterization* quaternion_parameterization =
      new ceres::QuaternionParameterization;
  problem.SetParameterization(&rotation_[0], quaternion_parameterization);

  ceres::Solve(options_, &problem, &summary_);
  //std::cout << summary_.FullReport() << "\n";
  //std::cout << summary_.BriefReport() << "\n";

  // convert params to trans matrix
  //for (int i = 0; i < 4; i++)
  //  printf("%f, ", rotation_[i]);
  //printf("\n");
  //for (int i = 0; i < 3; i++)
  //  printf("%f, ", translation_[i]);
  Eigen::Quaterniond q;
  q.w() = rotation_[0];
  q.x() = rotation_[1];
  q.y() = rotation_[2];
  q.z() = rotation_[3];
  assert(q.norm() - 1.0 < 1e-10);
  //std::cout << "\nNorm = " << q.norm() << "\n";

  Rt = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 3; i++)
    Rt(i,3) = translation_[i];
  Rt.block<3,3>(0,0) = q.toRotationMatrix();
  //std::cout << Rt << "\n";
  params_initialized_ = false;

  if (summary_.IsSolutionUsable())
    return true;
  else
    return false;
}

}
