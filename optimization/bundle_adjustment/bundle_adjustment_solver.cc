#include "bundle_adjustment_solver.h"

#include "../../core/math_helper.h"
#include "ceres_helper.h"

namespace optim {

BundleAdjustmentSolver::BundleAdjustmentSolver(const std::string loss_type,
                                               const std::vector<double>& params,
                                               bool use_weighting) : use_weighting_(use_weighting) {
  //loss_function_ = new ceres::CauchyLoss(0.15);
  loss_function_ = CeresHelper::CreateLoss(loss_type, params);
  //loss_function_ = nullptr;
}

void BundleAdjustmentSolver::AddCameraMotion(const Eigen::Matrix4d& Rt) {
  std::array<double,3> trans;
  //trans[0] = Rt(0,3);
  //trans[1] = Rt(1,3);
  //trans[2] = Rt(2,3);
  ////Eigen::Matrix3d mrot;
  ////double R[9];
  //Eigen::Matrix3d R;
  //for (int i = 0; i < 3; i++)
  //  for (int j = 0; j < 3; j++)
  //    //R[i*3+j] = Rt(i,j);
  //    R(i,j) = Rt(i,j);
  //Eigen::Quaterniond q(R);
  //// ceres is 2 times slower
  ////ceres::RotationMatrixToQuaternion(R.data(), &quaternion[0]);
  ////std::cout << q.toRotationMatrix() << "\n\n";
  std::array<double,4> quaternion;
  //quaternion[0] = q.w();
  //quaternion[1] = q.x();
  //quaternion[2] = q.y();
  //quaternion[3] = q.z();
  core::MathHelper::MotionMatrixToParams(Rt, quaternion, trans);
  translation_.push_back(trans);
  rotation_.push_back(quaternion);
}

void BundleAdjustmentSolver::AddTrackData(const TrackData& data) {
  const int num_motions = translation_.size();
  //const int max_frames = translation_.size();
  int end_frame = (num_motions - data.dist_from_cframe);
  int num_obs = data.left_tracks.size();
  int start_frame = std::max(0, (end_frame+1) - num_obs);
  int num_used_obs = end_frame - start_frame + 1;
  int start_obs = num_obs - num_used_obs;
  //std::cout << data.dist_from_cframe << "\n";
  //std::cout << num_used_obs << "\n\n";

  // add all costs
  //for (; start_obs < num_obs; start_obs++) {
  //  start_frame++;
  //  num_used_obs--;

  Eigen::Vector4d pt3d;
  core::MathHelper::Triangulate(camera_params_, data.left_tracks[start_obs],
                                data.right_tracks[start_obs], pt3d);
  for (int i = 1; i < num_used_obs; i++) {
    ReprojErrorStereo* reproj_error = new ReprojErrorStereo(pt3d, data.left_tracks[start_obs + i],
        data.right_tracks[start_obs + i], camera_params_, use_weighting_);
    switch (i) {
      case 1: {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4>(reproj_error);
        ceres_problem_.AddResidualBlock(cost, loss_function_,
                                        &translation_[start_frame][0],
                                        &rotation_[start_frame][0]);
        break;
      }
      case 2: {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4,3,4>(reproj_error);
        ceres_problem_.AddResidualBlock(cost, loss_function_,
                                        &translation_[start_frame][0],
                                        &rotation_[start_frame][0],
                                        &translation_[start_frame+1][0],
                                        &rotation_[start_frame+1][0]);
        break;
      }
      case 3: {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4,3,4,3,4>(reproj_error);
        ceres_problem_.AddResidualBlock(cost, loss_function_,
                                        &translation_[start_frame][0],
                                        &rotation_[start_frame][0],
                                        &translation_[start_frame+1][0],
                                        &rotation_[start_frame+1][0],
                                        &translation_[start_frame+2][0],
                                        &rotation_[start_frame+2][0]);
        break;
      }
      case 4: {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4,3,4,3,4,3,4>(reproj_error);
        ceres_problem_.AddResidualBlock(cost, loss_function_,
                                        &translation_[start_frame][0],
                                        &rotation_[start_frame][0],
                                        &translation_[start_frame+1][0],
                                        &rotation_[start_frame+1][0],
                                        &translation_[start_frame+2][0],
                                        &rotation_[start_frame+2][0],
                                        &translation_[start_frame+3][0],
                                        &rotation_[start_frame+3][0]);
        break;
      }
      default: {
        assert(false);
      }
    }
  }
}

//void BundleAdjustmentSolver::AddTrackData(const TrackData& data) {
//  const int num_motions = translation_.size();
//  ////const int max_frames = translation_.size();
//  int end_frame = (num_motions - data.dist_from_cframe);
//  int num_obs = data.left_tracks.size();
//  int start_frame = std::max(0, (end_frame+1) - num_obs);
//  int num_used_obs = end_frame - start_frame + 1;
//  //int num_used_obs = num_obs - start_obs;
//  int start_obs = num_obs - num_used_obs;
//  //std::cout << data.dist_from_cframe << "\n";
//  //std::cout << num_used_obs << "\n\n";
//
//  //int num_obs = data.left_tracks.size();
//  //int start_obs = num_obs - num_motions + data.dist_from_cframe;
//  //int start_frame = std::max(0, std::abs(start_obs));
//  //start_obs = std::max(0, start_obs);
//
//  // add all costs
//  for (; start_obs < num_obs; start_obs++, start_frame++, num_used_obs--) {
//    //num_used_obs = num_obs - start_obs;
//    //if (num_used_obs != num_used_obs2) throw 1;
//
//    Eigen::Vector4d pt3d;
//    core::MathHelper::Triangulate(camera_params_, data.left_tracks[start_obs],
//                                  data.right_tracks[start_obs], pt3d);
//    for (int i = 1; i < num_used_obs; i++) {
//      //ReprojErrorStereo* reproj_error = new ReprojErrorStereo(pt3d, data.left_tracks[start_obs + i],
//      //    data.right_tracks[start_obs + i], camera_params_);
//      ReprojErrorStereo* reproj_error = new ReprojErrorStereo(pt3d, data.left_tracks[start_obs + i],
//          data.right_tracks[start_obs + i], camera_params_, use_weighting_);
//      switch (i) {
//        case 1: {
//          ceres::CostFunction* cost =
//              new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4>(reproj_error);
//          ceres_problem_.AddResidualBlock(cost, loss_function_,
//                                          &translation_[start_frame][0],
//                                          &rotation_[start_frame][0]);
//          break;
//        }
//        case 2: {
//          ceres::CostFunction* cost =
//              new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4,3,4>(reproj_error);
//          ceres_problem_.AddResidualBlock(cost, loss_function_,
//                                          &translation_[start_frame][0],
//                                          &rotation_[start_frame][0],
//                                          &translation_[start_frame+1][0],
//                                          &rotation_[start_frame+1][0]);
//          break;
//        }
//        case 3: {
//          ceres::CostFunction* cost =
//              new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4,3,4,3,4>(reproj_error);
//          ceres_problem_.AddResidualBlock(cost, loss_function_,
//                                          &translation_[start_frame][0],
//                                          &rotation_[start_frame][0],
//                                          &translation_[start_frame+1][0],
//                                          &rotation_[start_frame+1][0],
//                                          &translation_[start_frame+2][0],
//                                          &rotation_[start_frame+2][0]);
//          break;
//        }
//        case 4: {
//          ceres::CostFunction* cost =
//              new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4,3,4,3,4,3,4>(reproj_error);
//          ceres_problem_.AddResidualBlock(cost, loss_function_,
//                                          &translation_[start_frame][0],
//                                          &rotation_[start_frame][0],
//                                          &translation_[start_frame+1][0],
//                                          &rotation_[start_frame+1][0],
//                                          &translation_[start_frame+2][0],
//                                          &rotation_[start_frame+2][0],
//                                          &translation_[start_frame+3][0],
//                                          &rotation_[start_frame+3][0]);
//          break;
//        }
//        case 5: {
//          ceres::CostFunction* cost =
//              new ceres::AutoDiffCostFunction<ReprojErrorStereo,4,3,4,3,4,3,4,3,4,3,4>(reproj_error);
//          ceres_problem_.AddResidualBlock(cost, loss_function_,
//                                          &translation_[start_frame][0],
//                                          &rotation_[start_frame][0],
//                                          &translation_[start_frame+1][0],
//                                          &rotation_[start_frame+1][0],
//                                          &translation_[start_frame+2][0],
//                                          &rotation_[start_frame+2][0],
//                                          &translation_[start_frame+3][0],
//                                          &rotation_[start_frame+3][0],
//                                          &translation_[start_frame+4][0],
//                                          &rotation_[start_frame+4][0]);
//          break;
//        }
//        default: {
//          throw 1;
//        }
//      }
//    }
//  }
//}

bool BundleAdjustmentSolver::Solve() {
  ceres::Solver::Options options;
  // For bundle adjustment problems with up to a hundred or so cameras, use DENSE_SCHUR.
  // For larger bundle adjustment problems with sparse Schur Complement/Reduced camera matrices
  // use SPARSE_SCHUR. This requires that you have SuiteSparse or CXSparse installed.
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_num_iterations = 100;
  //options.minimizer_progress_to_stdout = true;
  options.eta = 1e-2;
  options.num_threads = 4;
  //options.linear_solver_type = ceres::SPARSE_SCHUR;
  //options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  //options.num_threads = omp_get_num_threads();
  //options.logging_type = ceres::SILENT;

  ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
  for (size_t i = 0; i < rotation_.size(); i++)
    ceres_problem_.SetParameterization(&rotation_[i][0], quaternion_parameterization);

  ceres::Solver::Summary summary;
  ceres::Solve(options, &ceres_problem_, &summary);
  //std::cout << summary.FullReport() << "\n";
  if (summary.termination_type != ceres::TerminationType::CONVERGENCE)
    std::cout << "[BundleAdjustmentSolver] Warning: NO CONVERGENCE\n";
  return summary.termination_type != ceres::TerminationType::FAILURE;
}

// get camera extrinsic transform from cam local coord to world coord (1-frame coord)
Eigen::Matrix4d BundleAdjustmentSolver::GetCameraMotion(int camera_index) const {
  //Eigen::Matrix3d rmat;
  //ceres::AngleAxisToRotationMatrix(&extr_rot_[ci][0], (const double *)rmat.data());
  //ceres::AngleAxisToRotationMatrix((const double*) &rotation_[camera_index][0], rmat.data());
  //ceres::QuaternionToRotation()
  Eigen::Quaterniond q;
  q.w() = rotation_[camera_index][0];
  q.x() = rotation_[camera_index][1];
  q.y() = rotation_[camera_index][2];
  q.z() = rotation_[camera_index][3];
  assert(q.norm() - 1.0 < 1e-10);
  //std::cout << "\nNorm = " << q.norm() << "\n";

  Eigen::Matrix4d Rt = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 3; i++)
    Rt(i,3) = translation_[camera_index][i];
  Rt.block<3,3>(0,0) = q.toRotationMatrix();
  return Rt;
}

} // end namespace optim
