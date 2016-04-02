#include "sba_ceres.h"

#include <Eigen/Core>
//#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

namespace optim {

SBAceres::SBAceres(BALProblemStereo* problem, BAType ba_type)
  : sba_problem_(problem), ba_type_(ba_type) {}

// cam_params - vector: [fu fv cu cv baseline]
void SBAceres::setCameraIntrinsics(const cv::Mat& cam_params)
{
  cam_params.copyTo(cam_intr_);
}

// Rt - matrix 4x4
void SBAceres::addCameraMotion(const cv::Mat& Rt)
{
  std::array<double,3> trans;
  trans[0] = Rt.at<double>(0,3);
  trans[1] = Rt.at<double>(1,3);
  trans[2] = Rt.at<double>(2,3);

  Eigen::Matrix3d mrot;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      mrot(i,j) = Rt.at<double>(i,j);

  std::array<double,3> angleAxisV;
  //double angleAxis[3];
  //ceres::RotationMatrixToAngleAxis((const double*)mrot.data(), angleAxis);
  ceres::RotationMatrixToAngleAxis((const double*)mrot.data(), &angleAxisV[0]);
  //for(int i = 0; i < 3; i++)
  //  if(angleAxis[i] != angleAxisV[i]) throw "Error!";
  //  //std::cout << angleAxis[i] << " == " << angleAxisV[i] << "\n";

  cam_extr_trans_.push_back(trans);
  cam_extr_rot_.push_back(angleAxisV);
}


void SBAceres::addMonoProj(int ci, int pi, const core::Point& proj)
{
  throw("Error!");
  mono_projs_.insert(std::make_pair(std::make_pair(ci, pi), proj));
}

void SBAceres::addStereoProj(int ci, int pi, const core::Point& proj_left, const core::Point& proj_right)
{
  projs_.insert(std::make_pair(std::make_pair(ci, pi), std::make_pair(proj_left, proj_right)));
}

void SBAceres::addPoint(const cv::Mat& pt, double weight)
{
  pts3d_.push_back(pt.clone());
  weights_.push_back(weight);
}
//void SBAros::addPoint(double* pt)
//{
//   Eigen::Vector4d tmp_pt;
//   for(int i = 0; i < 3; i++)
//      pt(i) = pt[i];
//   pt(3) = 1.0;
//
//   sba_.addPoint(tmp_pt);
//}

void SBAceres::setupSBAproblem()
{

  // Configure the size of the problem
  int nviews = cam_extr_trans_.size();
  int npoints = pts3d_.size();
  //int nintrinsics = 5;
  assert(cam_intr_.rows == 1 || cam_intr_.cols == 1);
  int nintrinsics = std::max(cam_intr_.rows, cam_intr_.cols);
  sba_problem_->num_cameras_ = nviews;
  sba_problem_->num_points_ = npoints;
  sba_problem_->num_observations_ = projs_.size();
  sba_problem_->point_index_ = new int[sba_problem_->num_observations_];
  sba_problem_->camera_index_ = new int[sba_problem_->num_observations_];
  sba_problem_->observations_ = new double[4 * sba_problem_->num_observations_];
  sba_problem_->obs_weights_ = new double[sba_problem_->num_observations_];

  sba_problem_->cam_intrinsics_ = new double[nintrinsics];
  sba_problem_->num_parameters_ = sba_problem_->camera_block_size() * sba_problem_->num_cameras_ +
                                  sba_problem_->point_block_size() * sba_problem_->num_points_;
  sba_problem_->parameters_ = new double[sba_problem_->num_parameters_];

  int i = 0;
  for(auto iter = projs_.begin(); iter != projs_.end(); iter++) {
    int ci = iter->first.first;
    int pi = iter->first.second;
    sba_problem_->camera_index_[i] = ci;
    sba_problem_->point_index_[i] = pi;
    sba_problem_->obs_weights_[i] = weights_[pi];

    core::Point left_proj = iter->second.first;
    core::Point right_proj = iter->second.second;
//    //std::cout << "ci = " << ci << " -- pi = " << pi << " left: " << left_proj <<
//    //             ", right: " << right_proj << " " << "\n";
    sba_problem_->observations_[i*4] = left_proj.x_;
    sba_problem_->observations_[i*4+1] = left_proj.y_;
    sba_problem_->observations_[i*4+2] = right_proj.x_;
    sba_problem_->observations_[i*4+3] = right_proj.y_;
    i++;
  }

  for(int i = 0; i < nintrinsics; i++)
    sba_problem_->cam_intrinsics_[i] = cam_intr_.at<double>(i);

  // Add camera parameters (R, t, intrinsics)
  for(int i = 0; i < nviews; ++i) {
    // Rotation matrix to angle axis
//    //std::vector<double> angleAxis(3);
//    //ceres::RotationMatrixToAngleAxis((const double*)d._R[j].data(), &angleAxis[0]);
//    //// translation
//    //Vec3 t = d._t[j];
//    //double focal = d._K[j](0,0);

    // 6 mutable params
    //int cbs = 6;
    int cbs = sba_problem_->camera_block_size();
    sba_problem_->parameters_[i*cbs] = cam_extr_rot_[i][0];
    sba_problem_->parameters_[i*cbs+1] = cam_extr_rot_[i][1];
    sba_problem_->parameters_[i*cbs+2] = cam_extr_rot_[i][2];
    sba_problem_->parameters_[i*cbs+3] = cam_extr_trans_[i][0];
    sba_problem_->parameters_[i*cbs+4] = cam_extr_trans_[i][1];
    sba_problem_->parameters_[i*cbs+5] = cam_extr_trans_[i][2];
  }
//  // Add 3D points coordinates parameters
  double* points = sba_problem_->mutable_points();
  int pbs = 3;
  for (int i = 0; i < npoints; ++i) {
    for(int j = 0; j < 3; j++) {
      points[i*pbs + j] = pts3d_[i].at<double>(j);
    }
  }
}

void SBAceres::runSBA()
{
  // Setup a BA problem
  setupSBAproblem();

  //const double* observations = sba_problem_->observations();

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for(int i = 0; i < sba_problem_->num_observations(); i++) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.

    if(ba_type_ == kMotion) {
      // optimize only motion - better for stereo
      ceres::CostFunction* cost_function = ReprojectionErrorStereoMotion::create(
                                            &sba_problem_->observations()[4*i],
                                            sba_problem_->point_for_observation(i),
                                            sba_problem_->camera_intrinsics(),
                                            sba_problem_->get_observation_weight(i));
      problem.AddResidualBlock(cost_function,
                               NULL /* squared loss */,
                               sba_problem_->mutable_camera_for_observation(i));
    }
    else if(ba_type_ == kStructureAndMotion) {
      // optimize structure and motion
      ceres::CostFunction* cost_function = ReprojectionErrorStereoStructureAndMotion::create(
                                            &sba_problem_->observations()[4*i],
                                            sba_problem_->camera_intrinsics(),
                                            sba_problem_->get_observation_weight(i));
      problem.AddResidualBlock(cost_function,
                               NULL /* squared loss */,
                               sba_problem_->mutable_camera_for_observation(i),
                               sba_problem_->mutable_point_for_observation(i));
    }
    else throw 1;
  }

  // For bundle adjustment problems with up to a hundred or so cameras, use DENSE_SCHUR.
  // For larger bundle adjustment problems with sparse Schur Complement/Reduced camera matrices use SPARSE_SCHUR.
  // This requires that you have SuiteSparse or CXSparse installed.

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.linear_solver_type = ceres::SPARSE_SCHUR;     // default
  //options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  //options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  //options.minimizer_progress_to_stdout = true;

  //if(ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
  //  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  //else if(ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
  //  options.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
  //else
  //{
  //  // No sparse backend for Ceres.
  //  // Use dense solving
  //  throw "Error!\n";
  //  options.linear_solver_type = ceres::DENSE_SCHUR;
  //}
  //options.logging_type = ceres::SILENT;
//#ifdef USE_OPENMP
//  options.num_threads = omp_get_num_threads();
//#endif // USE_OPENMP

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  // update params with result
  int cbs = sba_problem_->camera_block_size();
  for(int i = 0; i < cam_extr_rot_.size(); i++) {
    double* cam_params = &sba_problem_->mutable_cameras()[i * cbs];
    for(int j = 0; j < 3; j++) {
      cam_extr_rot_[i][j] = cam_params[j];
      cam_extr_trans_[i][j] = cam_params[3+j];
    }
  }
}

cv::Mat SBAceres::getCameraRt(int ci) const
{
  assert(ci < cam_extr_trans_.size());
  Eigen::Matrix3d rmat;
  //ceres::AngleAxisToRotationMatrix(&cam_extr_rot_[ci][0], (const double *)rmat.data());
  ceres::AngleAxisToRotationMatrix((const double*) &cam_extr_rot_[ci][0], rmat.data());
  //std::cout << rmat << "\n";

  cv::Mat cvmat = cv::Mat::eye(4,4,CV_64F);
  for(int i = 0; i < rmat.rows(); i++)
    for(int j = 0; j < rmat.cols(); j++)
      cvmat.at<double>(i,j) = rmat(i,j);
  for(int i = 0; i < 3; i++)
    cvmat.at<double>(i,3) = cam_extr_trans_[ci][i];

  return cvmat;
}

}

