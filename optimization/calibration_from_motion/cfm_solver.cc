//TODO: learned inverse deformation on artificial dataset is wrong ... - error is to big

#include "cfm_solver.h"

#include <opencv2/highgui/highgui.hpp>

#include "../../tracker/base/eval_helper.h"
#include "../../core/math_helper.h"

namespace optim {

CFMSolver::CFMSolver(int img_rows, int img_cols, double loss_scale) :
                     img_rows_(img_rows), img_cols_(img_cols), loss_scale_(loss_scale) {
  bool use_init_calib = true;
  init_k1_[0] = -3.792567e-01;
  init_k1_[1] = 2.121203e-01;
  init_k1_[2] = 9.182571e-04;
  init_k1_[3] = 1.911304e-03;
  init_k1_[4] = -7.605535e-02;
  init_k2_[0] = -3.720803e-01;
  init_k2_[1] = 1.944116e-01;
  init_k2_[2] = -1.077099e-04;
  init_k2_[3] = -9.031379e-05;
  init_k2_[4] = -6.314998e-02;

  for (int i = 0; i < 2; i++) {
    f1_[i] = 1.0;
    f2_[i] = 1.0;
    //dc1_[i] = 0.0;
    //dc2_[i] = 0.0;
  }
  pp1_[0] = img_cols / 2.0;
  pp1_[1] = img_rows / 2.0;
  pp2_[0] = img_cols / 2.0;
  pp2_[1] = img_rows / 2.0;

  if (use_init_calib) {
    f1_[0] = 9.786977e+02;
    f1_[1] = 9.717435e+02;
    pp1_[0] = 6.900000e+02;
    pp1_[1] = 2.497222e+02;
    f2_[0] = 9.892043e+02;
    f2_[1] = 9.832048e+02;
    pp2_[0] = 7.020000e+02;
    pp2_[1] = 2.616538e+02;
  }
//TODO: write optmization which will find inverse dist params

  //for (int i = 0; i < kNumK; i++) {
  //  k1_[i] = init_k1_[i];
  //  k2_[i] = init_k2_[i];
  //}
  for (int i = 0; i < kNumK; i++) {
    k1_[i] = 0.0;
    k2_[i] = 0.0;
  }
  //k1_[0] = -1.8e1;
  //k2_[0] = -1.8e1;

  rot_[0] = 1.0;
  for (int i = 0; i < 3; i++) {
    rot_[i+1] = 0.0;
    trans_[i] = 0.0;
  }
  trans_[0] = 0.5;

  if (use_init_calib) {
    double rdata[] = { 9.993424e-01,1.830363e-02,-3.129928e-02,
                       -1.856768e-02,9.997943e-01,-8.166432e-03,
                       3.114337e-02,8.742218e-03,9.994767e-01 };
    Eigen::Matrix3d rmat(rdata);
    Eigen::Quaterniond q(rmat);
    q.normalize();
    rot_[0] = q.w();
    rot_[1] = q.x();
    rot_[2] = q.y();
    rot_[3] = q.z();
    trans_[0] = 5.370000e-01;
    trans_[1] = -5.591661e-03;
    trans_[2] = 1.200541e-02;
  }

  min_points_ = 1000;
  //google::InitGoogleLogging("ceres");
  //char** argv = "--logtostderr=1";
  //google::ParseCommandLineFlags(&argc, &argv, true);
}

CFMSolver::~CFMSolver() {
}

//void CFMSolver::ClearBadTracks() {
//
//}

// Rt is the GT motion of world points with respect to camera
void CFMSolver::UpdateTracks(const track::StereoTrackerBase& tracker, const cv::Mat& Rt) {
  std::vector<std::tuple<core::Point,core::Point>> pts_left, pts_right;
  std::vector<int> age;

  gt_rt_.push_back(Rt.clone());
  std::array<double,3> translation;
  std::array<double,4> quaternion;
  core::MathHelper::MotionMatrixToParams(Rt, quaternion, translation);

  egomotion_translation_.push_back(translation);
  egomotion_rotation_.push_back(quaternion);
  //std::copy(quaternion.begin(), quaternion.end(), std::ostream_iterator<double>(std::cout,","));

  // ARTIF DATASET
  //std::random_device rd;
  //std::mt19937 gen(rd());
  //// values near the mean are the most likely
  //// standard deviation affects the dispersion of generated values from the mean
  //std::normal_distribution<double> x_nd(0, 6);
  //std::normal_distribution<double> y_nd(1, 2);
  //std::normal_distribution<double> z_nd(15, 6);

  //size_t num = 200;
  //for (size_t i = 0; i < num; i++) {
  //  //Eigen::Vector4d pt3d;
  //  double pt3d_prev[3], pt3d_curr[3];
  //  pt3d_prev[0] = x_nd(gen);
  //  pt3d_prev[1] = y_nd(gen);
  //  pt3d_prev[2] = z_nd(gen);
  //  ceres::UnitQuaternionRotatePoint(&quaternion[0], pt3d_prev, pt3d_curr);
  //  for (size_t k = 0; k < 3; k++)
  //    pt3d_curr[k] += translation[k];
  //  //std::cout << "Prev: " << PrintPoint3D<double>(pt3d_prev) << "\n";
  //  //std::cout << "Curr: " << PrintPoint3D<double>(pt3d_curr) << "\n";

  //  if (pt3d_prev[2] > 1 && pt3d_curr[2] > 1) {
  //    age.push_back(1);
  //  }
  //  else {
  //    age.push_back(-1);
  //    track::FeatureInfo feat_left, feat_right;
  //    auto left_track = std::make_tuple(feat_left.prev_, feat_left.curr_);
  //    auto right_track = std::make_tuple(feat_right.prev_, feat_right.curr_);
  //    pts_left.push_back(left_track);
  //    pts_right.push_back(right_track);
  //    continue;
  //  }
  //  
  //  track::FeatureInfo feat_left, feat_right;
  //  double pt2d[2];
  //  ProjectToCamera<double>(pt3d_prev, f1_, pp1_, k1_, pt2d);
  //  feat_left.prev_.x_ = pt2d[0];
  //  feat_left.prev_.y_ = pt2d[1];
  //  ProjectToCamera<double>(pt3d_curr, f1_, pp1_, k1_, pt2d);
  //  feat_left.curr_.x_ = pt2d[0];
  //  feat_left.curr_.y_ = pt2d[1];

  //  double pt3d_right_tprev[3];
  //  double pt3d_right_tcurr[3];
  //  for (int i = 0; i < 3; i++) {
  //    pt3d_right_tprev[i] = pt3d_prev[i] - trans_[i];
  //    pt3d_right_tcurr[i] = pt3d_curr[i] - trans_[i];
  //  }
  //  double pt3d_right_prev[3];
  //  double pt3d_right_curr[3];
  //  double q_inv[4];
  //  InvertQuaternion<double>(rot_, q_inv);
  //  ceres::UnitQuaternionRotatePoint(q_inv, pt3d_right_tprev, pt3d_right_prev);
  //  ceres::UnitQuaternionRotatePoint(q_inv, pt3d_right_tcurr, pt3d_right_curr);

  //  ProjectToCamera<double>(pt3d_right_prev, f2_, pp2_, k2_, pt2d);
  //  feat_right.prev_.x_ = pt2d[0];
  //  feat_right.prev_.y_ = pt2d[1];
  //  ProjectToCamera<double>(pt3d_right_curr, f2_, pp2_, k2_, pt2d);
  //  feat_right.curr_.x_ = pt2d[0];
  //  feat_right.curr_.y_ = pt2d[1];
  //  auto left_track = std::make_tuple(feat_left.prev_, feat_left.curr_);
  //  auto right_track = std::make_tuple(feat_right.prev_, feat_right.curr_);
  //  //std::cout << feat_left.prev_ << "\n";
  //  //std::cout << feat_left.curr_ << "\n";
  //  //std::cout << feat_right.prev_ << "\n";
  //  //std::cout << feat_right.curr_ << "\n";
  //  pts_left.push_back(left_track);
  //  pts_right.push_back(right_track);
  //}

  for (int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo feat_left = tracker.featureLeft(i);
    track::FeatureInfo feat_right = tracker.featureRight(i);
    age.push_back(feat_left.age_);
    auto left_track = std::make_tuple(feat_left.prev_, feat_left.curr_);
    auto right_track = std::make_tuple(feat_right.prev_, feat_right.curr_);
    pts_left.push_back(left_track);
    pts_right.push_back(right_track);
  }

  left_tracks_.push_back(pts_left);
  right_tracks_.push_back(pts_right);
  age_.push_back(age);
}

void CFMSolver::Solve() {
  ceres::Problem problem;
  // add residuals
  for (size_t i = 0; i < left_tracks_.size(); i++) {
    for (size_t j = 0; j < left_tracks_[i].size(); j++) {
      if (age_[i][j] < 1) continue;
      const core::Point& left_prev = std::get<0>(left_tracks_[i][j]);
      const core::Point& left_curr = std::get<1>(left_tracks_[i][j]);
      const core::Point& right_prev = std::get<0>(right_tracks_[i][j]);
      const core::Point& right_curr = std::get<1>(right_tracks_[i][j]);
      double disp = left_prev.x_ - right_prev.x_;
      if(disp > 0.01) {
        //ceres::LossFunction* loss_func = new ceres::CauchyLoss(loss_scale_);
        // TODO
        ceres::LossFunction* loss_func = new ceres::CauchyLoss(0.001);
        //ceres::LossFunction* loss_func = nullptr;

        ceres::CostFunction* cost = ReprojectionErrorResidual::Create(i, j,
            left_prev, left_curr, right_prev, right_curr, egomotion_translation_[i], egomotion_rotation_[i]);
        problem.AddResidualBlock(cost, loss_func, f1_, pp1_, k1_, f2_, pp2_, k2_,
                                 rot_, trans_);

        //ceres::CostFunction* cost = DistortionOnlyLoss::Create(i, j, left_prev, left_curr,
        //    right_prev, right_curr, f1_, f2_, pp1_, pp2_, rot_, trans_, egomotion_rotation_[i],
        //    egomotion_translation_[i]);
        //problem.AddResidualBlock(cost, loss_func, k1_, k2_);
      }
    }
  }
  //problem.SetParameterLowerBound(k1_, 0, 0.0);
  //problem.SetParameterLowerBound(k2_, 0, 0.0);

  //ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
  //problem.SetParameterization(rot_, quaternion_parameterization);
  //for (int i = 0; i < 2; i++) {
  //  problem.SetParameterLowerBound(f1_, i, 1.0);
  //  problem.SetParameterUpperBound(f1_, i, 1500.0);
  //  problem.SetParameterLowerBound(f2_, i, 1.0);
  //  problem.SetParameterUpperBound(f2_, i, 1500.0);
  //}


  //double lower_bound = -0.000001;
  //double upper_bound = 0.000001;
  //int central_bin = num_bins_ / 2;
  //problem.SetParameterLowerBound(&left_dx_[central_bin], 0, lower_bound);
  //problem.SetParameterLowerBound(&left_dy_[central_bin], 0, lower_bound);
  //problem.SetParameterLowerBound(&right_dx_[central_bin], 0, lower_bound);
  //problem.SetParameterLowerBound(&right_dy_[central_bin], 0, lower_bound);
  //problem.SetParameterUpperBound(&left_dx_[central_bin], 0, upper_bound);
  //problem.SetParameterUpperBound(&left_dy_[central_bin], 0, upper_bound);
  //problem.SetParameterUpperBound(&right_dx_[central_bin], 0, upper_bound);
  //problem.SetParameterUpperBound(&right_dy_[central_bin], 0, upper_bound);

  //for (int i = 0; i < num_bins_; i++) {
  //  problem.SetParameterLowerBound(&left_dx_[i], 0, -2.0);
  //  //problem.SetParameterLowerBound(left_dx_, i, -2.0);
  //  //problem.SetParameterLowerBound(left_dy_, i, -2.0);
  //  //problem.SetParameterLowerBound(right_dx_, i, -2.0);
  //  //problem.SetParameterLowerBound(right_dy_, i, -2.0);
  //  //problem.SetParameterUpperBound(left_dx_, i, 2.0);
  //  //problem.SetParameterUpperBound(left_dy_, i, 2.0);
  //  //problem.SetParameterUpperBound(right_dx_, i, 2.0);
  //  //problem.SetParameterUpperBound(right_dy_, i, 2.0);
  //}

  // Run the solver!
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  //options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
  options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
  options.line_search_direction_type = ceres::LineSearchDirectionType::STEEPEST_DESCENT;
  //options.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
  options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;

  options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
  //options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
  //options.check_gradients = true;

  //options.initial_trust_region_radius = 1e1; //1e4
  //options.max_trust_region_radius = 1e16;

  //options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
  //options.line_search_direction_type = ceres::LineSearchDirectionType::BFGS;

  //options.line_search_direction_type = ceres::LineSearchDirectionType::STEEPEST_DESCENT;
  //options.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
  //options.line_search_direction_type = ceres::LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT;
  //options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
  //options.step_size
  //options.line_se
  //options.parameter_tolerance = 1e-18;
  //options.function_tolerance = 1e-18;
  //options.gradient_tolerance = 1e-18;
  //options.use_nonmonotonic_steps = true;
  //options.max_consecutive_nonmonotonic_steps = 5;
  options.max_num_iterations = 200;
  options.num_threads = 1;
  options.num_linear_solver_threads = 1;
  //options.num_threads = 12;
  //options.num_linear_solver_threads = 1;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n\n--Full report--\n";
  std::cout << summary.FullReport() << "\n";

  std::cout << "--LEFT camera--\n";
  std::cout << "fx = " << f1_[0] << "\n";
  std::cout << "fy = " << f1_[1] << "\n";
  std::cout << "cx = " << pp1_[0] << "\n";
  std::cout << "cy = " << pp1_[1] << "\n";
  for (int i = 0; i < kNumK; i++)
    std::cout << "k" + std::to_string(i+1) + " = " << k1_[i] << "\n";

  std::cout << "--RIGHT camera--\n";
  std::cout << "fx = " << f2_[0] << "\n";
  std::cout << "fy = " << f2_[1] << "\n";
  std::cout << "cx = " << pp2_[0] << "\n";
  std::cout << "cy = " << pp2_[1] << "\n";
  for (int i = 0; i < kNumK; i++)
    std::cout << "k" + std::to_string(i+1) + " = " << k2_[i] << "\n";

  std::cout << "\nR = ";
  for (int i = 0; i < 4; i++)
    std::cout << rot_[i] << ", ";
  std::cout << "\nT = ";
  for (int i = 0; i < 3; i++)
    std::cout << trans_[i] << ", ";
  std::cout << "\n";
}


}
