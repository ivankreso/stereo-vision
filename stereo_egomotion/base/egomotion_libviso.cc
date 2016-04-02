#include "egomotion_libviso.h"

#include <random>
#include <unordered_set>
#include <omp.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "matrix.h"


namespace egomotion
{

namespace
{

std::vector<int> getRandomSample(std::uniform_int_distribution<int>& udist,
                                 std::mt19937& rng, size_t N)
{
  std::unordered_set<int> set;
  std::vector<int> sample;
  while(sample.size() < N) {
    int rnum = udist(rng);
    if (set.find(rnum) == set.end()) {
      set.insert(rnum);
      sample.push_back(rnum);
    }
  }
  return sample;
}

void TransformationVectorToMatrix(const std::vector<double>& tr, Eigen::Matrix4d& Rt)
{
  // extract parameters
  double rx = tr[0];
  double ry = tr[1];
  double rz = tr[2];
  double tx = tr[3];
  double ty = tr[4];
  double tz = tr[5];
  // precompute sine/cosine
  double sx = std::sin(rx);
  double cx = std::cos(rx);
  double sy = std::sin(ry);
  double cy = std::cos(ry);
  double sz = std::sin(rz);
  double cz = std::cos(rz);

  // compute transformation
  Rt(0,0) = +cy*cz;            Rt(0,1) = -cy*sz;          
  Rt(1,0) = +sx*sy*cz+cx*sz;   Rt(1,1) = -sx*sy*sz+cx*cz; 
  Rt(2,0) = -cx*sy*cz+sx*sz;   Rt(2,1) = +cx*sy*sz+sx*cz; 
  Rt(3,0) = 0;                 Rt(3,1) = 0;               

  Rt(0,2) = +sy;               Rt(0,3) = tx;
  Rt(1,2) = -sx*cy;            Rt(1,3) = ty;
  Rt(2,2) = +cx*cy;            Rt(2,3) = tz;
  Rt(3,2) = 0;                 Rt(3,3) = 1;
}

void TransformationVectorToMatrix(const std::vector<double>& tr, cv::Mat& Rt)
{
  // extract parameters
  double rx = tr[0];
  double ry = tr[1];
  double rz = tr[2];
  double tx = tr[3];
  double ty = tr[4];
  double tz = tr[5];
  // precompute sine/cosine
  double sx = std::sin(rx);
  double cx = std::cos(rx);
  double sy = std::sin(ry);
  double cy = std::cos(ry);
  double sz = std::sin(rz);
  double cz = std::cos(rz);

  // compute transformation
  Rt.create(4, 4, CV_64F);

  Rt.at<double>(0,0) = +cy*cz;            Rt.at<double>(0,1) = -cy*sz;          
  Rt.at<double>(1,0) = +sx*sy*cz+cx*sz;   Rt.at<double>(1,1) = -sx*sy*sz+cx*cz; 
  Rt.at<double>(2,0) = -cx*sy*cz+sx*sz;   Rt.at<double>(2,1) = +cx*sy*sz+sx*cz; 
  Rt.at<double>(3,0) = 0;                 Rt.at<double>(3,1) = 0;               

  Rt.at<double>(0,2) = +sy;               Rt.at<double>(0,3) = tx;
  Rt.at<double>(1,2) = -sx*cy;            Rt.at<double>(1,3) = ty;
  Rt.at<double>(2,2) = +cx*cy;            Rt.at<double>(2,3) = tz;
  Rt.at<double>(3,2) = 0;                 Rt.at<double>(3,3) = 1;
}

libviso::Matrix TransformationVectorToLibvisoMatrix(std::vector<double> tr)
{
  // extract parameters
  double rx = tr[0];
  double ry = tr[1];
  double rz = tr[2];
  double tx = tr[3];
  double ty = tr[4];
  double tz = tr[5];

  // precompute sine/cosine
  double sx = std::sin(rx);
  double cx = std::cos(rx);
  double sy = std::sin(ry);
  double cy = std::cos(ry);
  double sz = std::sin(rz);
  double cz = std::cos(rz);

  // compute transformation
  libviso::Matrix Tr(4,4);
  Tr.val[0][0] = +cy*cz;          Tr.val[0][1] = -cy*sz;          Tr.val[0][2] = +sy;    Tr.val[0][3] = tx;
  Tr.val[1][0] = +sx*sy*cz+cx*sz; Tr.val[1][1] = -sx*sy*sz+cx*cz; Tr.val[1][2] = -sx*cy; Tr.val[1][3] = ty;
  Tr.val[2][0] = -cx*sy*cz+sx*sz; Tr.val[2][1] = +cx*sy*sz+sx*cz; Tr.val[2][2] = +cx*cy; Tr.val[2][3] = tz;
  Tr.val[3][0] = 0;               Tr.val[3][1] = 0;               Tr.val[3][2] = 0;      Tr.val[3][3] = 1;
  return Tr;
}

void DrawRansacSample(std::vector<int>& sample, std::vector<EgomotionLibviso::StereoMatch>& tracks,
                      cv::Mat img_lp)
{
  cv::Mat disp_lp;
  cv::cvtColor(img_lp, disp_lp, cv::COLOR_GRAY2RGB);
  cv::Scalar color_pt(0,0,255);

  for (int i = 0; i < (int)sample.size(); i++) {
    cv::Point cvpt;
    cvpt.x = tracks[sample[i]].u1p;
    cvpt.y = tracks[sample[i]].v1p;
    cv::circle(disp_lp, cvpt, 2, color_pt, -1);
  }
  cv::imshow("RANSAC winner",  disp_lp);
  //cv::waitKey(0);
}

}

EgomotionLibviso::EgomotionLibviso(parameters params) : params_(params)
{
  use_deformation_map_ = false;
}

//EgomotionLibviso::EgomotionLibviso(parameters params, std::string weights_filename,
//                                           int img_rows, int img_cols)
//                                           : params_(params), img_rows_(img_rows), img_cols_(img_cols)
//{
//  std::cout << weights_filename << "\n";
//  cv::FileStorage mat_file(weights_filename, cv::FileStorage::READ);
//  mat_file["variance_matrix"] >> weights_mat_;
//  bin_width_ = (double)img_cols / weights_mat_.cols;
//  bin_height_ = (double)img_rows / weights_mat_.rows;
//}

EgomotionLibviso::EgomotionLibviso(parameters params,
                                   std::string deformation_params_fname,
                                   int img_rows, int img_cols)
                                   : params_(params), img_rows_(img_rows), img_cols_(img_cols)
{
  cv::FileStorage mat_file(deformation_params_fname, cv::FileStorage::READ);
  if (!mat_file.isOpened()) {
    std::cout << "Deformation file missing!\n";
    throw 1;
  }
  mat_file["left_dx"] >> left_dx_;
  mat_file["left_dy"] >> left_dy_;
  mat_file["right_dx"] >> right_dx_;
  mat_file["right_dy"] >> right_dy_;
  use_deformation_map_ = true;

  cell_width_ = (double)img_cols / left_dx_.cols;
  cell_height_ = (double)img_rows / left_dx_.rows;

  ComputeCellCenters();
}

void EgomotionLibviso::ComputeCellCenters()
{
  int rows = left_dx_.rows;
  int cols = left_dx_.cols;
  cell_centers_x_ = cv::Mat::zeros(rows, cols, CV_64F);
  cell_centers_y_ = cv::Mat::zeros(rows, cols, CV_64F);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cell_centers_x_.at<double>(i,j) = (j * cell_width_) + (cell_width_ / 2.0);
      cell_centers_y_.at<double>(i,j) = (i * cell_height_) + (cell_height_ / 2.0);
    }
  }
  //std::cout << cell_centers_x_ << "\n\n" << cell_centers_y_ << "\n";
}

void EgomotionLibviso::GetPointDeformation(const core::Point& pt, const cv::Mat& def_x,
                                           const cv::Mat& def_y, double& dx, double& dy)
{
  // number of rows/cols in interpolation grid is smaller by 1
  int rows = left_dx_.rows - 1;
  int cols = left_dx_.cols - 1;
  double half_width = cell_width_ / 2.0;
  double half_height = cell_height_ / 2.0;
  int real_row = static_cast<int>(pt.y_ / cell_height_);
  int real_col = static_cast<int>(pt.x_ / cell_width_);
  int row = static_cast<int>(std::floor((pt.y_ - half_height) / cell_height_));
  int col = static_cast<int>(std::floor((pt.x_ - half_width) / cell_width_));
  double cell_x = 0.0, cell_y = 0.0;
  if (row >= 0)
    cell_x = (pt.x_ - half_width) - (col * cell_width_);
  if (col >= 0)
    cell_y = (pt.y_ - half_height) - (row * cell_height_);

  // compute bilinear interpolation
  if (row >= 0 && row < rows && col >= 0 && col < cols) {
    assert(cell_x >= 0.0 && cell_x <= cell_width_);
    assert(cell_y >= 0.0 && cell_y <= cell_height_);
    InterpolateBilinear(def_x, row, col, cell_x, cell_y, dx);
    InterpolateBilinear(def_y, row, col, cell_x, cell_y, dy);
  }
  // compute left-right liner interpolation on horizontal edges
  else if ((row < 0 || row >= rows) && (col >= 0 || col < cols)) {
    assert(cell_x >= 0.0 && cell_x <= cell_width_);
    double q1 = def_x.at<double>(real_row, col);
    double q2 = def_x.at<double>(real_row, col+1);
    InterpolateLinear(q1, q2, cell_x, cell_width_, dx);
    q1 = def_y.at<double>(real_row, col);
    q2 = def_y.at<double>(real_row, col+1);
    InterpolateLinear(q1, q2, cell_x, cell_width_, dy);
  }
  // compute up-down linear interpolation on vertical edges
  else if ((row >= 0 || row < rows) && (col < 0 || col >= cols)) {
    assert(cell_y >= 0.0 && cell_y <= cell_height_);
    double q1 = def_x.at<double>(row, real_col);
    double q2 = def_x.at<double>(row+1, real_col);
    InterpolateLinear(q1, q2, cell_y, cell_height_, dx);
    q1 = def_y.at<double>(row, real_col);
    q2 = def_y.at<double>(row+1, real_col);
    InterpolateLinear(q1, q2, cell_y, cell_height_, dy);
  }
  // we can't interpolate on corners
  else {
    dx = left_dx_.at<double>(real_row, real_col);
    dy = left_dy_.at<double>(real_row, real_col);
  }
}

void EgomotionLibviso::GetTracksFromStereoTracker(track::StereoTrackerBase& tracker,
                                                      std::vector<StereoMatch>& tracks,
                                                      std::vector<int>& active_tracks)
{
  tracks.clear();
  active_tracks.clear();
  int feats_num = tracker.countFeatures();
  EgomotionLibviso::StereoMatch match;
  for (int i = 0; i < feats_num; i++) {
    track::FeatureInfo feat_left = tracker.featureLeft(i);
    track::FeatureInfo feat_right = tracker.featureRight(i);
    if (feat_left.age_ > 0) {
      if (!use_deformation_map_) {
        match.u1p = feat_left.prev_.x_;
        match.v1p = feat_left.prev_.y_;
        match.u1c = feat_left.curr_.x_;
        match.v1c = feat_left.curr_.y_;
        match.u2p = feat_right.prev_.x_;
        match.v2p = feat_right.prev_.y_;
        match.u2c = feat_right.curr_.x_;
        match.v2c = feat_right.curr_.y_;
      }
      // TODO: try combining triangulation in prev and curr
      //if (!use_deformation_map_) {
      //  match.u1c = feat_left.prev_.x_;
      //  match.v1c = feat_left.prev_.y_;
      //  match.u1p = feat_left.curr_.x_;
      //  match.v1p = feat_left.curr_.y_;
      //  match.u2c = feat_right.prev_.x_;
      //  match.v2c = feat_right.prev_.y_;
      //  match.u2p = feat_right.curr_.x_;
      //  match.v2p = feat_right.curr_.y_;
      //}
      // with interpolation
      else {
        double dx, dy;
        GetPointDeformation(feat_left.prev_, left_dx_, left_dy_, dx, dy);
        match.u1p = feat_left.prev_.x_ + dx;
        match.v1p = feat_left.prev_.y_ + dy;
        //std::cout << "Before = \n" << feat_left.prev_ << "\n";
        //std::cout << "After = \n" << match.u1p << " -- " << match.v1p << "\n";

        GetPointDeformation(feat_left.curr_, left_dx_, left_dy_, dx, dy);
        match.u1c = feat_left.curr_.x_ + dx;
        match.v1c = feat_left.curr_.y_ + dy;

        GetPointDeformation(feat_right.prev_, right_dx_, right_dy_, dx, dy);
        match.u2p = feat_right.prev_.x_ + dx;
        match.v2p = feat_right.prev_.y_ + dy;

        GetPointDeformation(feat_right.curr_, right_dx_, right_dy_, dx, dy);
        match.u2c = feat_right.curr_.x_ + dx;
        match.v2c = feat_right.curr_.y_ + dy;
      }
      //// without interpolation
      //else {
      //  int row, col;
      //  GetPointCell(feat_left.prev_, row, col);
      //  match.u1p = feat_left.prev_.x_ + left_dx_.at<double>(row, col);
      //  match.v1p = feat_left.prev_.y_ + left_dy_.at<double>(row, col);
      //  //std::cout << "Before = \n" << feat_left.prev_ << "\n";
      //  //std::cout << "After = \n" << match.u1p << " -- " << match.v1p << "\n";

      //  GetPointCell(feat_left.curr_, row, col);
      //  match.u1c = feat_left.curr_.x_ + left_dx_.at<double>(row, col);
      //  match.v1c = feat_left.curr_.y_ + left_dy_.at<double>(row, col);

      //  GetPointCell(feat_right.prev_, row, col);
      //  match.u2p = feat_right.prev_.x_ + right_dx_.at<double>(row, col);
      //  match.v2p = feat_right.prev_.y_ + right_dy_.at<double>(row, col);

      //  GetPointCell(feat_right.curr_, row, col);
      //  match.u2c = feat_right.curr_.x_ + right_dx_.at<double>(row, col);
      //  match.v2c = feat_right.curr_.y_ + right_dy_.at<double>(row, col);
      //}

      assert(!(std::isnan(match.u1p) || std::isnan(match.v1p) || std::isnan(match.u1c)
             || std::isnan(match.v1c) || std::isnan(match.u2p) || std::isnan(match.v2p)
             || std::isnan(match.u2c) || std::isnan(match.v2c)));

      tracks.push_back(match);
      active_tracks.push_back(i);
    }
  }
}

std::vector<double> EgomotionLibviso::estimateMotion(std::vector<StereoMatch>& tracks)
{
  // return value
  bool success = true;
  // get number of matches
  int N  = tracks.size();
  if (N < 6)
    return std::vector<double>();

  // allocate dynamic memory
  X          = new double[N];
  Y          = new double[N];
  Z          = new double[N];
  W          = new double[N];
  //double* D  = new double[N];

  // project matches of previous image into 3d
  for (int32_t i = 0; i < N; i++) {
    double d = std::max(tracks[i].u1p - tracks[i].u2p, 0.01f);
    //double d = p_matched[i].u1p - p_matched[i].u2p;
    //std::cout << d << ": " << p_matched[i].u1p << " - " << p_matched[i].u2p << "\n";
    if (d <= 0.0) {
      std::cout << "[EgomotionLibviso] zero/negative disp: " << d << " -> ";
      std::cout << tracks[i].u1p << " - " << tracks[i].u2p << "\n";
      throw 1;
    }
    //else if (d < 1.0) cout << "[EgomotionLibviso] small disp: " << d << "\n";

    //double d = max(p_matched[i].u1p - p_matched[i].u2p, 0.001f);
    //d = std::max(d, 0.0001);
    X[i] = (tracks[i].u1p - params_.calib.cu) * params_.base / d;
    Y[i] = (tracks[i].v1p - params_.calib.cv) * params_.base / d;
    Z[i] = params_.calib.f * params_.base / d;
    //D[i] = d;
    if (params_.reweighting) {
      if (!weights_mat_.empty()) {
        int r = tracks[i].v1p / cell_height_;
        int c = tracks[i].u1p / cell_width_;
        double variance = weights_mat_.at<double>(r,c);
        // TODO
        //W[i] = 1.0 / variance;
        W[i] = 1.0 / std::sqrt(variance);
      }
      else {
        //W[i] = 1.0/(fabs(p_observe[4*i+0] - params_.calib.cu) / fabs(params_.calib.cu) + 0.05);
        W[i] = 1.0/(fabs(tracks[i].u1c - params_.calib.cu) / fabs(params_.calib.cu) + 0.05);
      }
    }
  }
  // mark all observations active
  std::vector<int> active_all;
  for (int i = 0; i < (int)tracks.size(); i++)
    active_all.push_back(i);


  std::vector<std::vector<int>> active;
  std::vector<std::vector<int>> iter_inliers;
  std::vector<std::vector<double>> tr_delta;
  active.resize(params_.ransac_iters);
  iter_inliers.resize(params_.ransac_iters);
  tr_delta.resize(params_.ransac_iters);
  // initial RANSAC estimate
  //omp_set_num_threads(1);
  #pragma omp parallel
  {
  double* J          = new double[4*N*6];   // jacobian
  double* p_predict  = new double[4*N];     // predicted 2d points
  double* p_observe  = new double[4*N];     // observed 2d points
  double* p_residual = new double[4*N];     // residuals (p_residual=p_observe-p_predict)

  std::uniform_int_distribution<int> udist(0, N-1);
  //rng_type rng(clock() + std::this_thread::get_id().hash());
  std::mt19937 rng(int(time(NULL)) ^ omp_get_thread_num());
  //rng.seed(seedval);
  #pragma omp for
  for (int32_t k = 0; k < params_.ransac_iters; k++) {
    // draw random sample set
    active[k] = getRandomSample(udist, rng, 3);
    //bool good_pick = false;
    //while(!good_pick) {
    //  active[k] = getRandomSample(udist, rng, 3);
    //  for (int idx : active[k])
    //    if (D[idx] > 5.0) {
    //      good_pick = true;
    //      break;
    //    }
    //}

    //int thread_num = omp_get_thread_num();
    //for (int num : active[k])
    //printf("Iter = %d - Thread = %d - Random num = %d\n", k, thread_num, active[k][0]);

    // clear parameter vector
    tr_delta[k].assign(6, 0.0);
    // minimize reprojection errors
    ResultState result = UPDATED;
    int32_t iter = 0;
    while(result == UPDATED) {
      result = updateParameters(p_observe, p_predict, p_residual, J, tracks, active[k], tr_delta[k], 1, 1e-6);
      if (iter++ > 20 || result == CONVERGED)
        break;
    }
    if (result != FAILED)
      iter_inliers[k] = getInliers(p_observe, p_predict, p_residual, J, tracks, tr_delta[k], active_all);
  }

  delete[] J;
  delete[] p_predict;
  delete[] p_observe;
  delete[] p_residual;
  }

  int best_iter = 0;
  int most_inliers = iter_inliers[0].size();
  for (int i = 1; i < (int)iter_inliers.size(); i++) {
    if ((int)iter_inliers[i].size() > most_inliers) {
      best_iter = i;
      most_inliers = iter_inliers[i].size();
    }
  }
  printf("[EgomotionLibviso]: RANSAC found most inliers in iter %d / %d\n",
         best_iter, params_.ransac_iters);

  //DrawRansacSample(active[best_iter], tracks, img_left_prev_);

  // final optimization (refinement)
  double* J          = new double[4*N*6];   // jacobian
  double* p_predict  = new double[4*N];     // predicted 2d points
  double* p_observe  = new double[4*N];     // observed 2d points
  double* p_residual = new double[4*N];     // residuals (p_residual=p_observe-p_predict)
  //printf("Final optimization\n");
  inliers_ = iter_inliers[best_iter];
  if (inliers_.size() >= 6) {
    int32_t iter = 0;
    ResultState result = UPDATED;
    while(result == UPDATED) {
      // orig
      result = updateParameters(p_observe, p_predict, p_residual, J, tracks, inliers_,
                                tr_delta[best_iter], 1, 1e-8);
      // mine - last resort
      //result = updateParameters(p_matched,inliers,tr_delta,1,1e-7);
      // orig
      //if (iter++ > 100 || result==CONVERGED)
      // mine
      if (iter++ > 500 || result == CONVERGED) {
        printf("[libviso]: Newton method final iters: %d\n", iter+1);
        break;
      }
    }
    // not converged
    if (result != CONVERGED) {
      success = false;
      inliers_.clear();
    }
    // not enough inliers
  } else {
    success = false;
      inliers_.clear();
  }
  // release dynamic memory
  delete[] X;
  delete[] Y;
  delete[] Z;
  //delete[] D;
  delete[] W;

  delete[] J;
  delete[] p_predict;
  delete[] p_observe;
  delete[] p_residual;

  // parameter estimate succeeded?
  if (success) return tr_delta[best_iter];
  else         return std::vector<double>();
}

std::vector<int> EgomotionLibviso::getInliers(double* p_observe,
                                              double* p_predict,
                                              double* p_residual,
                                              double* J,
                                              std::vector<StereoMatch> &tracks,
                                              std::vector<double> &tr,
                                              std::vector<int>& active)
{
  // extract observations and compute predictions
  computeObservations(p_observe, tracks, active);
  computeResidualsAndJacobian(p_observe, p_predict, p_residual, J, tr, active);

  // compute inliers
  std::vector<int32_t> inliers;
  for (int32_t i = 0; i < (int32_t)tracks.size(); i++)
    if (pow(p_observe[4*i+0] - p_predict[4*i+0],2) + pow(p_observe[4*i+1] - p_predict[4*i+1],2) +
       pow(p_observe[4*i+2] - p_predict[4*i+2],2) + pow(p_observe[4*i+3] - p_predict[4*i+3],2)
       < params_.inlier_threshold * params_.inlier_threshold)
      inliers.push_back(i);
  return inliers;
}

EgomotionLibviso::ResultState EgomotionLibviso::updateParameters(
                                                  double* p_observe,
                                                  double* p_predict,
                                                  double* p_residual,
                                                  double* J,
                                                  std::vector<StereoMatch> &tracks,
                                                  std::vector<int32_t>& active,
                                                  std::vector<double>& tr,
                                                  double step_size, double eps) {

  // we need at least 3 observations
  if (active.size() < 3)
    return FAILED;

  // extract observations and compute predictions
  computeObservations(p_observe, tracks, active);
  computeResidualsAndJacobian(p_observe, p_predict, p_residual, J, tr, active);

  // init
  libviso::Matrix A(6,6);
  libviso::Matrix B(6,1);

  // fill matrices A and B
  for (int32_t m=0; m<6; m++) {
    for (int32_t n=0; n<6; n++) {
      double a = 0;
      for (int32_t i=0; i<4*(int32_t)active.size(); i++) {
        a += J[i*6+m]*J[i*6+n];
      }
      //if (std::isnan(a)) {
      //  printf("NAN\n");
      //  throw "Error\n";
      //}
      A.val[m][n] = a;
    }
    double b = 0;
    for (int32_t i=0; i<4*(int32_t)active.size(); i++) {
      b += J[i*6+m]*(p_residual[i]);
    }
    //if (std::isnan(b)) {
    //  printf("NAN\n");
    //  throw "Error\n";
    //}
    B.val[m][0] = b;
  }

  // perform elimination
  if (B.solve(A)) {
    bool converged = true;
    for (int32_t m=0; m<6; m++) {
      tr[m] += step_size*B.val[m][0];
      //printf("%e\n", fabs(B.val[m][0]));
      if (fabs(B.val[m][0])>eps)
        converged = false;
    }
    if (converged)
      return CONVERGED;
    else
      return UPDATED;
  } else {
    return FAILED;
  }
}

void EgomotionLibviso::computeObservations(double* p_observe, std::vector<StereoMatch>& tracks,
                                               std::vector<int> &active)
{

  // set all observations
  for (int i = 0; i < (int32_t)active.size(); i++) {
    p_observe[4*i+0] = tracks[active[i]].u1c; // u1
    p_observe[4*i+1] = tracks[active[i]].v1c; // v1
    p_observe[4*i+2] = tracks[active[i]].u2c; // u2
    p_observe[4*i+3] = tracks[active[i]].v2c; // v2
  }
}

void EgomotionLibviso::computeResidualsAndJacobian(double* p_observe,
                                                       double* p_predict,
                                                       double* p_residual,
                                                       double* J,
                                                       std::vector<double>& tr,
                                                       std::vector<int>& active)
{
  // extract motion parameters
  double rx = tr[0]; double ry = tr[1]; double rz = tr[2];
  double tx = tr[3]; double ty = tr[4]; double tz = tr[5];

  // precompute sine/cosine
  double sx = sin(rx); double cx = cos(rx); double sy = sin(ry);
  double cy = cos(ry); double sz = sin(rz); double cz = cos(rz);

  // compute rotation matrix and derivatives
  double r00    = +cy*cz;          double r01    = -cy*sz;          double r02    = +sy;
  double r10    = +sx*sy*cz+cx*sz; double r11    = -sx*sy*sz+cx*cz; double r12    = -sx*cy;
  double r20    = -cx*sy*cz+sx*sz; double r21    = +cx*sy*sz+sx*cz; double r22    = +cx*cy;
  double rdrx10 = +cx*sy*cz-sx*sz; double rdrx11 = -cx*sy*sz-sx*cz; double rdrx12 = -cx*cy;
  double rdrx20 = +sx*sy*cz+cx*sz; double rdrx21 = -sx*sy*sz+cx*cz; double rdrx22 = -sx*cy;
  double rdry00 = -sy*cz;          double rdry01 = +sy*sz;          double rdry02 = +cy;
  double rdry10 = +sx*cy*cz;       double rdry11 = -sx*cy*sz;       double rdry12 = +sx*sy;
  double rdry20 = -cx*cy*cz;       double rdry21 = +cx*cy*sz;       double rdry22 = -cx*sy;
  double rdrz00 = -cy*sz;          double rdrz01 = -cy*cz;
  double rdrz10 = -sx*sy*sz+cx*cz; double rdrz11 = -sx*sy*cz-cx*sz;
  double rdrz20 = +cx*sy*sz+sx*cz; double rdrz21 = +cx*sy*cz-sx*sz;

  // loop variables
  double X1p,Y1p,Z1p;
  double X1c,Y1c,Z1c,X2c;
  double X1cd,Y1cd,Z1cd;

  // for all observations do
  for (int32_t i = 0; i < (int32_t)active.size(); i++) {
    // get 3d point in previous coordinate system
    X1p = X[active[i]];
    Y1p = Y[active[i]];
    Z1p = Z[active[i]];

    // compute 3d point in current left coordinate system
    X1c = r00*X1p+r01*Y1p+r02*Z1p+tx;
    Y1c = r10*X1p+r11*Y1p+r12*Z1p+ty;
    Z1c = r20*X1p+r21*Y1p+r22*Z1p+tz;
    //if (std::isnan(Z1c))
    //  printf("Z1c NAN\n");

    // weighting
    double weight = 1.0;
    if (params_.reweighting) {
      // current left u
      //weight = 1.0/(fabs(p_observe[4*i+0] - params_.calib.cu) / fabs(params_.calib.cu) + 0.05);
      // TODO
      //weight = weight_list[active[i]];
      weight = W[active[i]];
      //weight = 1.0;
    }
    // TODO try to learn a constant
    //weight *= 0.8;
    //printf("weight = %f\n", weight);

    // compute 3d point in current right coordinate system
    X2c = X1c - params_.base;

    // for all paramters do
    for (int j = 0; j < 6; j++) {
      // derivatives of 3d pt. in curr. left coordinates wrt. param j
      switch (j) {
        case 0: X1cd = 0;
                Y1cd = rdrx10*X1p+rdrx11*Y1p+rdrx12*Z1p;
                Z1cd = rdrx20*X1p+rdrx21*Y1p+rdrx22*Z1p;
                break;
        case 1: X1cd = rdry00*X1p+rdry01*Y1p+rdry02*Z1p;
                Y1cd = rdry10*X1p+rdry11*Y1p+rdry12*Z1p;
                Z1cd = rdry20*X1p+rdry21*Y1p+rdry22*Z1p;
                break;
        case 2: X1cd = rdrz00*X1p+rdrz01*Y1p;
                Y1cd = rdrz10*X1p+rdrz11*Y1p;
                Z1cd = rdrz20*X1p+rdrz21*Y1p;
                break;
        case 3: X1cd = 1; Y1cd = 0; Z1cd = 0; break;
        case 4: X1cd = 0; Y1cd = 1; Z1cd = 0; break;
        case 5: X1cd = 0; Y1cd = 0; Z1cd = 1; break;
      }

      // TODO - increase weights for nearby points for j = translation
      //double disp = std::max(p_observe[4*i+0] - p_observe[4*i+2], 0.1);
      //std::cout << "DISP = " << disp << '\n';
      //double weight_trans = 1.0;
      //if (j > 2)
      //  //weight_trans = 1.0 * disp;
      //  weight_trans = 1.0 / std::max(10.0 - disp, 1.0);

      //double rx = tr[0];
      //double ry = tr[1];
      //double rz = tr[2];
      //double tx = tr[3];
      //double ty = tr[4];
      //double tz = tr[5];

      //J[(4*i+0)*6+j] = weight_trans * weight*params_.calib.f*(X1cd*Z1c-X1c*Z1cd)/(Z1c*Z1c); // left u'
      //J[(4*i+1)*6+j] = weight_trans * weight*params_.calib.f*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // left v'
      //J[(4*i+2)*6+j] = weight_trans * weight*params_.calib.f*(X1cd*Z1c-X2c*Z1cd)/(Z1c*Z1c); // right u'
      //J[(4*i+3)*6+j] = weight_trans * weight*params_.calib.f*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // right v'

      // set jacobian entries (project via K)
      J[(4*i+0)*6+j] = weight*params_.calib.f*(X1cd*Z1c-X1c*Z1cd)/(Z1c*Z1c); // left u'
      J[(4*i+1)*6+j] = weight*params_.calib.f*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // left v'
      J[(4*i+2)*6+j] = weight*params_.calib.f*(X1cd*Z1c-X2c*Z1cd)/(Z1c*Z1c); // right u'
      J[(4*i+3)*6+j] = weight*params_.calib.f*(Y1cd*Z1c-Y1c*Z1cd)/(Z1c*Z1c); // right v'
    }

    // set prediction (project via K)
    p_predict[4*i+0] = params_.calib.f*X1c/Z1c+params_.calib.cu; // left u
    p_predict[4*i+1] = params_.calib.f*Y1c/Z1c+params_.calib.cv; // left v
    p_predict[4*i+2] = params_.calib.f*X2c/Z1c+params_.calib.cu; // right u
    p_predict[4*i+3] = params_.calib.f*Y1c/Z1c+params_.calib.cv; // right v

    // set residuals
    p_residual[4*i+0] = weight*(p_observe[4*i+0]-p_predict[4*i+0]);
    p_residual[4*i+1] = weight*(p_observe[4*i+1]-p_predict[4*i+1]);
    p_residual[4*i+2] = weight*(p_observe[4*i+2]-p_predict[4*i+2]);
    p_residual[4*i+3] = weight*(p_observe[4*i+3]-p_predict[4*i+3]);
  }
}

void EgomotionLibviso::updateTrackerInliers(const std::vector<int>& active_tracks)
{
  std::vector<bool> dead_tracks(active_tracks.size(), true);
  tracker_inliers_.clear();
  tracker_outliers_.clear();
  for (size_t i = 0; i < inliers_.size(); i++)
    dead_tracks[inliers_[i]] = false;

  for (size_t i = 0; i < dead_tracks.size(); i++) {
    if (dead_tracks[i] == false)
      tracker_inliers_.push_back(active_tracks[i]);
    else
      tracker_outliers_.push_back(active_tracks[i]);
  }

  //for (size_t i = 0; i < inliers_.size(); i++) {
  //  int inlier_idx = active_tracks[inliers_[i]];
  //  tracker_inliers_.push_back(inlier_idx);
  //}
}

bool EgomotionLibviso::GetMotion(track::StereoTrackerBase& tracker, Eigen::Matrix4d& Rt) {
  // estimate motion
  std::vector<StereoMatch> tracks;
  std::vector<int> active_tracks;
  GetTracksFromStereoTracker(tracker, tracks, active_tracks);

  std::vector<double> tr_delta = estimateMotion(tracks);
  
  // on failure
  if (tr_delta.size() != 6)
    return false;

  updateTrackerInliers(active_tracks);
  // set transformation matrix (previous to current frame)
  TransformationVectorToMatrix(tr_delta, Rt);
  return true;
}

}


//{
//  // init sample and totalset
//  std::vector<int32_t> sample;
//  std::vector<int32_t> totalset;
//  
//  // create vector containing all indices
//  for (int32_t i=0; i<N; i++)
//    totalset.push_back(i);
//
//  // add num indices to current sample
//  sample.clear();
//  for (int32_t i=0; i<num; i++) {
//    int32_t j = rand()%totalset.size();
//    sample.push_back(totalset[j]);
//    totalset.erase(totalset.begin()+j);
//  }
//  
//  // return sample
//  return sample;
//}




  //// loop variables
  //vector<double> tr_delta;
  //// clear parameter vector
  //inliers.clear();
  //// initial RANSAC estimate
  //int32_t k;
  ////#pragma omp parallel for shared(tr_delta, inliers) private(k, tr_delta_curr) schedule(dynamic)
  //#pragma omp parallel for shared(tr_delta) private(k) schedule(dynamic)
  //for (k=0; k<param.ransac_iters; k++) {
  //  // draw random sample set
  //  vector<int32_t> active = getRandomSample(N,3);
  //  vector<double> tr_delta_curr(6);
  //  // clear parameter vector
  //  for (int32_t i=0; i<6; i++)
  //    tr_delta_curr[i] = 0;

  //  // minimize reprojection errors
  //  EgomotionLibviso::result result = UPDATED;
  //  int32_t iter=0;
  //  while (result==UPDATED) {
  //    result = updateParameters(p_matched,active,tr_delta_curr,1,1e-6);
  //    if (iter++ > 20 || result==CONVERGED)
  //      break;
  //  }

  //  // overwrite best parameters if we have more inliers
  //  if (result!=FAILED) {
  //    vector<int32_t> inliers_curr = getInlier(p_matched,tr_delta_curr);
  //    #pragma omp critical
  //    {
  //    if (inliers_curr.size()>inliers.size()) {
  //      inliers = inliers_curr;
  //      tr_delta = tr_delta_curr;
  //    }
  //    }
  //  }
  //}
