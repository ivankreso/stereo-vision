#ifndef OPTIMIZATION_CFM_SOLVER_
#define OPTIMIZATION_CFM_SOLVER_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "../../tracker/stereo/stereo_tracker_base.h"
#include "../../core/cpp_gems.h"

//#define DEBUG

namespace optim {

typedef ceres::Jet<double,25> JetType;
//typedef ceres::Jet<double,10> JetType;

static const int kNumK = 5;

class CFMSolver {
 public:
  CFMSolver(int img_rows, int img_cols, double loss_scale);
  ~CFMSolver();
  void UpdateTracks(const track::StereoTrackerBase& tracker, const cv::Mat& Rt);
  //void UpdateReverseTracks(const track::StereoTrackerBase& tracker, const cv::Mat& Rt);
  void Solve();

  void GenerateRandomPoints();
  // Factory to hide the construction of the CostFunction object from the client code.
  void AddCostResidual(const core::Point& left_prev, const core::Point& left_curr,
                       const core::Point& right_prev, const core::Point& right_curr,
                       const std::array<double,3>& cam_trans,
                       const std::array<double,4>& cam_rot,
                       ceres::Problem* ceres_problem);
 private:
  std::vector<std::vector<std::tuple<core::Point,core::Point>>> left_tracks_, right_tracks_;
  std::vector<std::vector<int>> age_;
  std::vector<cv::Mat> gt_rt_;
  std::vector<std::array<double,3>> egomotion_translation_;
  std::vector<std::array<double,4>> egomotion_rotation_;

  double f1_[2];
  double f2_[2];
  double pp1_[2];
  double pp2_[2];
  //double dc1_[2];
  //double dc2_[2];
  double k1_[kNumK];
  double k2_[kNumK];
  double init_k1_[kNumK];
  double init_k2_[kNumK];
  double rot_[4];
  double trans_[3];

  int img_rows_, img_cols_;

  double loss_scale_;
  int min_points_;
};

namespace {

template <typename T>
bool IsValid(const T& val) {
  if(ceres::IsInfinite(val) || ceres::IsNaN(val)) {
    //std::cout << "a = " << ((JetType)val).a << "\n";
    //std::cout << val << "\n";
    throw 1;
    return false;
  }
  return true;
}

template <typename T>
void Print(const T& val) {
  std::cout << ((JetType)val).a << "\n";
}

template <typename T>
std::string PrintPoint2D(const T pt[2]) {
  //std::cout << "[" << ((JetType)pt[0]).a << ", " << ((JetType)pt[1]).a << "]";
  return "[" + std::to_string(((JetType)pt[0]).a) + ", " + std::to_string(((JetType)pt[1]).a) + "]";
}
template <typename T>
std::string PrintPoint3D(const T pt[3]) {
  //std::cout << std::setprecision(12) << ((JetType)pt[0]).a << ", " << ((JetType)pt[1]).a << ", "
  //          << ((JetType)pt[2]).a << "\n";
  return "[" + std::to_string(((JetType)pt[0]).a) + ", " + std::to_string(((JetType)pt[1]).a) +
         ", " + std::to_string(((JetType)pt[2]).a) + "]";
}
template <typename T>
std::string GetVector4D(const T v[4]) {
  return "[" + std::to_string(((JetType)v[0]).a) + ", " + std::to_string(((JetType)v[1]).a) +
         ", " + std::to_string(((JetType)v[2]).a) + ", " + std::to_string(((JetType)v[3]).a) + "]";
}

template <typename T> inline
double GetValue(const T& val) {
  return ((JetType)val).a;
}

std::ostream& operator<<(std::ostream& os, const JetType& v) {
  os << v.a;
  return os;
}

//template <typename T> inline
//void UndistortPoint(const T* const dist_pt, const T* const k, T* const undist_pt) {
//  // compensate distortion iteratively
//  const T x0 = dist_pt[0];
//  const T y0 = dist_pt[1];
//  T x = x0;
//  T y = y0;
//  
//  int iters = 5;
//  for(int i = 0; i < iters; i++) {
//    std::cout << "iter = " << i << "\n";
//    Print(x);
//    Print(y);
//    T r2 = x*x + y*y;
//    T radial = (T(1.0) + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(T(1.0) + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
//    T tang_x = T(2.0)*k[2]*x*y + k[3]*(r2 + T(2.0)*x*x) + k[8]*r2+k[9]*r2*r2;
//    T tang_y = k[2]*(r2 + T(2.0)*y*y) + T(2.0)*k[3]*x*y + k[10]*r2+k[11]*r2*r2;
//    x = (x0 - tang_x) * radial;
//    y = (y0 - tang_y) * radial;
//    IsValid(x);
//    IsValid(y);
//  }
//  undist_pt[0] = x;
//  undist_pt[1] = y;
//}

//template <typename T> inline
//void UndistortPoint(const T dist_pt[2], const T k[kNumK], T undist_pt[2]) {
//  // compensate distortion iteratively
//  const T x0 = dist_pt[0];
//  const T y0 = dist_pt[1];
//  T x = x0;
//  T y = y0;
//  
//  int iters = 5;
//  for(int i = 0; i < iters; i++) {
//    //std::cout << "iter = " << i << "\n";
//    //Print(x);
//    //Print(y);
//    T r2 = x*x + y*y;
//    T radial = (T(1.0) + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
//    T tang_x = T(2.0)*k[2]*x*y + k[3]*(r2 + T(2.0)*x*x);
//    T tang_y = k[2]*(r2 + T(2.0)*y*y) + T(2.0)*k[3]*x*y;
//    x = (x0 - tang_x) / radial;
//    y = (y0 - tang_y) / radial;
//    IsValid(x);
//    IsValid(y);
//  }
//  undist_pt[0] = x;
//  undist_pt[1] = y;
//}

// from paper
//template <typename T> inline
//void DistortPoint(const T pt_u[2], const T k[kNumK], T pt_d[2]) {
//  const T& x = pt_u[0];
//  const T& y = pt_u[1];
//  T r2 = x*x + y*y;
//  T radial = (T(1.0) + (k[1]*r2 + k[0])*r2);
//  T tang_x = T(2.0)*k[2]*x*y + k[3]*(r2 + T(2.0)*x*x);
//  T tang_y = k[2]*(r2 + T(2.0)*y*y) + T(2.0)*k[3]*x*y;
//  pt_d[0] = x * radial + tang_x;
//  pt_d[1] = y * radial + tang_y;
//}
//template <typename T> inline
//void UndistortPoint(const T dist_pt[2], const T k[kNumK], T undist_pt[2]) {
//  T pt[2];
//  DistortPoint(dist_pt, k, pt);
//  const T r2 = dist_pt[0]*dist_pt[0] + dist_pt[1]*dist_pt[1];
//  const T s = T(4.0)*k[0]*r2 + T(6.0)*k[1]*r2*r2 + T(8.0)*k[2]*dist_pt[1] +
//              T(8.0)*k[3]*dist_pt[0] + T(1.0);
//  pt[0] = pt[0] / s;
//  pt[1] = pt[1] / s;
//  undist_pt[0] = dist_pt[0] - pt[0];
//  undist_pt[1] = dist_pt[1] - pt[1];
//}

template <typename T> inline
void DistortPoint(const T pt_u[2], const T k[kNumK], T pt_d[2]) {
  throw 1;
  const T& x = pt_u[0];
  const T& y = pt_u[1];
  T r2 = x*x + y*y;
  T radial = (T(1.0) + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
  T tang_x = T(2.0)*k[2]*x*y + k[3]*(r2 + T(2.0)*x*x);
  T tang_y = k[2]*(r2 + T(2.0)*y*y) + T(2.0)*k[3]*x*y;
  pt_d[0] = x * radial + tang_x;
  pt_d[1] = y * radial + tang_y;
}

template <typename T> inline
void UndistortPoint(const T pt_dist[2], const T k[kNumK], T pt_undist[2]) {
  const T& x = pt_dist[0];
  const T& y = pt_dist[1];
  T r2 = x*x + y*y;
  T radial = (T(1.0) + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
  //pt_undist[0] = x * radial;
  //pt_undist[1] = y * radial;
  T tang_x = T(2.0)*k[2]*x*y + k[3]*(r2 + T(2.0)*x*x);
  T tang_y = k[2]*(r2 + T(2.0)*y*y) + T(2.0)*k[3]*x*y;
  pt_undist[0] = x * radial + tang_x;
  pt_undist[1] = y * radial + tang_y;
}

template <typename T> inline
void Transform3DPoint(const T pt1[3], const T R[4], const T t[3], T pt2[3]) {
  ceres::UnitQuaternionRotatePoint(R, pt1, pt2);
  for (int i = 0; i < 3; i++)
    pt2[i] += t[i];
}

template <typename T> inline
void ProjectToCamera(const T pt3d[3], const T f[2], const T pp[2], const T k[kNumK], T pt2d[2]) {
  T pt_proj[2];
  for (int i = 0; i < 2; i++)
    pt_proj[i] = pt3d[i] / pt3d[2];
  DistortPoint(pt_proj, k, pt2d);
  for (int i = 0; i < 2; i++)
    pt2d[i] = f[i] * pt2d[i] + pp[i];
}
template <typename T> inline
void ProjectToCamera(const T pt3d[3], T pt2d[2]) {
  for (int i = 0; i < 2; i++)
    pt2d[i] = pt3d[i] / pt3d[2];
}

template <typename T> inline
void InvertQuaternion(const T q[4], T q_inv[4]) {
  q_inv[0] = q[0];
  for (int i = 1; i < 4; i++)
    q_inv[i] = -q[i];
}

template <typename T> inline
void GetUnitVector(const T pt2d[2], T pt3d[3]) {
  pt3d[0] = pt2d[0];
  pt3d[1] = pt2d[1];
  pt3d[2] = T(1.0);
  T norm = ceres::sqrt(pt3d[0]*pt3d[0] + pt3d[1]*pt3d[1] + pt3d[2]*pt3d[2]);
  for (int i = 0; i < 3; i++)
    pt3d[i] /= norm;
}

template<typename T> inline
T Dot(const T x[3], const T y[3]) {
  return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template<typename T> inline
double GetDist(const T x[2], const T y[2]) {
  return GetValue(ceres::sqrt((x[0] - y[0])*(x[0] - y[0]) +
                              (x[1] - y[1])*(x[1] - y[1])));
}


template <typename T>
bool ComputeReprojectionErrorResidual(
    const T* const f1,
    const T* const pp1,
    const T* const k1,
    const T* const f2,
    const T* const pp2,
    const T* const k2,
    const T* const rot,
    const T* const trans,
    const core::Point& left_prev,
    const core::Point& right_prev,
    const core::Point& left_curr,
    const core::Point& right_curr,
    const std::array<double,4>& egomotion_rot,
    const std::array<double,3>& egomotion_trans,
    T* out_residuals) {
  // left previous point
  T lp1[2];
  lp1[0] = T(left_prev.x_);
  lp1[1] = T(left_prev.y_);

  // left current point
  T lc1[2];
  lc1[0] = T(left_curr.x_);
  lc1[1] = T(left_curr.y_);

  // right previous point
  T rp1[2];
  rp1[0] = T(right_prev.x_);
  rp1[1] = T(right_prev.y_);

  // right current point
  T rc1[2];
  rc1[0] = T(right_curr.x_);
  rc1[1] = T(right_curr.y_);
  
  // 1. normalize coordinates
  for (int i = 0; i < 2; i++) {
    lp1[i] = (lp1[i] - pp1[i]) / f1[i];
    lc1[i] = (lc1[i] - pp1[i]) / f1[i];
    rp1[i] = (rp1[i] - pp2[i]) / f2[i];
    rc1[i] = (rc1[i] - pp2[i]) / f2[i];
  }
  // 2. undistort points
  // compensate distortion iteratively
  T lp[2], rp[2];
  UndistortPoint(lp1, k1, lp);
  UndistortPoint(rp1, k2, rp);
  T lc[2], rc[2];
  UndistortPoint(lc1, k1, lc);
  UndistortPoint(rc1, k2, rc);

  //T lp2[2], rp2[2], lc2[2], rc2[2];
  //DistortPoint(lp, k1, lp2);
  //DistortPoint(rp, k2, rp2);
  //DistortPoint(lc, k1, lc2);
  //DistortPoint(rc, k2, rc2);

  //T lp3[2];
  //UndistortPoint(lp2, k1, lp3);
  //std::cout << "LP: " << PrintPoint2D(lp1) << " -> " << PrintPoint2D(lp) << "\n";
  //std::cout << "RP: " << PrintPoint2D(rp1) << " -> " << PrintPoint2D(rp) << "\n";
  //std::cout << "LP: " << PrintPoint2D(lp1) << " == " << PrintPoint2D(lp2) << "\n";
  //std::cout << "RP: " << PrintPoint2D(rp1) << " == " << PrintPoint2D(rp2) << "\n";
  //std::cout << "LC: " << PrintPoint2D(lc1) << " -> " << PrintPoint2D(lc) << "\n";
  //std::cout << "RC: " << PrintPoint2D(rc1) << " -> " << PrintPoint2D(rc) << "\n";
  //std::cout << "LC: " << PrintPoint2D(lc1) << " == " << PrintPoint2D(lc2) << "\n";
  //std::cout << "RC: " << PrintPoint2D(rc1) << " == " << PrintPoint2D(rc2) << "\n";

  //duplo manji init cost
  //double thr = 1e-3;
  //double thr = 0.3e-3;
  //double thr = 1e-4;
  //if (GetDist(lp2, lp1) > thr || GetDist(lc2, lc1) > thr ||
  //    GetDist(rp2, rp1) > thr || GetDist(rc2, rc1) > thr) {
  //  //throw 1;
  //  //out_residuals[0] = T(0);
  //  //out_residuals[1] = T(0);
  //  //out_residuals[2] = T(0);
  //  //out_residuals[3] = T(0);
  //  //std::cout << "nogodd\n";
  //  return false;
  //}
  //else std::cout << "Is good\n";

  //for (int i = 0; i < 2; i++) {
  //  IsValid(lp[i]);
  //  //if(!ceres::IsNormal(rp[i])) throw 1;
  //  //if(!ceres::IsNormal(lc[i])) throw 1;
  //  //if(!ceres::IsNormal(rc[i])) throw 1;
  //}

  // 3. triangulate in previous frame
  //T C1[3];
  // C2 is trans
  //const T* const C2 = trans;
  //for (int i = 0; i < 3; i++)
  //  C1[i] = T(0.0);
  T r1[3];
  GetUnitVector(lp, r1);
  T r2[3], r2_right[3];
  // get r2 in right coordinate system
  GetUnitVector(rp, r2_right);
  // convert it to left coordinate system
  //std::cout << "Right to left motion = " << GetVector4D(rot) << " -- "
  //          << PrintPoint3D(trans) << "\n";
  ceres::UnitQuaternionRotatePoint(rot, r2_right, r2);
  //std::cout << "r1 = " << PrintPoint3D(r1) << "\n";
  //std::cout << "r2 = " << PrintPoint3D(r2_right) << " -> " << PrintPoint3D(r2) << "\n";
  T r12dot = Dot(r1, r2);
  T c2r2dot = Dot(trans, r2);
  T m1 = (Dot(trans, r1) - (c2r2dot * r12dot))
         / (T(1.0) - (r12dot * r12dot));
  T m2 = m1 * r12dot - c2r2dot;

  T Pp[3], P1[3], P2[3];
  T dP[3];
  for (int i = 0; i < 3; i++) {
    //P1[i] = C1[i] + (m1 * r1[i]);
    P1[i] = m1 * r1[i];
    P2[i] = trans[i] + (m2 * r2[i]);
    Pp[i] = (P1[i] + P2[i]) / T(2.0);
    dP[i] = P1[i] - P2[i];
  }
  //if (Dot(dP, r1) > 1e-10) throw 1;
  //if (Dot(dP, r2) > 1e-10) throw 1;
  //std::cout << "P1 = " << PrintPoint3D(P1) << "\n";
  //std::cout << "P2 = " << PrintPoint3D(P2) << "\n";
  //std::cout << "Pp = " << PrintPoint3D(Pp) << "\n";

  T motion_rot[4];
  for (int i = 0; i < 4; i++)
    motion_rot[i] = T(egomotion_rot[i]);
  T motion_trans[3];
  for (int i = 0; i < 3; i++)
    motion_trans[i] = T(egomotion_trans[i]);

  //const T* const motion_rot = (T*)egomotion_rot;

  //std::cout << "Apply camera motion = " << GetVector4D(motion_rot) << " -- "
  //          << PrintPoint3D(motion_trans) << "\n";
  T Pc_left[3];
  Transform3DPoint(Pp, motion_rot, motion_trans, Pc_left);
  //std::cout << "Pc_left = " << PrintPoint3D(Pc_left) << "\n";

  // convert to the right cam coordinate system
  T rot_inv[4];
  InvertQuaternion(rot, rot_inv);
  //std::cout << "Inverse rotation = " << GetVector4D(rot) << " -> " << GetVector4D(rot_inv) << "\n";
  T Pc_right_tmp[3];
  for (int i = 0 ; i < 3; i++)
    Pc_right_tmp[i] = Pc_left[i] - trans[i];
  T Pc_right[3];
  ceres::UnitQuaternionRotatePoint(rot_inv, Pc_right_tmp, Pc_right);

  ////std::cout << "Pc_right = " << PrintPoint3D(Pc_right) << "\n";
  //T predict_left[2], predict_right[2];
  //ProjectToCamera(Pc_left, f1, pp1, k1, predict_left);
  //ProjectToCamera(Pc_right, f2, pp2, k2, predict_right);

  ////std::cout << "prev = " << left_prev << " -- " << right_prev << "\n";
  ////std::cout << "curr = " << left_curr << " -- " << right_curr << "\n";
  ////std::cout << "pred_left = " << GetValue(predict_left[0]) << ", " << GetValue(predict_left[1]) << "\n";
  ////std::cout << "pred_right = " << GetValue(predict_right[0]) << ", " << GetValue(predict_right[1]) << "\n";

  //// Compute and return the error is the difference between the predicted and observed position
  //out_residuals[0] = predict_left[0] - T(left_curr.x_);
  //out_residuals[1] = predict_left[1] - T(left_curr.y_);
  //out_residuals[2] = predict_right[0] - T(right_curr.x_);
  //out_residuals[3] = predict_right[1] - T(right_curr.y_);

  T predict_left[2], predict_right[2];
  ProjectToCamera(Pc_left, predict_left);
  ProjectToCamera(Pc_right, predict_right);

  // Compute and return the error is the difference between the predicted and observed position
  out_residuals[0] = predict_left[0] - lc[0];
  out_residuals[1] = predict_left[1] - lc[1];
  out_residuals[2] = predict_right[0] - rc[0];
  out_residuals[3] = predict_right[1] - rc[1];

  //double rthr = 0.2;
  //for (int i = 0; i < 4; i++) {
  //  if (GetValue(out_residuals[i] > rthr)) {
  //    std::cout << "obs_left = " << lc[0] << ", " << lc[1] << "\n";
  //    std::cout << "pred_left = " << predict_left[0] << ", " << predict_left[1] << "\n";
  //    std::cout << "obs_right = " << rc[0] << ", " << rc[1] << "\n";
  //    std::cout << "pred_right = " << predict_right[0] << ", " << predict_right[1] << "\n";
  //  }
  //}

  return true;

  //double thr = 100.0;
  //for (int i = 0; i < 4; i++) {
  //  double r = GetValue(out_residuals[i]);
  //  if (r > thr) {
  //    std::cout << r << "\n";
  //    std::cout << "PREV = " << left_prev << " -- " << right_prev << "\n";
  //    std::cout << "CURR = " << left_curr << " -- " << right_curr << "\n";
  //    std::cout << "pred_left = " << GetValue(predict_left[0]) << ", " << GetValue(predict_left[1]) << "\n";
  //    std::cout << "pred_right = " << GetValue(predict_right[0]) << ", " << GetValue(predict_right[1]) << "\n";
  //    break;
  //  }
  //}
}

struct ReprojectionErrorResidual {
  ReprojectionErrorResidual(int frame_id, int track_id,
                            const core::Point& left_prev, const core::Point& left_curr,
                            const core::Point& right_prev, const core::Point& right_curr,
                            const std::array<double,3>& wmotion_trans,
                            const std::array<double,4>& wmotion_rot) :
                            wmotion_rot_(wmotion_rot), wmotion_trans_(wmotion_trans) {
    pt_left_prev_ = left_prev;
    pt_left_curr_ = left_curr;
    pt_right_prev_ = right_prev;
    pt_right_curr_ = right_curr;
    frame_id_ = frame_id;
    track_id_ = track_id;
  }

  template <typename T>
  bool operator()(
      const T* const f1,
      const T* const pp1,
      const T* const k1,
      const T* const f2,
      const T* const pp2,
      const T* const k2,
      const T* const rot,
      const T* const trans,
      T* out_residuals) const {
    //std::cout << "\nFrame = " << frame_id_ << "\nTrack = " << track_id_ << "\n";
    return ComputeReprojectionErrorResidual(f1, pp1, k1, f2, pp2, k2, rot, trans, pt_left_prev_,
                                     pt_right_prev_, pt_left_curr_, pt_right_curr_, wmotion_rot_,
                                     wmotion_trans_, out_residuals);
    //if (frame_id_ == 19 && track_id_ == 676) throw 1;
    //if (frame_id_ == 19 && track_id_ == 40) throw 1;
    //return true;
  }

  static ceres::CostFunction* Create(int frame_id, int track_id,
      const core::Point& left_prev, const core::Point& left_curr,
      const core::Point& right_prev, const core::Point& right_curr,
      const std::array<double,3>& cam_trans,
      const std::array<double,4>& cam_rot) {
    return  new ceres::AutoDiffCostFunction<ReprojectionErrorResidual,
      4,2,2,kNumK,2,2,kNumK,4,3>(new ReprojectionErrorResidual(
            frame_id, track_id, left_prev, left_curr, right_prev, right_curr, cam_trans, cam_rot));
  }

  const std::array<double,4>& wmotion_rot_;
  const std::array<double,3>& wmotion_trans_;
  core::Point pt_left_prev_;
  core::Point pt_right_prev_;
  core::Point pt_left_curr_;
  core::Point pt_right_curr_;
  int frame_id_, track_id_;
};


struct DistortionOnlyLoss {
  DistortionOnlyLoss(int frame_id, int track_id,
                     const core::Point& left_prev, const core::Point& left_curr,
                     const core::Point& right_prev, const core::Point& right_curr,
                     const double* f1, const double* f2, const double* pp1, const double* pp2,
                     const double* rotation, const double* translation,
                     const std::array<double,4>& egomotion_rot,
                     const std::array<double,3>& egomotion_trans) :
                     egomotion_rot_(egomotion_rot), egomotion_trans_(egomotion_trans) {
    f1_ = f1;
    f2_ = f2;
    pp1_ = pp1;
    pp2_ = pp2;
    rot_ = rotation;
    trans_ = translation;

    left_prev_ = left_prev;
    left_curr_ = left_curr;
    right_prev_ = right_prev;
    right_curr_ = right_curr;
    frame_id_ = frame_id;
    track_id_ = track_id;
  }

  template <typename T>
  bool operator()(const T* const k1, const T* const k2, T* out_residuals) const {
#ifdef DEBUG
    std::cout << "\nFrame = " << frame_id_ << "\nTrack = " << track_id_ << "\n";
#endif

    T f1[2], f2[2], pp1[2], pp2[2];
    for (int i = 0; i < 2; i++) {
      f1[i] = T(f1_[i]);
      f2[i] = T(f2_[i]);
      pp1[i] = T(pp1_[i]);
      pp2[i] = T(pp2_[i]);
    }

    T rot[4];
    T trans[3];
    T egomotion_rot[4];
    T egomotion_trans[3];
    for (int i = 0; i < 3; i++) {
      trans[i] = T(trans_[i]);
      egomotion_trans[i] = T(egomotion_trans_[i]);
    }
    for (int i = 0; i < 4; i++) {
      rot[i] = T(rot_[i]);
      egomotion_rot[i] = T(egomotion_rot_[i]);
    }

    // left previous point
    T lp1[2];
    lp1[0] = T(left_prev_.x_);
    lp1[1] = T(left_prev_.y_);

    // left current point
    T lc1[2];
    lc1[0] = T(left_curr_.x_);
    lc1[1] = T(left_curr_.y_);

    // right previous point
    T rp1[2];
    rp1[0] = T(right_prev_.x_);
    rp1[1] = T(right_prev_.y_);

    // right current point
    T rc1[2];
    rc1[0] = T(right_curr_.x_);
    rc1[1] = T(right_curr_.y_);

#ifdef DEBUG
    std::cout << "LP: " << PrintPoint2D(lp1) << "\n";
    std::cout << "RP: " << PrintPoint2D(rp1) << "\n";
    std::cout << "LC: " << PrintPoint2D(lc1) << "\n";
    std::cout << "RC: " << PrintPoint2D(rc1) << "\n";
#endif
    
    // 1. normalize coordinates
    for (int i = 0; i < 2; i++) {
      lp1[i] = (lp1[i] - pp1[i]) / f1[i];
      lc1[i] = (lc1[i] - pp1[i]) / f1[i];
      rp1[i] = (rp1[i] - pp2[i]) / f2[i];
      rc1[i] = (rc1[i] - pp2[i]) / f2[i];
    }
    // 2. undistort points
    T lp[2], rp[2], lc[2], rc[2];
    UndistortPoint(lp1, k1, lp);
    UndistortPoint(lc1, k1, lc);
    UndistortPoint(rp1, k2, rp);
    UndistortPoint(rc1, k2, rc);

#ifdef DEBUG
    std::cout << "LP: " << PrintPoint2D(lp1) << " -> " << PrintPoint2D(lp) << "\n";
    std::cout << "RP: " << PrintPoint2D(rp1) << " -> " << PrintPoint2D(rp) << "\n";
    std::cout << "LC: " << PrintPoint2D(lc1) << " -> " << PrintPoint2D(lc) << "\n";
    std::cout << "RC: " << PrintPoint2D(rc1) << " -> " << PrintPoint2D(rc) << "\n\n";
    T lp2[2], rp2[2], lc2[2], rc2[2];
    DistortPoint(lp, k1, lp2);
    DistortPoint(lc, k1, lc2);
    DistortPoint(rp, k2, rp2);
    DistortPoint(rc, k2, rc2);

    //T lp3[2];
    //UndistortPoint(lp2, k1, lp3);
    //if (std::abs(GetValue(lp1[0])) < std::abs(GetValue(lp[0]))) {
    //if (std::abs(GetValue(lp1[0])) > std::abs(GetValue(lp[0]))) {
    std::cout << "LP: " << PrintPoint2D(lp1) << " == " << PrintPoint2D(lp2) << "\n";
    std::cout << "RP: " << PrintPoint2D(rp1) << " == " << PrintPoint2D(rp2) << "\n";
    std::cout << "LC: " << PrintPoint2D(lc1) << " == " << PrintPoint2D(lc2) << "\n";
    std::cout << "RC: " << PrintPoint2D(rc1) << " == " << PrintPoint2D(rc2) << "\n";
#endif
    //}


    //duplo manji init cost
    //double thr = 1e-3;
    //double thr = 0.3e-3;
    //double thr = 1e-4;
    //if (GetDist(lp2, lp1) > thr || GetDist(lc2, lc1) > thr ||
    //    GetDist(rp2, rp1) > thr || GetDist(rc2, rc1) > thr) {
    //  //throw 1;
    //  //out_residuals[0] = T(0);
    //  //out_residuals[1] = T(0);
    //  //out_residuals[2] = T(0);
    //  //out_residuals[3] = T(0);
    //  //std::cout << "nogodd\n";
    //  return false;
    //}
    //else std::cout << "Is good\n";

    //for (int i = 0; i < 2; i++) {
    //  IsValid(lp[i]);
    //  //if(!ceres::IsNormal(rp[i])) throw 1;
    //  //if(!ceres::IsNormal(lc[i])) throw 1;
    //  //if(!ceres::IsNormal(rc[i])) throw 1;
    //}

    // 3. triangulate in previous frame
    //T C1[3];
    // C2 is trans
    //const T* const C2 = trans;
    //for (int i = 0; i < 3; i++)
    //  C1[i] = T(0.0);
    T r1[3];
    GetUnitVector(lp, r1);
    T r2[3], r2_right[3];
    // get r2 in right coordinate system
    GetUnitVector(rp, r2_right);
    // convert it to left coordinate system
    //std::cout << "Right to left motion = " << GetVector4D(rot) << " -- "
    //          << PrintPoint3D(trans) << "\n";
    ceres::UnitQuaternionRotatePoint(rot, r2_right, r2);
#ifdef DEBUG
    std::cout << "r1 = " << PrintPoint3D(r1) << "\n";
    std::cout << "r2 = " << PrintPoint3D(r2_right) << " -> " << PrintPoint3D(r2) << "\n";
#endif
    T r12dot = Dot(r1, r2);
    T c2r2dot = Dot(trans, r2);
    T m1 = (Dot(trans, r1) - (c2r2dot * r12dot))
           / (T(1.0) - (r12dot * r12dot));
    T m2 = m1 * r12dot - c2r2dot;

    T Pp[3], P1[3], P2[3];
    T dP[3];
    for (int i = 0; i < 3; i++) {
      //P1[i] = C1[i] + (m1 * r1[i]);
      P1[i] = m1 * r1[i];
      P2[i] = trans[i] + (m2 * r2[i]);
      Pp[i] = (P1[i] + P2[i]) / T(2.0);
      dP[i] = P1[i] - P2[i];
    }
    if (Dot(dP, r1) > 1e-10) throw 1;
    if (Dot(dP, r2) > 1e-10) throw 1;

#ifdef DEBUG
    std::cout << "m1 = " << GetValue(m1) << "\n";
    std::cout << "m2 = " << GetValue(m2) << "\n";
    std::cout << "P1 = " << PrintPoint3D(P1) << "\n";
    std::cout << "P2 = " << PrintPoint3D(P2) << "\n";
    std::cout << "Pp = " << PrintPoint3D(Pp) << "\n";
#endif

    //if (GetValue(m1) < 0) {
    //  throw 1;
    //}
    //if (GetValue(m2) < 0) {
    //  throw 1;
    //}

    //T motion_rot[4];
    //for (int i = 0; i < 4; i++)
    //  motion_rot[i] = T(egomotion_rot[i]);
    //T motion_trans[3];
    //for (int i = 0; i < 3; i++)
    //  motion_trans[i] = T(egomotion_trans[i]);

    //std::cout << "Apply camera motion = " << GetVector4D(motion_rot) << " -- "
    //          << PrintPoint3D(motion_trans) << "\n";
    T Pc_left[3];
    Transform3DPoint(Pp, egomotion_rot, egomotion_trans, Pc_left);
#ifdef DEBUG
    std::cout << "Pc_left = " << PrintPoint3D(Pc_left) << "\n";
#endif
    //std::cout << "Pc_left = " << PrintPoint3D(Pc_left) << "\n";

    // convert to the right cam coordinate system
    T rot_inv[4];
    InvertQuaternion(rot, rot_inv);
    //std::cout << "Inverse rotation = " << GetVector4D(rot) << " -> " << GetVector4D(rot_inv) << "\n";
    T Pc_right_tmp[3];
    for (int i = 0 ; i < 3; i++)
      Pc_right_tmp[i] = Pc_left[i] - trans[i];
    T Pc_right[3];
    ceres::UnitQuaternionRotatePoint(rot_inv, Pc_right_tmp, Pc_right);
#ifdef DEBUG
    std::cout << "Pc_right = " << PrintPoint3D(Pc_right) << "\n";
#endif

    //T predict_left[2], predict_right[2];
    //ProjectToCamera(Pc_left, f1, pp1, k1, predict_left);
    //ProjectToCamera(Pc_right, f2, pp2, k2, predict_right);
    //// Compute and return the error is the difference between the predicted and observed position
    //out_residuals[0] = predict_left[0] - T(left_curr_.x_);
    //out_residuals[1] = predict_left[1] - T(left_curr_.y_);
    //out_residuals[2] = predict_right[0] - T(right_curr_.x_);
    //out_residuals[3] = predict_right[1] - T(right_curr_.y_);

    T predict_left[2], predict_right[2];
    ProjectToCamera(Pc_left, predict_left);
    ProjectToCamera(Pc_right, predict_right);
    // Compute and return the error is the difference between the predicted and observed position
    out_residuals[0] = predict_left[0] - lc[0];
    out_residuals[1] = predict_left[1] - lc[1];
    out_residuals[2] = predict_right[0] - rc[0];
    out_residuals[3] = predict_right[1] - rc[1];

#ifdef DEBUG
    double rthr = 6.0 / 1000.0;
    for (int i = 0; i < 4; i++) {
      double err = GetValue(out_residuals[i]);
      if (err > rthr) {
        std::cout << "error = " << err << "\n";
        std::cout << "obs_left = " << lc[0] << ", " << lc[1] << "\n";
        std::cout << "pred_left = " << predict_left[0] << ", " << predict_left[1] << "\n";
        std::cout << "obs_right = " << rc[0] << ", " << rc[1] << "\n";
        std::cout << "pred_right = " << predict_right[0] << ", " << predict_right[1] << "\n";
        throw 1;
      }
    }
#endif

    return true;

    //if (frame_id_ == 19 && track_id_ == 676) throw 1;
    //if (frame_id_ == 19 && track_id_ == 40) throw 1;
    //return true;
  }

  static ceres::CostFunction* Create(int frame_id, int track_id,
                     const core::Point& left_prev, const core::Point& left_curr,
                     const core::Point& right_prev, const core::Point& right_curr,
                     const double* f1, const double* f2, const double* pp1, const double* pp2,
                     const double* rotation, const double* translation,
                     const std::array<double,4>& egomotion_rot,
                     const std::array<double,3>& egomotion_trans) {
    return  new ceres::AutoDiffCostFunction<DistortionOnlyLoss,4,kNumK,kNumK>(
        new DistortionOnlyLoss(frame_id, track_id, left_prev, left_curr, right_prev, right_curr,
                               f1, f2, pp1, pp2, rotation, translation,
                               egomotion_rot, egomotion_trans));
  }

  const double *f1_, *f2_, *pp1_, *pp2_, *rot_, *trans_;
  const std::array<double,4>& egomotion_rot_;
  const std::array<double,3>& egomotion_trans_;
  core::Point left_prev_;
  core::Point right_prev_;
  core::Point left_curr_;
  core::Point right_curr_;
  int frame_id_, track_id_;
};

} // end empty namespace

}

#endif
