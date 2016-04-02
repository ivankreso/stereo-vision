#ifndef CERES_EXAMPLES_BAL_PROBLEM_STEREO_H_
#define CERES_EXAMPLES_BAL_PROBLEM_STEREO_H_

#include <string>

namespace optim {

class BALProblemStereo {
 public:
  explicit BALProblemStereo(const std::string& filename, bool use_quaternions);
  BALProblemStereo(bool use_quaternions);
  ~BALProblemStereo();

  void WriteToFile(const std::string& filename) const;

  // Move the "center" of the reconstruction to the origin, where the
  // center is determined by computing the marginal median of the
  // points. The reconstruction is then scaled so that the median
  // absolute deviation of the points measured from the origin is
  // 100.0.
  //
  // The reprojection error of the problem remains the same.
  void Normalize();

  // Perturb the camera pose and the geometry with random normal
  // numbers with corresponding standard deviations.
  void Perturb(const double rotation_sigma,
               const double translation_sigma,
               const double point_sigma);

  int camera_block_size()      const { return use_quaternions_ ? 7 : 6; }
  int point_block_size()       const { return 3;                         }
  int num_cameras()            const { return num_cameras_;              }
  int num_points()             const { return num_points_;               }
  int num_observations()       const { return num_observations_;         }
  int num_parameters()         const { return num_parameters_;           }
  const int* point_index()     const { return point_index_;              }
  const int* camera_index()    const { return camera_index_;             }
  const double* observations() const { return observations_;             }
  const double* parameters()   const { return parameters_;               }
  const double* points()       const { return parameters_  + camera_block_size() * num_cameras_; }
  const double* camera_intrinsics() const { return cam_intrinsics_; }
  double* mutable_cameras()          { return parameters_;               }
  double* mutable_points()           { return parameters_  + camera_block_size() * num_cameras_; }
  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * camera_block_size();
  }
  const double* point_for_observation(int i) const {
    return points() + point_index_[i] * point_block_size();
  }
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * point_block_size();
  }
  double get_observation_weight(int i) const { return obs_weights_[i]; }


 private:
  void CameraToAngleAxisAndCenter(const double* camera,
                                  double* angle_axis,
                                  double* center);

  void AngleAxisAndCenterToCamera(const double* angle_axis,
                                  const double* center,
                                  double* camera);
 public:
  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
  bool use_quaternions_ = false;
  int num_intr_ = 5;

  int* point_index_ = nullptr;
  int* camera_index_ = nullptr;
  double* observations_ = nullptr;
  double* obs_weights_ = nullptr;
  // The parameter vector is laid out as follows
  // [camera_1, ..., camera_n, point_1, ..., point_m]
  double* parameters_ = nullptr;
  double* cam_intrinsics_ = nullptr;
};

}  // namespace optim

#endif
