#include "bal_problem_stereo.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <ceres/rotation.h>
//#include <glog/logging.h>

namespace optim {

namespace {
typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

inline double RandDouble() {
  double r = static_cast<double>(rand());
  return r / RAND_MAX;
}

// Box-Muller algorithm for normal random number generation.
// http://en.wikipedia.org/wiki/Box-Muller_transform
inline double RandNormal() {
  double x1, x2, w;
  do {
    x1 = 2.0 * RandDouble() - 1.0;
    x2 = 2.0 * RandDouble() - 1.0;
    w = x1 * x1 + x2 * x2;
  } while ( w >= 1.0 || w == 0.0 );

  w = sqrt((-2.0 * log(w)) / w);
  return x1 * w;
}

template<typename T>
void FscanfOrDie(FILE* fptr, const char* format, T* value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    //LOG(FATAL) << "Invalid UW data file.";
    throw "Invalid UW data file.";
  }
}

void PerturbPoint3(const double sigma, double* point) {
  for (int i = 0; i < 3; ++i) {
    point[i] += RandNormal() * sigma;
  }
}

double Median(std::vector<double>* data) {
  int n = data->size();
  std::vector<double>::iterator mid_point = data->begin() + n / 2;
  std::nth_element(data->begin(), mid_point, data->end());
  return *mid_point;
}

}  // namespace

BALProblemStereo::BALProblemStereo(bool use_quaternions) : use_quaternions_(use_quaternions)
{}

BALProblemStereo::BALProblemStereo(const std::string& filename, bool use_quaternions)
                                  : use_quaternions_(use_quaternions)
{
  FILE* fptr = fopen(filename.c_str(), "r");

  if (fptr == NULL) {
    //LOG(FATAL) << "Error: unable to open file " << filename;
    std::cout << "Error: unable to open file " << filename;
    return;
  };

  num_intr_ = 5;
  cam_intrinsics_ = new double[num_intr_];
  for(int i = 0; i < num_intr_; i++) {
    FscanfOrDie(fptr, "%lf", &cam_intrinsics_[i]);
    std::cout << cam_intrinsics_[i] << " - cam_intrinsics_\n";
  }

  // This wil die horribly on invalid files. Them's the breaks.
  FscanfOrDie(fptr, "%d", &num_cameras_);
  FscanfOrDie(fptr, "%d", &num_points_);
  FscanfOrDie(fptr, "%d", &num_observations_);

  std::cout << "Header: " << num_cameras_
          << " " << num_points_
          << " " << num_observations_;

  point_index_ = new int[num_observations_];
  camera_index_ = new int[num_observations_];
  observations_ = new double[4 * num_observations_];

  num_parameters_ = camera_block_size() * num_cameras_ + point_block_size() * num_points_;
  parameters_ = new double[num_parameters_];

  for (int i = 0; i < num_observations_; ++i) {
    FscanfOrDie(fptr, "%d", camera_index_ + i);
    FscanfOrDie(fptr, "%d", point_index_ + i);
    for (int j = 0; j < 4; ++j) {
      FscanfOrDie(fptr, "%lf", observations_ + 4*i + j);
    }
  }

  for (int i = 0; i < num_parameters_; ++i) {
    FscanfOrDie(fptr, "%lf", parameters_ + i);
  }

  fclose(fptr);

  if (use_quaternions) {
    // Switch the angle-axis rotations to quaternions.
    num_parameters_ = camera_block_size() * num_cameras_ + point_block_size() * num_points_;
    double* quaternion_parameters = new double[num_parameters_];
    double* original_cursor = parameters_;
    double* quaternion_cursor = quaternion_parameters;
    for (int i = 0; i < num_cameras_; ++i) {
      ceres::AngleAxisToQuaternion(original_cursor, quaternion_cursor);
      quaternion_cursor += 4;
      original_cursor += 3;
      for (int j = 4; j < camera_block_size(); ++j) {
       *quaternion_cursor++ = *original_cursor++;
      }
    }
    // Copy the rest of the points.
    for (int i = 0; i < point_block_size() * num_points_; ++i) {
      *quaternion_cursor++ = *original_cursor++;
    }
    // Swap in the quaternion parameters.
    delete[] parameters_;
    parameters_ = quaternion_parameters;
  }
}

// This function writes the problem to a file in the same format that
// is read by the constructor.
void BALProblemStereo::WriteToFile(const std::string& filename) const {
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    std::cout << "Error: unable to open file " << filename;
    return;
  };

  for(int i = 0; i < num_intr_; i++)
    fprintf(fptr, "%g ", cam_intrinsics_[i]);
  fprintf(fptr, "\n");

  fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);

  for (int i = 0; i < num_observations_; ++i) {
    fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
    for (int j = 0; j < 4; ++j) {
      fprintf(fptr, " %g", observations_[4 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  for (int i = 0; i < num_cameras(); ++i) {
    double angleaxis[6];
    if (use_quaternions_) {
      // Output in angle-axis format.
      ceres::QuaternionToAngleAxis(parameters_ + camera_block_size() * i, angleaxis);
      memcpy(angleaxis + 3, parameters_ + camera_block_size() * i + 4, (camera_block_size()-4) * sizeof(double));
    } else {
      memcpy(angleaxis, parameters_ + camera_block_size() * i, camera_block_size() * sizeof(double));
    }
    for (int j = 0; j < camera_block_size(); ++j) {
      fprintf(fptr, "%.16g\n", angleaxis[j]);
    }
  }

  const double* points = parameters_ + camera_block_size() * num_cameras_;
  for (int i = 0; i < num_points(); ++i) {
    const double* point = points + i * point_block_size();
    for (int j = 0; j < point_block_size(); ++j) {
      fprintf(fptr, "%.16g\n", point[j]);
    }
  }

  fclose(fptr);
}

void BALProblemStereo::CameraToAngleAxisAndCenter(const double* camera,
                                            double* angle_axis,
                                            double* center) {
  VectorRef angle_axis_ref(angle_axis, 3);
  if (use_quaternions_) {
    ceres::QuaternionToAngleAxis(camera, angle_axis);
  } else {
    angle_axis_ref = ConstVectorRef(camera, 3);
  }

  int point_pos = (use_quaternions_ ? 4 : 3);
  // c = -R't
  Eigen::VectorXd inverse_rotation = -angle_axis_ref;
  ceres::AngleAxisRotatePoint(inverse_rotation.data(), camera + point_pos, center);
  VectorRef(center, 3) *= -1.0;
}

void BALProblemStereo::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) {
  ConstVectorRef angle_axis_ref(angle_axis, 3);
  if (use_quaternions_) {
    ceres::AngleAxisToQuaternion(angle_axis, camera);
  } else {
    VectorRef(camera, 3) = angle_axis_ref;
  }

  int point_pos = (use_quaternions_ ? 4 : 3);
  // t = -R * c
  ceres::AngleAxisRotatePoint(angle_axis,
                       center,
                       camera + point_pos);
  VectorRef(camera + point_pos, 3) *= -1.0;
}


void BALProblemStereo::Normalize() {
  // Compute the marginal median of the geometry.
  std::vector<double> tmp(num_points_);
  Eigen::Vector3d median;
  double* points = mutable_points();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < num_points_; ++j) {
      tmp[j] = points[3 * j + i];
    }
    median(i) = Median(&tmp);
  }

  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    tmp[i] = (point - median).lpNorm<1>();
  }

  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100.
  const double scale = 100.0 / median_absolute_deviation;

  std::cout << "median: " << median.transpose();
  std::cout << "median absolute deviation: " << median_absolute_deviation;
  std::cout  << "scale: " << scale;

  // X = scale * (X - median)
  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    point = scale * (point - median);
  }

  double* cameras = mutable_cameras();
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras_; ++i) {
    double* camera = cameras + camera_block_size() * i;
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    // center = scale * (center - median)
    VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
    AngleAxisAndCenterToCamera(angle_axis, center, camera);
  }
}

void BALProblemStereo::Perturb(const double rotation_sigma,
                               const double translation_sigma,
                               const double point_sigma)
{
  CHECK_GE(point_sigma, 0.0);
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);

  double* points = mutable_points();
  if (point_sigma > 0) {
    for (int i = 0; i < num_points_; ++i) {
      PerturbPoint3(point_sigma, points + 3 * i);
    }
  }

  for (int i = 0; i < num_cameras_; ++i) {
    double* camera = mutable_cameras() + camera_block_size() * i;

    double angle_axis[3];
    double center[3];
    // Perturb in the rotation of the camera in the angle-axis
    // representation.
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    if (rotation_sigma > 0.0) {
      PerturbPoint3(rotation_sigma, angle_axis);
    }
    AngleAxisAndCenterToCamera(angle_axis, center, camera);

    int point_pos = (use_quaternions_ ? 4 : 3);
    if (translation_sigma > 0.0) {
      PerturbPoint3(translation_sigma, camera + point_pos);
    }
  }
}

BALProblemStereo::~BALProblemStereo() {
  delete[] point_index_;
  delete[] camera_index_;
  delete[] observations_;
  delete[] parameters_;
  delete[] cam_intrinsics_;
}

}  // namespace optim
