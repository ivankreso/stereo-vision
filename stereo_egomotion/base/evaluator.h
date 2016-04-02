#ifndef STEREO_EGOMOTION_BASE_EVALUATOR_
#define STEREO_EGOMOTION_BASE_EVALUATOR_

#include <vector>
#include <string>
#include <Eigen/Core>

#include "matrix.h"

namespace egomotion {

namespace Evaluator {

void Eval(const std::string& gt_fname, const std::vector<libviso::Matrix>& egomotion_poses,
          double& trans_error, double& rot_error);
void Eval(const std::string& gt_fname, const std::vector<Eigen::Matrix4d>& egomotion_poses,
          double& trans_error, double& rot_error);

libviso::Matrix EigenMatrixToLibvisoMatrix(const Eigen::Matrix4d& mat1);

void EigenVectorToLibvisoVector(const std::vector<Eigen::Matrix4d>& vec1,
                                std::vector<libviso::Matrix>& vec2);

} // namespace Evaluator

} // namespace egomotion

#endif  // STEREO_EGOMOTION_BASE_EVALUATOR_
