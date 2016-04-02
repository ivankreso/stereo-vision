#ifndef OPTIMIZATION_BUNDLE_ADJUSTMENT_CERES_HELPER_
#define OPTIMIZATION_BUNDLE_ADJUSTMENT_CERES_HELPER_

#include <ceres/ceres.h>

namespace optim {

namespace CeresHelper {
  static ceres::LossFunction* CreateLoss(const std::string loss_type,
                                         const std::vector<double>& params);

}   // end CeresHelper

ceres::LossFunction* CeresHelper::CreateLoss(const std::string loss_type,
                                             const std::vector<double>& params) {
  ceres::LossFunction* loss_function;
  if (loss_type == "Cauchy")
    loss_function = new ceres::CauchyLoss(params[0]);
  else if (loss_type == "Huber")
    loss_function = new ceres::HuberLoss(params[0]);
  else if (loss_type == "Squared")
    loss_function = nullptr;
  else if (loss_type == "SoftLOneLoss")
    loss_function = new ceres::SoftLOneLoss(params[0]);
  else {
    std::cout << "Unknown loss type: " << loss_type << " - defaulting to Squared loss.\n";
    loss_function = nullptr;
  }
  return loss_function;
}

}   // end optim



#endif  // OPTIMIZATION_BUNDLE_ADJUSTMENT_CERES_HELPER_
