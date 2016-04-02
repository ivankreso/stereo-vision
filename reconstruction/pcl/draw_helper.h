#ifndef RECONSTRUCTION_BASE_DRAW_HELPER_H_
#define RECONSTRUCTION_BASE_DRAW_HELPER_H_

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "dem.h"
#include "dem_voting.h"

//#define GROUND_THR 0.05
#define OBSTACLE_THR 0.05
#define SUPER_OBSTACLE_THR 0.3

namespace recon
{

namespace DrawHelper
{
  void drawDEM(const recon::DEMvoting& dem, const std::vector<Eigen::Vector2d>& dem_projs,
               const std::vector<int>& class_ids, cv::Mat& img_dem);

  void drawBasicDEM(const recon::DEM& dem, const std::vector<Eigen::Vector2d>& dem_projs,
                    const std::vector<double>& center_elev, cv::Mat& img_dem);
}

}

#endif
