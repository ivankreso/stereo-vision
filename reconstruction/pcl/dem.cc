#include "dem.h"

namespace recon {

DEM::DEM(double x_start, double x_end, double z_start, double z_end, int x_cells, int z_cells, double elevation_thr)
  : x_start_(x_start), x_end_(x_end), z_start_(z_start), z_end_(z_end), x_cells_(x_cells), z_cells_(z_cells),
    elevation_thr_(elevation_thr)
{
  assert(x_end > x_start && z_end > z_start);
  width_ = x_end - x_start;
  length_ = z_end - z_start;
  cell_width_ = width_ / x_cells;
  cell_length_ = length_ / z_cells;

  int n_cells = x_cells * z_cells;
  dem_.assign(n_cells, 0.0);
  visibility_map_.assign(n_cells, false);
  density_map_.assign(n_cells, 0);
}


void DEM::update(const Eigen::Vector3d& point, double elevation)
{
  double x = point[0];
  double z = point[2];
  if(x < x_start_ || x > x_end_ || z < z_start_ || z > z_end_)
    throw "[DEM]: point out of range\n";

  int cell_x = (int)((x - x_start_) / cell_width_);
  int cell_z = (int)((z - z_start_) / cell_length_);
  int cell_idx = cell_z * x_cells_ + cell_x;
  if(dem_[cell_idx] < elevation)
    dem_[cell_idx] = elevation;
  if(visibility_map_[cell_idx] == false)
    visibility_map_[cell_idx] = true;
}

void DEM::update(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, const std::vector<double>& elevations)
{
  for(size_t i = 0; i < point_cloud->points.size(); i++) {
    double x = point_cloud->points[i].x;
    double z = point_cloud->points[i].z;
    if(x < x_start_ || x > x_end_ || z < z_start_ || z > z_end_)
      continue;
    // ignore point above threshold
    if(elevations[i] > elevation_thr_)
      continue;

    int cell_x = (int)((x - x_start_) / cell_width_);
    int cell_z = (int)((z - z_start_) / cell_length_);
    assert(cell_x >= 0 && cell_x < x_cells_ && cell_z >= 0 && cell_z < z_cells_);
    int cell_idx = cell_z * x_cells_ + cell_x;
    if(dem_[cell_idx] < elevations[i])
      dem_[cell_idx] = elevations[i];
    if(visibility_map_[cell_idx] == false)
      visibility_map_[cell_idx] = true;
  }
}

double DEM::getCell(int x_cell, int z_cell, const Eigen::Vector4d& plane, Eigen::Vector3d& center) const
{
  double elevation = dem_[z_cell*x_cells_ + x_cell];
  // calculate the Y as the point above groud plane with elevated distance from it
  // just get the point with that distance from the intersection point (x-z line and plane)
  // in direction of the plane normal
  // TODO: this is just approx for now
  center[1] = -plane[3] + elevation;

  center[0] = (x_start_ + x_cell*cell_width_) + (cell_width_/2);
  center[2] = (z_start_ + z_cell*cell_length_) + (cell_length_/2);
  return elevation;
}

bool DEM::isCellVisible(int x_cell, int z_cell) const
{
  int idx = z_cell*x_cells_ + x_cell;
  return visibility_map_[idx];
}


DEM::~DEM()
{

}

} // end namespace
