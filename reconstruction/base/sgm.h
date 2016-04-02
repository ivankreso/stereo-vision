#ifndef RECONSTRUCTION_BASE_SEMI_GLOBAL_MATCHING_
#define RECONSTRUCTION_BASE_SEMI_GLOBAL_MATCHING_

#include <iostream>
#include <fstream>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/StdVector>

#include <opencv2/core/core.hpp>

//#define COST_CENSUS
//#define COST_ZSAD
#define COST_CNN

namespace recon {

struct SGMParams {
  int disp_range;
  int window_sz;
  int penalty1;
  int penalty2;
};

#ifdef COST_SAD
// SAD
typedef uint16_t CostType;
typedef uint32_t ACostType;
#endif

#ifdef COST_ZSAD
// ZSAD
typedef float CostType;  // for ZSAD
typedef float ACostType;  // accumulated cost type
#endif

#ifdef COST_CENSUS
// Census
typedef uint8_t CostType;  // for 1x1 SAD, 5x5 Census
typedef uint32_t ACostType;  // accumulated cost type, Census
#endif

#ifdef COST_CNN
typedef float CostType;  // for 1x1 SAD, 5x5 Census
typedef float ACostType;  // accumulated cost type, Census
typedef std::vector<std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>> DescriptorTensor;
#endif

//typedef uint8_t ACostType;  // accumulated cost type - byte for Census?
// if we want to use raw arrays
//typedef CostType* CostArray1D;
//typedef CostType*** CostArray3D;
//typedef ACostType* ACostArray1D;
//typedef ACostType*** ACostArray3D;

typedef std::vector<CostType> CostArray1D;
typedef std::vector<ACostType> ACostArray1D;
typedef std::vector<std::vector<CostArray1D>> CostArray;
typedef std::vector<std::vector<ACostArray1D>> ACostArray;

class SGM
{
 public:
  SGM(SGMParams& params) : params_(params) {}
  cv::Mat Compute(const std::string left_desc_path, const std::string right_desc_path);

 protected:
  void aggregate_costs(const CostArray& costs, const int dir_x, const int dir_y, ACostArray& aggr_costs);

  void sum_costs(const ACostArray& costs1, ACostArray& costs2);
  void LoadRepresentationFromFile(const std::string& desciptors_path, DescriptorTensor* descriptors);

  template<typename T1, typename T2>
  void copy_vector(const std::vector<T1>& vec1, std::vector<T2>& vec2);
  template<typename T1, typename T2>
  void sum_vectors(const std::vector<T1>& vec1, std::vector<T2>& vec2);
  template<typename T>
  T get_min(const std::vector<T>& vec);
  int FindMinDisp(const std::vector<CostType>& costs);
  int find_min_disp(const std::vector<ACostType>& costs);
  int find_min_disp_right(const std::vector<std::vector<ACostType>>& costs, int x);
  void init_costs(ACostType init_val, ACostArray& costs);

  cv::Mat GetDisparityImage(const CostArray& costs, int msz);
  cv::Mat get_disparity_matrix_float(const ACostArray& costs, int msz);
  cv::Mat get_disparity_image_uint16(const ACostArray& costs, int msz);
  cv::Mat get_disparity_image(const ACostArray& costs, int msz);
  void aggregate_path(const std::vector<ACostType>& prior, const std::vector<CostType>& local,
                      std::vector<ACostType>& costs);

  //inline getCensusCost();
  SGMParams params_;
};

template<typename T1, typename T2>
inline
void SGM::sum_vectors(const std::vector<T1>& vec1, std::vector<T2>& vec2)
{
  assert(vec1.size() == vec2.size());
  for(size_t i = 0; i < vec2.size(); i++)
    vec2[i] += (T2)vec1[i];
}

template<typename T1, typename T2>
inline
void SGM::copy_vector(const std::vector<T1>& vec1, std::vector<T2>& vec2)
{
  assert(vec1.size() == vec2.size());
  for(size_t i = 0; i < vec1.size(); i++) {
    vec2[i] = (T2)vec1[i];
    //std::cout << "d = " << i << " - " << (T2)vec1[i] << " == " << vec2[i] << "\n";
  }
}

inline
void SGM::sum_costs(const ACostArray& costs1, ACostArray& costs2)
{
  size_t height = costs1.size();
  size_t width = costs1[0].size();
  assert(costs1.size() == costs2.size());
  for(size_t y = 0; y < height; y++) {
    assert(costs1[y].size() == costs2[y].size());
    for(size_t x = 0; x < width; x++) {
      sum_vectors<ACostType,ACostType>(costs1[y][x], costs2[y][x]);
    }
  }
}

inline
void SGM::aggregate_path(const std::vector<ACostType>& prior, const std::vector<CostType>& local,
                         std::vector<ACostType>& costs)
{
  assert(params_.disp_range == costs.size());
  int P1 = params_.penalty1;
  int P2 = params_.penalty2;
  copy_vector<CostType,ACostType>(local, costs);
  int max_disp = params_.disp_range;

  ACostType min_prior = get_min<ACostType>(prior);
  // decrease the P2 error if the gradient is big which is a clue for the discontinuites
  // TODO: works very bad on KITTI...
  //P2 = std::max(P1, gradient ? (int)std::round(static_cast<float>(P2/gradient)) : P2);
  for(int d = 0; d < max_disp; d++) {
    ACostType error = min_prior + P2;
    error = std::min(error, prior[d]);
    if(d > 0)
      error = std::min(error, prior[d-1] + P1);
    if(d < (max_disp - 1))
      error = std::min(error, prior[d+1] + P1);

    // ACostType can be uint8_t and e_smooth int
    // Normalize by subtracting min of prior cost
    // Now we have upper limit on cost: e_smooth <= C_max + P2
    // LR check won't work without this normalization also
    costs[d] += (error - min_prior);
  }
}

//inline
//std::vector<ACostType> SGM::aggregate_path(const std::vector<ACostType>& prior,
//                                                 const std::vector<CostType>& local)
//{
//  int P1 = params_.penalty1;
//  int P2 = params_.penalty2;
//  std::vector<ACostType> curr_cost;
//  copy_vector<CostType,ACostType>(local, curr_cost);
//  for(int d = 0; d < params_.disp_range; d++) {
//    //int e_smooth = std::numeric_limits<int>::max();
//    ACostType e_smooth = std::numeric_limits<ACostType>::max();
//    for(int d_p = 0; d_p < params_.disp_range; d_p++) {
//      if(d_p - d == 0) {
//        // No penality
//        e_smooth = std::min(e_smooth, prior[d_p]);
//      }
//      else if(std::abs(d_p - d) == 1) {
//        // Small penality
//        e_smooth = std::min(e_smooth, prior[d_p] + P1);
//      } else {
//        // Large penality
//        //e_smooth = std::min(e_smooth, prior[d_p] + std::max(P1, path_gradient ? P2/path_gradient : P2));
//        e_smooth = std::min(e_smooth, prior[d_p] + P2);
//      }
//    }
//    curr_cost[d] += e_smooth;
//  }
//
//  // TODO: why
//  // Normalize by subtracting min of prior cost
//  //ACostType min = get_min<ACostType>(prior);
//  //for(size_t i = 0; i < curr_cost.size(); i++)
//  //  curr_cost[i] -= min;
//  return curr_cost;
//}

inline
void SGM::init_costs(ACostType init_val, ACostArray& costs)
{
  size_t disp_range = costs[0][0].size();
  size_t width = costs[0].size();
  size_t height = costs.size();
  for(size_t i = 0; i < height; i++)
    for(size_t j = 0; j < width; j++)
      for(size_t k = 0; k < disp_range; k++)
        costs[i][j][k] = init_val;
}

template<typename T>
inline
T SGM::get_min(const std::vector<T>& vec)
{
  T min = vec[0];
  for(size_t i = 1; i < vec.size(); i++) {
    if(vec[i] < min)
      min = vec[i];
  }
  return min;
}

inline
int SGM::find_min_disp(const std::vector<ACostType>& costs) {
  int d = 0;
  for(size_t i = 1; i < costs.size(); i++) {
    if(costs[i] < costs[d])
      d = i;
  }
  return d;
}

inline
int SGM::FindMinDisp(const std::vector<CostType>& costs) {
  int d = 0;
  for(size_t i = 1; i < costs.size(); i++) {
    if(costs[i] < costs[d])
      d = i;
  }
  return d;
}

inline
int SGM::find_min_disp_right(const std::vector<std::vector<ACostType>>& costs, int x)
{
  int d = 0;
  //ACostType min_cost = costs[x+d][d];
  int width = costs.size();
  int max_disp = std::min(params_.disp_range, (width - x));
  for(int i = 1; i < max_disp; i++) {
    if(costs[x+i][i] < costs[x+d][d])
      d = i;
  }
  return d;
}

inline
cv::Mat SGM::GetDisparityImage(const CostArray& costs, int msz)
{
  int height = costs.size();
  int width = costs[0].size();
  cv::Mat img = cv::Mat::zeros(height + 2*msz, width + 2*msz, CV_8U);
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      int d = FindMinDisp(costs[y][x]);
      //img.at<uint8_t>(y,x) = 4 * d;
      img.at<uint8_t>(msz+y, msz+x) = d;
    }
  }
  return img;
}

inline
cv::Mat SGM::get_disparity_image(const ACostArray& costs, int msz)
{
  int height = costs.size();
  int width = costs[0].size();
  cv::Mat img = cv::Mat::zeros(height + 2*msz, width + 2*msz, CV_8U);
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      int d = find_min_disp(costs[y][x]);
      //img.at<uint8_t>(y,x) = 4 * d;
      img.at<uint8_t>(msz+y, msz+x) = d;
    }
  }
  return img;
}

inline
cv::Mat SGM::get_disparity_image_uint16(const ACostArray& costs, int msz)
{
  int height = costs.size();
  int width = costs[0].size();
  //cv::Mat img = cv::Mat::zeros(height + 2*msz, width + 2*msz, CV_16U);
  cv::Mat img = cv::Mat::zeros(height, width, CV_16U);
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      // find minimum cost disparity
      int d = find_min_disp(costs[y][x]);
      // TODO: do the fast LR check
      if((x-d) >= 0) {
        int d_right = find_min_disp_right(costs[y], x-d);
        //std::cout << "d = " << d << " , " << " d_r = " << d_right << "\n";
        if(std::abs(d - d_right) > 2) {
          //img.at<uint16_t>(msz+y, msz+x) = 0;
          img.at<uint16_t>(y,x) = 0;
          continue;
        }
      }
      else {
        //img.at<uint16_t>(msz+y, msz+x) = 0;
        img.at<uint16_t>(y,x) = 0;
        continue;
      }
      // perform equiangular subpixel interpolation
      if(d >= 1 && d < (params_.disp_range-1)) {
        float C_left = costs[y][x][d-1];
        float C_center = costs[y][x][d];
        float C_right = costs[y][x][d+1];
        float d_s = 0;
        if(C_right < C_left)
          d_s = 0.5f * (C_right - C_left) / (C_center - C_left);
        else
          d_s = 0.5f * (C_right - C_left) / (C_center - C_right);
        //std::cout << d << " -- " << d+d_s << "\n";
        //img.at<uint16_t>(msz+y, msz+x) = static_cast<uint16_t>(std::round(256.0 * (d + d_s)));
        img.at<uint16_t>(y,x) = static_cast<uint16_t>(std::round(256.0 * (d + d_s)));
      }
      else {
        //img.at<uint16_t>(msz+y, msz+x) = static_cast<uint16_t>(std::round(256.0 * d));
        img.at<uint16_t>(y,x) = static_cast<uint16_t>(std::round(256.0 * d));
      }
    }
  }
  return img;
}

inline
cv::Mat SGM::get_disparity_matrix_float(const ACostArray& costs, int msz)
{
  int height = costs.size();
  int width = costs[0].size();
  cv::Mat img = cv::Mat::zeros(height + 2*msz, width + 2*msz, CV_32F);
  //#pragma omp parallel for
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      // find minimum cost disparity
      int d = find_min_disp(costs[y][x]);
      // TODO: do the fast LR check
      if((x-d) >= 0) {
        int d_right = find_min_disp_right(costs[y], x-d);
        //std::cout << "d = " << d << " , " << " d_r = " << d_right << "\n";
        if(std::abs(d - d_right) > 2) {
          img.at<float>(msz+y, msz+x) = -1.0f;
          continue;
        }
      }
      else {
        img.at<float>(msz+y, msz+x) = -1.0f;
        continue;
      }
      // perform equiangular subpixel interpolation
      if(d >= 1 && d < (params_.disp_range-1)) {
        float C_left = costs[y][x][d-1];
        float C_center = costs[y][x][d];
        float C_right = costs[y][x][d+1];
        float d_s = 0;
        if(C_right < C_left)
          d_s = 0.5f * (C_right - C_left) / (C_center - C_left);
        else
          d_s = 0.5f * (C_right - C_left) / (C_center - C_right);
        //std::cout << d << " -- " << d+d_s << "\n";
        img.at<float>(msz+y, msz+x) = static_cast<float>(d + d_s);
      }
      else
        img.at<float>(msz+y, msz+x) = static_cast<float>(d);
    }
  }
  return img;
}

inline
void SGM::LoadRepresentationFromFile(const std::string& descriptors_path,
                                     DescriptorTensor* descriptors) {
  std::ifstream file(descriptors_path, std::ios::binary);
  //for (int i = 0; i < 100; i++) {
  //  //uint64_t dims;
  //  int dims;
  //  file.read(reinterpret_cast<char*>(&dims), sizeof(dims));
  //  std::cout << dims << ", ";
  //}

  int dims;
  file.read(reinterpret_cast<char*>(&dims), sizeof(dims));
  if (dims != 3) throw 1;
  std::vector<uint64_t> size;
  size.assign(dims, 0);
  for (int i = 0; i < dims; i++) {
    //file >> size[i];
    file.read(reinterpret_cast<char*>(&size[i]), sizeof(size[i]));
  }
  descriptors->resize(size[0]);
  for (uint64_t i = 0; i < size[0]; i++) {
    //(*descriptors)[i].resize(size[1]);
    (*descriptors)[i].assign(size[1], Eigen::VectorXf(size[2]));
    for (uint64_t j = 0; j < size[1]; j++) {
        for (uint64_t k = 0; k < size[2]; k++) {
          file.read(reinterpret_cast<char*>(&(*descriptors)[i][j][k]), sizeof((*descriptors)[i][j][k]));
          //file.read(reinterpret_cast<char*>(&(*descriptors)[i][j][k]), sizeof(float));
          //std::cout << (*descriptors)[i][j][k]  << "\n";
        }
    }
  }
}

}

#endif
