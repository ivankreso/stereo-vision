#ifndef RECONSTRUCTION_BASE_SGM_STEREO_H_
#define RECONSTRUCTION_BASE_SGM_STEREO_H_

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

namespace recon {

class SGMStereo {
  typedef float CostType;
  //typedef float DisparityType;
  typedef std::vector<std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>> DescriptorTensor;

  // Default parameters
  static const int kNumPaths = 4;
  static const int kDisparityRange = 256;
  static const int kDisparityFactor = 256;
  static const int kP1 = 3;
  static const int kP2 = 40;
  static const int kConsistencyThreshold = 1;

 public:
  SGMStereo();
  //SGMStereo(const cv::Mat& img_left, const cv::Mat& img_right);
  void Compute(const std::string left_descriptors_path,
               const std::string right_descriptors_path,
               cv::Mat* disparity);
  void Compute(const std::string data_cost_path, cv::Mat* disparity);
  void ComputeSGM(cv::Mat* disparity);

  void SetSmoothnessCostParameters(const double P1, const double P2);
  void SetConsistencyThreshold(const int consistency_threshold);

 private:
  void LoadRepresentationFromFile(const std::string& descriptors_path,
                                  DescriptorTensor* descriptors);
  void LoadDataCostFromFile(const std::string file_path);
  void Initialize(const DescriptorTensor& left_desc, const DescriptorTensor& right_desc);
  void SetImageSize(const DescriptorTensor& left_desc, const DescriptorTensor& right_desc);
  void AllocateDataBuffer();
  void ComputeCostImage(const DescriptorTensor& left_descriptors,
                        const DescriptorTensor& right_descriptors);
  void ComputeLeftCostImage(const DescriptorTensor& left_descriptors,
                            const DescriptorTensor& right_descriptors);
  void ComputeRightCostImage(const DescriptorTensor& left_descriptors,
                             const DescriptorTensor& right_descriptors);
  void ComputeRightCostImage();

  void PerformSGM(const CostType* data_cost, int* disparity_img, float* subpix_img);
  void EnforceLeftRightConsistency(const int* left_disparity_image,
                                   const int* right_disparity_image,
                                   char* disp_states) const;
  void InterpolateDisparities(char* disp_states, float* disparity_image) const;
  void SpeckleFilter(const int maxSpeckleSize, const int maxDifference, int* image) const;
  void FreeDataBuffer();

  // Parameter
  int disp_range_;
  double disparity_factor_;

  CostType P1_;
  CostType P2_;
  int consistency_threshold_;

  // Data
  int width_;
  int height_;
  //int widthStep_;
  int sum_cost_size_;
  int lr_size_;
  CostType* left_cost_;
  CostType* right_cost_;
  CostType* sum_cost_;
  CostType* lr_prev_[kNumPaths];
  CostType* lr_curr_[kNumPaths];
  CostType* lr_min_prev_[kNumPaths];
  CostType* lr_min_curr_[kNumPaths];
};

} // namespace recon
#endif
