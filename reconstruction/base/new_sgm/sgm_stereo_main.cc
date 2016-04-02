#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "sgm_stereo.h"

//void ConvertDispImageToCvMat8(const png::image<png::gray_pixel_16>& img, cv::Mat& cvimg) {
//  cvimg.create(img.get_height(), img.get_width(), CV_8U);
//  for (int i = 0; i < cvimg.rows; i++)
//    for (int j = 0; j < cvimg.cols; j++)
//      cvimg.at<uint8_t>(i,j) = (uint8_t)std::min(255.0, std::round(img.get_pixel(j,i) / 256.0));
//}
//
//void ConvertDispImageToCvMat16(const png::image<png::gray_pixel_16>& img, cv::Mat& cvimg) {
//  cvimg.create(img.get_height(), img.get_width(), CV_16U);
//  for (int i = 0; i < cvimg.rows; i++)
//    for (int j = 0; j < cvimg.cols; j++)
//      cvimg.at<uint16_t>(i,j) = img.get_pixel(j,i);
//}

void ConvertMat16ToMat8(const cv::Mat& mat16, cv::Mat* mat8) {
  mat8->create(mat16.rows, mat16.cols, CV_8U);
  for (int i = 0; i < mat8->rows; i++)
    for (int j = 0; j < mat8->cols; j++)
      mat8->at<uint8_t>(i,j) = static_cast<uint8_t>(std::min(255.0, std::round(mat16.at<uint16_t>(i,j) / 256.0)));
}

void ConvertFloatDispToMat16(const float* disp, const int width, const int height, cv::Mat* img) {
  img->create(height, width, CV_16U);
  for (int i = 0; i < img->rows; i++) {
    for (int j = 0; j < img->cols; j++) {
      float d = disp[i*width + j];
      if (d < 0 || d > 255) throw 1;
      img->at<uint16_t>(i,j) = static_cast<uint16_t>(std::round(d * 256.0));
    }
  }
}

void main1(int argc, char** argv) {
  std::string left_desc_path = argv[1];
  std::string right_desc_path = argv[2];
  std::string out_folder = argv[3];
  const int P1 = std::stoi(argv[4]);
  const int P2 = std::stoi(argv[5]);
  const int consistency_threshold = std::stoi(argv[6]);

  //png::image<png::rgb_pixel> leftImage(leftImageFilename);
  //png::image<png::rgb_pixel> rightImage(rightImageFilename);

  recon::SGMStereo sgm;
  sgm.SetSmoothnessCostParameters(P1, P2);
  sgm.SetConsistencyThreshold(consistency_threshold);
  //sps.setIterationTotal(outerIterationTotal, innerIterationTotal);
  //sps.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);
  //sps.setInlierThreshold(lambda_d);
  //sps.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen);

  //int width = leftImage.get_width();
  //int height = leftImage.get_height();
  //float* disparities = reinterpret_cast<float*>(malloc(width * height * sizeof(float)));
  cv::Mat img16;
  sgm.Compute(left_desc_path, right_desc_path, &img16);
  //for (int i = 0; i < height; i++) {
  //  for (int j = 0; j < width; j++) {
  //    std::cout << disparities[i*width + j] << "\n";
  //  }
  //}


  std::string outputBaseFilename = left_desc_path;
  size_t slashPosition = outputBaseFilename.rfind('/');
  if (slashPosition != std::string::npos) outputBaseFilename.erase(0, slashPosition+1);
  size_t dotPosition = outputBaseFilename.rfind('.') - 5;
  if (dotPosition != std::string::npos) outputBaseFilename.erase(dotPosition);
  std::string save_name = outputBaseFilename + ".png";
  cv::Mat img8;
  //ConvertFloatDispToMat16(disparities, width, height, &img16);
  //cv::medianBlur(img16, img16, 3);
  ConvertMat16ToMat8(img16, &img8);
  cv::normalize(img8, img8, 0, 255, cv::NORM_MINMAX);
  cv::imwrite(out_folder + "/disparities/" + save_name, img16);
  cv::imwrite(out_folder + "/norm_hist/" + save_name, img8);

  //png::image<png::gray_pixel_16> segmentImage;
  //png::image<png::gray_pixel_16> disparityImage;
  //std::vector< std::vector<double> > disparityPlaneParameters;
  //std::vector< std::vector<int> > boundaryLabels;
  //sps.compute(superpixelTotal, leftImage, rightImage, segmentImage, disparityImage, disparityPlaneParameters, boundaryLabels);

  //std::string outputDisparityImageFilename = outputBaseFilename + "_left_disparity.png";
  //std::string outputSegmentImageFilename = outputBaseFilename + "_segment.png";
  //std::string outputBoundaryImageFilename = outputBaseFilename + "_boundary.png";
  //std::string outputDisparityPlaneFilename = outputBaseFilename + "_plane.txt";
  //std::string outputBoundaryLabelFilename = outputBaseFilename + "_label.txt";

  //cv::Mat disp_img16, disp_img8, disp_img_eq;
  //ConvertDispImageToCvMat8(disparityImage, disp_img8);
  //ConvertDispImageToCvMat16(disparityImage, disp_img16);
  ////std::cout << disp_img16;
  //cv::normalize(disp_img8, disp_img_eq, 0, 255, cv::NORM_MINMAX);
  ////cv::imwrite(out_folder + "/" + outputBaseFilename + ".png", disp_img16);
  //disparityImage.write(out_folder + "/" + outputBaseFilename + ".png");
  //cv::imwrite(out_folder + "_norm_hist/" + outputBaseFilename + "_norm_hist.png", disp_img_eq);

  //disparityImage.write(outputDisparityImageFilename);
  //segmentImage.write(outputSegmentImageFilename);
  //segmentBoundaryImage.write(outputBoundaryImageFilename);
  //writeDisparityPlaneFile(disparityPlaneParameters, outputDisparityPlaneFilename);
  //writeBoundaryLabelFile(boundaryLabels, outputBoundaryLabelFilename);
}

void main2(int argc, char** argv) {
  std::string data_cost_path = argv[1];
  std::string out_folder = argv[2];
  const int P1 = std::stoi(argv[3]);
  const int P2 = std::stoi(argv[4]);
  const int consistency_threshold = std::stoi(argv[5]);

  //png::image<png::rgb_pixel> leftImage(leftImageFilename);
  //png::image<png::rgb_pixel> rightImage(rightImageFilename);

  recon::SGMStereo sgm;
  sgm.SetSmoothnessCostParameters(P1, P2);
  sgm.SetConsistencyThreshold(consistency_threshold);

  //int width = leftImage.get_width();
  //int height = leftImage.get_height();
  //float* disparities = reinterpret_cast<float*>(malloc(width * height * sizeof(float)));
  cv::Mat img16;
  sgm.Compute(data_cost_path, &img16);

  std::string outputBaseFilename = data_cost_path;
  size_t slashPosition = outputBaseFilename.rfind('/');
  if (slashPosition != std::string::npos) outputBaseFilename.erase(0, slashPosition+1);
  size_t dotPosition = outputBaseFilename.rfind('.') - 5;
  if (dotPosition != std::string::npos) outputBaseFilename.erase(dotPosition);
  std::string save_name = outputBaseFilename + ".png";
  cv::Mat img8;
  //ConvertFloatDispToMat16(disparities, width, height, &img16);
  //cv::medianBlur(img16, img16, 3);
  ConvertMat16ToMat8(img16, &img8);
  cv::normalize(img8, img8, 0, 255, cv::NORM_MINMAX);
  cv::imwrite(out_folder + "/disparities/" + save_name, img16);
  cv::imwrite(out_folder + "/norm_hist/" + save_name, img8);
}

int main(int argc, char** argv) {
  if (argc == 7)
    main1(argc, argv);
  else if (argc == 6)
    main2(argc, argv);
  else {
    std::cerr << "usage: ./sgm left right out_folder P1 P2 consistency_threshold" << std::endl;
    std::cerr << "usage: ./sgm data_cost_path out_folder P1 P2 consistency_threshold" << std::endl;
  }
  return 0;
}
