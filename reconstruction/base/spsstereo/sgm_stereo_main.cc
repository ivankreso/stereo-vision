#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <png++/png.hpp>
#include "SGMStereo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

void ConvertDispImageToCvMat8(const png::image<png::gray_pixel_16>& img, cv::Mat& cvimg) {
  cvimg.create(img.get_height(), img.get_width(), CV_8U);
  for (int i = 0; i < cvimg.rows; i++)
    for (int j = 0; j < cvimg.cols; j++)
      cvimg.at<uint8_t>(i,j) = (uint8_t)std::min(255.0, std::round(img.get_pixel(j,i) / 256.0));
}

void ConvertDispImageToCvMat16(const png::image<png::gray_pixel_16>& img, cv::Mat& cvimg) {
  cvimg.create(img.get_height(), img.get_width(), CV_16U);
  for (int i = 0; i < cvimg.rows; i++)
    for (int j = 0; j < cvimg.cols; j++)
      cvimg.at<uint16_t>(i,j) = img.get_pixel(j,i);
}

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

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: sgmstereo left right out_folder" << std::endl;
    exit(1);
  }

  std::string leftImageFilename = argv[1];
  std::string rightImageFilename = argv[2];
  std::string out_folder = argv[3];

  png::image<png::rgb_pixel> leftImage(leftImageFilename);
  png::image<png::rgb_pixel> rightImage(rightImageFilename);
  //for (int i = 0; i < leftImage.get_height(); i++) {
  //  for (int j = 0; j < leftImage.get_width(); j++) {
  //    std::cout << (int)leftImage.get_pixel(j,i).green << "\t";
  //    std::cout << (int)leftImage.get_pixel(j,i).red << "\t";
  //    std::cout << (int)leftImage.get_pixel(j,i).blue << "\n";
  //  }
  //}

  SGMStereo sps;
  //sps.setIterationTotal(outerIterationTotal, innerIterationTotal);
  //sps.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);
  //sps.setInlierThreshold(lambda_d);
  //sps.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen);

  int width = leftImage.get_width();
  int height = leftImage.get_height();
  float* disparities = reinterpret_cast<float*>(malloc(width * height * sizeof(float)));
  sps.compute(leftImage, rightImage, disparities);
  //for (int i = 0; i < height; i++) {
  //  for (int j = 0; j < width; j++) {
  //    std::cout << disparities[i*width + j] << "\n";
  //  }
  //}


  std::string outputBaseFilename = leftImageFilename;
  size_t slashPosition = outputBaseFilename.rfind('/');
  if (slashPosition != std::string::npos) outputBaseFilename.erase(0, slashPosition+1);
  //size_t dotPosition = outputBaseFilename.rfind('.');
  //if (dotPosition != std::string::npos) outputBaseFilename.erase(dotPosition);
  cv::Mat img16, img8;
  ConvertFloatDispToMat16(disparities, width, height, &img16);
  cv::medianBlur(img16, img16, 3);
  ConvertMat16ToMat8(img16, &img8);
  cv::normalize(img8, img8, 0, 255, cv::NORM_MINMAX);
  cv::imwrite(out_folder + "/disp_" + outputBaseFilename, img16);
  cv::imwrite(out_folder + "/norm_hist_" + outputBaseFilename, img8);

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
