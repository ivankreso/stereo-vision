#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
  if (argc != 2)
    return 1;
  std::string img_path(argv[1]);

  std::vector<cv::KeyPoint> points;
  cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);

  //explicit ORB_CUDA(int nFeatures = 500, float scaleFactor = 1.2f,
  //                   int nLevels = 8, int edgeThreshold = 31,
  //                   int firstLevel = 0, int WTA_K = 2,
  //                   int scoreType = 0, int patchSize = 31);
  // HARRIS_SCORE=0, FAST_SCORE=1
  cv::ORB detector(5000, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);
  cv::Mat descriptors;

  auto start = std::chrono::system_clock::now();  
  for (int i = 0; i < 1000; i++)
    detector(img, cv::Mat(), points, descriptors);
  auto end = std::chrono::system_clock::now();
  //auto elapsed = end - start;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";

  //detector(src, cv::cuda::GpuMat(), dst, descriptors);
  std::cout << "Detected num of points = " << points.size() << "\n";
  //cv::Mat pts_host(dst);
  //std::cout << pts_host;
  cv::Mat disp;
  cv::drawKeypoints(img, points, disp);
  cv::imshow("Display", disp);
  cv::waitKey(0);

  //cv::cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
  //cv::Mat result_host = dst;
  //cv::imshow("Result", result_host);
  //cv::waitKey();

  return 0;
}
