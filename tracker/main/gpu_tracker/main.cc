#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/cudafeatures2d.hpp>


void DrawMatches(const cv::Mat img, const std::vector<cv::KeyPoint>& query_pts,
                 const std::vector<cv::KeyPoint>& train_pts,
                 const std::vector<std::vector<cv::DMatch>>& matches, double max_dist, int max_xdiff) {
  cv::Mat disp_img;
  cv::cvtColor(img, disp_img, cv::COLOR_GRAY2BGR);
  size_t cnt_matches = 0;
  for (size_t i = 0; i < matches.size(); i++) {
    if (matches[i].size() == 0) { continue; }
    if (matches[i][0].distance < max_dist) {
      const cv::Point2f& query_pt = query_pts[matches[i][0].queryIdx].pt;
      const cv::Point2f& train_pt = train_pts[matches[i][0].trainIdx].pt;
      if (std::abs(query_pt.x - train_pt.x) > max_xdiff) continue;
      if (std::abs(query_pt.y - train_pt.y) > (max_xdiff/2)) continue;
      cnt_matches++;
      cv::arrowedLine(disp_img, query_pt, train_pt, cv::Scalar(0,255,0), 1, 8, 0, 0.1);
      //std::cout << query_pts[matches[i].queryIdx].pt << " --> " <<
      //             train_pts[matches[i].trainIdx].pt << "\n";
    }
  }
  std::cout << "Matches = " << cnt_matches << "\n";
  cv::imshow("tracks", disp_img);
  cv::waitKey(0);
}
void DrawMatches(const cv::Mat img, const std::vector<cv::KeyPoint>& query_pts,
                 const std::vector<cv::KeyPoint>& train_pts,
                 const std::vector<cv::DMatch>& matches, double max_dist, int max_xdiff) {
  cv::Mat disp_img;
  cv::cvtColor(img, disp_img, cv::COLOR_GRAY2BGR);
  size_t cnt_matches = 0;
  for (size_t i = 0; i < matches.size(); i++) {
    if (matches[i].distance < max_dist) {
      const cv::Point2f& query_pt = query_pts[matches[i].queryIdx].pt;
      const cv::Point2f& train_pt = train_pts[matches[i].trainIdx].pt;
      if (std::abs(query_pt.x - train_pt.x) > max_xdiff) continue;
      if (std::abs(query_pt.y - train_pt.y) > (max_xdiff/2)) continue;
      cnt_matches++;
      cv::arrowedLine(disp_img, query_pt, train_pt, cv::Scalar(0,255,0), 1, 8, 0, 0.1);
      //std::cout << query_pts[matches[i].queryIdx].pt << " --> " <<
      //             train_pts[matches[i].trainIdx].pt << "\n";
    }
  }
  std::cout << "Matches = " << cnt_matches << "\n";
  cv::imshow("tracks", disp_img);
  cv::waitKey(0);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " img1 img2\n";
    return 1;
  }
  int gpu_id = 0;
  //cv::cuda::printCudaDeviceInfo(gpu_id);
  cv::cuda::setDevice(gpu_id);

  std::string img_path1(argv[1]);
  std::string img_path2(argv[2]);

  cv::Mat img1 = cv::imread(img_path1, cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(img_path2, cv::IMREAD_GRAYSCALE);
  //explicit ORB_CUDA(int nFeatures = 500, float scaleFactor = 1.2f,
  //                   int nLevels = 8, int edgeThreshold = 31,
  //                   int firstLevel = 0, int WTA_K = 2,
  //                   int scoreType = 0, int patchSize = 31);
  // HARRIS_SCORE=0, FAST_SCORE=1

  //cv::cuda::ORB detector(5000, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);
  //cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create(4000, 1.2, 8, 31, 0, 2, cv::cuda::ORB::HARRIS_SCORE, 31);
  //int patch_size = 31;
  //float scale_factor = 1.2;
  //int num_levels = 8;
  int patch_size = 11;
  float scale_factor = 1.1;
  int num_levels = 2;
  float max_dist = 50;
  cv::Ptr<cv::cuda::ORB> detector = cv::cuda::ORB::create(4000, scale_factor, num_levels, patch_size, 0, 2,
      cv::cuda::ORB::HARRIS_SCORE, patch_size);

  cv::cuda::GpuMat gpu_img1, gpu_img2;
  gpu_img1.upload(img1);
  gpu_img2.upload(img2);

  std::vector<cv::KeyPoint> points1, points2;
  cv::cuda::GpuMat gpu_desc1, gpu_desc2;
  detector->detectAndCompute(gpu_img1, cv::cuda::GpuMat(), points1, gpu_desc1);
  detector->detectAndCompute(gpu_img2, cv::cuda::GpuMat(), points2, gpu_desc2);
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
      cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

  std::vector<std::vector<cv::DMatch>> matches;
  matcher->radiusMatch(gpu_desc1, gpu_desc2, matches, max_dist);
  //matcher->knnMatch(gpu_desc1, gpu_desc2, matches, 2);
  //std::vector<cv::DMatch> matches;
  //matcher->match(gpu_desc1, gpu_desc2, matches);

  int num_iter = 20;
  auto start = std::chrono::system_clock::now();  

  //// num_iter = 20 --> 1.36103 sec
  //// 1 - Sync test
  //std::vector<std::vector<cv::DMatch>> matches(num_iter);
  //std::vector<cv::KeyPoint> points1(num_iter), points2(num_iter);
  //cv::cuda::GpuMat gpu_points1(num_iter);
  //cv::cuda::GpuMat gpu_points2(num_iter);
  //cv::cuda::GpuMat gpu_descs1(num_iter);
  //cv::cuda::GpuMat gpu_descs2(num_iter);
  //cv::cuda::GpuMat gpu_matches(num_iter);
  //for (int i = 0; i < num_iter; i++) {
  //detector->detectAndCompute(gpu_img1, cv::cuda::GpuMat(), points1, gpu_descs1);
  //detector->detectAndCompute(gpu_img2, cv::cuda::GpuMat(), points2, gpu_descs2);
  //cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
  //    cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  //matcher->radiusMatch(gpu_descs1, gpu_descs2, matches, max_dist);
  //}
  //// 1 - Sync test end

  //// num_iter = 20 --> 1.34993 sec
  //// 2 - Sync test
  //std::vector<std::vector<std::vector<cv::DMatch>>> matches(num_iter);
  //std::vector<std::vector<cv::KeyPoint>> points1(num_iter), points2(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_points1(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_points2(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_descs1(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_descs2(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_matches(num_iter);
  //for (int i = 0; i < num_iter; i++) {
  //detector->detectAndCompute(gpu_img1, cv::cuda::GpuMat(), points1[i], gpu_descs1[i]);
  //detector->detectAndCompute(gpu_img2, cv::cuda::GpuMat(), points2[i], gpu_descs2[i]);
  //cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
  //    cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  //matcher->radiusMatch(gpu_descs1[i], gpu_descs2[i], matches[i], max_dist);
  //}
  //// Sync test end

  //// num_iter = 20 --> 0.269 sec
  //// 3 - Async test
  //cv::cuda::Stream cuda_stream;
  //std::vector<cv::cuda::GpuMat> gpu_matches(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_points1(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_points2(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_descs1(num_iter);
  //std::vector<cv::cuda::GpuMat> gpu_descs2(num_iter);
  //for (int i = 0; i < num_iter; i++) {
  //  detector->detectAndComputeAsync(gpu_img1, cv::cuda::GpuMat(), gpu_points1[i], gpu_descs1[i], false, cuda_stream);
  //  detector->detectAndComputeAsync(gpu_img2, cv::cuda::GpuMat(), gpu_points2[i], gpu_descs2[i], false, cuda_stream);
  //  //detector->convert(gpu_points1, points1);
  //  //detector->convert(gpu_points2, points2);
  //  cuda_stream.waitForCompletion();
  //  cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
  //      cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  //  matcher->radiusMatchAsync(gpu_descs1[i], gpu_descs2[i], gpu_matches[i], max_dist,
  //                            cv::cuda::GpuMat(), cuda_stream);
  //}
  //// Async test end

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";

  DrawMatches(img1, points1, points2, matches, max_dist, 100);

  //int num_matches = 0;
  //for (size_t i = 0; i < matches.size(); i++) {
  //  if (matches[i].size() > 0)
  //    num_matches++;
  //  //for (auto match : matches[i]) {
  //  //  std::cout << i << " --> " << match.trainIdx << "\tDist = " << match.distance << "\n";
  //  //}
  //    //std::cout << i << " --> " << match.queryIdx << "\n";
  //}

  //std::cout << "Detected num of points1 = " << points1.size() << "\n";
  //std::cout << "Detected num of points2 = " << points2.size() << "\n";
  //std::cout << "Matched = " << num_matches << "\n";
  //cv::Mat pts_host(dst);
  //std::cout << pts_host;

  cv::Mat disp_img;
  //cv::drawMatches(img1, points1, img2, points2, matches, disp_img);
  //cv::drawMatches(img1, points1, img2, points2, matches, disp_img, cv::Scalar::all(-1),
  //                cv::Scalar::all(-1),  std::vector<std::vector<char>>(),
  //                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  //cv::imshow("Display", disp_img);
  //cv::waitKey(0);

  //for (size_t i = 0; i < matches.size(); i++) {
  //  if (matches[i].size() > 0) {
  //    cv::drawMatches(img1, points1, img2, points2, matches[i], disp_img, cv::Scalar::all(-1),
  //                    cv::Scalar::all(-1),  std::vector<char>(),
  //                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  //    cv::imshow("Display", disp_img);
  //    cv::waitKey(0);
  //  }
  //}

  //cv::Mat disp1, disp2;
  //cv::drawKeypoints(img1, points1, disp1);
  //cv::drawKeypoints(img2, points2, disp2);
  //cv::imshow("Display1", disp1);
  //cv::imshow("Display1", disp2);
  //cv::waitKey(0);
  //cv::imshow("Display1", disp);

  //cv::cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
  //cv::Mat result_host = dst;
  //cv::imshow("Result", result_host);
  //cv::waitKey();

  return 0;
}
