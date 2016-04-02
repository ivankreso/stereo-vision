#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>


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
  std::string img_path1(argv[1]);
  std::string img_path2(argv[2]);

  cv::Mat img1 = cv::imread(img_path1, cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(img_path2, cv::IMREAD_GRAYSCALE);

  //static Ptr< AgastFeatureDetector > 	create (int threshold=10, bool nonmaxSuppression=true,
  // int type=AgastFeatureDetector::OAST_9_16)
  //  Ptr<FREAK> cv::xfeatures2d::FREAK::create 	( 	bool  	orientationNormalized = true,
  //	bool  	scaleNormalized = true,
  //	float  	patternScale = 22.0f,
  //	int  	nOctaves = 4,
  //	const std::vector< int > &  	selectedPairs = std::vector< int >() 

  //cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::AgastFeatureDetector::create();
  cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::AgastFeatureDetector::create(
      10, true, cv::xfeatures2d::AgastFeatureDetector::OAST_9_16);
      //10, true, cv::xfeatures2d::AgastFeatureDetector::AGAST_7_12s);
  cv::Ptr<cv::Feature2D> descriptor = cv::xfeatures2d::FREAK::create(true, true, 22.0, 4);

  auto start = std::chrono::system_clock::now();
  std::vector<cv::KeyPoint> points1, points2;
  detector->detect(img1, points1);
  detector->detect(img2, points2);
  std::cout << "Detected features = " << points1.size() << " -- " << points2.size() << "\n";
  cv::Mat desc1, desc2;
  descriptor->compute(img1, points1, desc1);
  descriptor->compute(img2, points2, desc2);

  int max_dist = 50;
  //cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create("BruteForce-Hamming");
  bool cross_check = false;
  cv::BFMatcher matcher(cv::NORM_HAMMING, cross_check);
  std::vector<std::vector<cv::DMatch>> matches;
  matcher.radiusMatch(desc1, desc2, matches, max_dist);

  //matcher->knnMatch(gpu_desc1, gpu_desc2, matches, 2);
  //std::vector<cv::DMatch> matches;
  //matcher->match(gpu_desc1, gpu_desc2, matches);

  // FLANN does not work
  //cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20,10,2));
  //matcher.add(desc1);
  //matcher.train();
  ////std::vector<std::vector<cv::DMatch>> matches;
  //std::vector<cv::DMatch> matches;
  ////matcher.knnMatch(desc2, desc1, matches, 2);
  //matcher.match(desc2, matches);

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";

  cv::Mat disp1, disp2;
  cv::drawKeypoints(img1, points1, disp1);
  cv::drawKeypoints(img2, points2, disp2);
  cv::imshow("Display1", disp1);
  cv::imshow("Display1", disp2);
  //cv::waitKey(0);
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

  //cv::Mat disp_img;
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

  //cv::imshow("Display1", disp);

  //cv::cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
  //cv::Mat result_host = dst;
  //cv::imshow("Result", result_host);
  //cv::waitKey();

  return 0;
}
