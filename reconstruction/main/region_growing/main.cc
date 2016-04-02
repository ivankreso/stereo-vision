#include <vector>
#include <iostream>
#include <string>
#include <tuple>
#include <deque>
#include <map>
#include <set>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/passthrough.h>
#include <vtkPlaneSource.h>
#include <vtkImplicitPlaneWidget.h>

#include "../../../core/format_helper.h"
#include "../../../core/math_helper.h"

static void onMouse(int event, int x, int y, int, void*);

cv::Mat img_left, img_disp;
pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
std::map<std::tuple<int,int>, int> pts_map;
std::vector<std::tuple<int,int>> road;

//template<class T>
//bool exists(std::tuple<int,int>& key, T& set)
//{
//  if(set.find(key) == set.end())
//    return false;
//  return true;
//}
bool exists(std::tuple<int,int>& key, std::set<std::tuple<int,int>>& set)
{
  if(set.find(key) == set.end())
    return false;
  return true;
}
bool exists(std::tuple<int,int>& key, std::map<std::tuple<int,int>,int>& map)
{
  if(map.find(key) == map.end())
    return false;
  return true;
}

void addPointNeighborhood(std::tuple<int,int>& pixel, std::set<std::tuple<int,int>>& searched_pts,
                          std::deque<std::tuple<int,int>>& nbh)
{
  int x = std::get<0>(pixel);
  int y = std::get<1>(pixel);
  for(int i = -1; i < 2; i+=2) {
    auto pt = std::make_tuple(x+i, y);
    if(!exists(pt, searched_pts)) {
      nbh.push_back(pt);
      searched_pts.insert(pt);
    }
    pt = std::make_tuple(x, y+i);
    if(!exists(pt, searched_pts)) {
      nbh.push_back(pt);
      searched_pts.insert(pt);
    }
  }
}

void addPointNeighborhood(int img_width, int img_height, std::tuple<int,int>& pixel,
                          std::set<std::tuple<int,int>>& searched_pts, std::deque<std::tuple<int,int>>& nbh)
{
  int x = std::get<0>(pixel);
  int y = std::get<1>(pixel);
  for(int i = -1; i < 2; i+=2) {
    auto pt = std::make_tuple(x+i, y);
    if(!exists(pt, searched_pts)) {
      nbh.push_back(pt);
      searched_pts.insert(pt);
    }
    pt = std::make_tuple(x, y+i);
    if(!exists(pt, searched_pts)) {
      nbh.push_back(pt);
      searched_pts.insert(pt);
    }
  }
}

template<typename T>
T getAngleNorm(T *vec1, T *vec2)
{
  T dot = 0.0;
  for(int i = 0; i < 3; i++)
    dot += vec1[i] * vec2[i];
  return std::acos(dot);
}

//void segmentRoad(cv::Mat& img_left, pcl::PointCloud<pcl::PointNormal>::Ptr point_cloud)
void segmentRoad(int x, int y, std::vector<std::tuple<int,int>>& road)
{
  if (pts_map.find(std::make_tuple(x,y)) == pts_map.end()) {
    std::cout << "Soory, no data for this point. Try again.\n";
    return;
  }
  //while(pts_map.find(std::make_tuple(x,y)) == pts_map.end()) {
  //  std::vector<std::tuple<int,int>> nbh = getPointNeighborhood(x, y);
  //}
  const double PI = 3.141592653589793;
  float global_angle[3] = {0.0, 1.0, 0.0};
  //float angle_threshold = (20.0 / 180.0) * PI;
  float angle_threshold = (18.0 / 180.0) * PI;
  //float angle_threshold = (15.0 / 180.0) * PI;

  //std::vector<std::tuple<int,int>> road;
  road.clear();
  std::deque<std::tuple<int,int>> search_region;
  std::set<std::tuple<int,int>> explored_region;
  //road.push_back(std::make_tuple(x,y));
  search_region.push_back(std::make_tuple(x,y));
  //addPointNeighborhood(x, y, searched_pts, search_region);

  while(search_region.size() > 0) {
    // get next pixel in search region
    auto px = search_region.front();
    search_region.pop_front();
    if(exists(px, pts_map)) {
      int idx = pts_map.at(px);
      //for(int i = 0; i < 3; i++) {
      //  std::cout << cloud_normals->points[idx].normal[i] << " ";
      //}
      //std::cout << "\n";
      float angle = getAngleNorm<float>(cloud_normals->points[idx].normal, global_angle);
      //std::cout << "Angle(normal, y-axis) = " << angle << "  (threshold = " << angle_threshold << ")\n";
      if(std::isnan(angle))
        std::cout << "Warning: Normal is NaN -- " << std::get<0>(px) << "-" << std::get<1>(px) << "\n";
      // if the 3D point normal is below threshold add the point to inliers
      if(angle <= angle_threshold) {
        road.push_back(px);
        // expand the search region
        addPointNeighborhood(px, explored_region, search_region);
        //addPointNeighborhood(img_width, img_height, px, explored_region, search_region);
      }
    }
    // if 3D data doesn't exist add it to the region but do not expand further from it
    else {
      road.push_back(px);
      //addPointNeighborhood(img_width, img_height, px, explored_region, search_region);
    }
  }
  //std::cout << "Seed: " << x << " - " << y << "\n";
}

void redrawImage()
{
  //cv::Mat img_draw = img_disp.clone();
  cv::Mat img_draw = img_left.clone();
  //img_draw = img_left.clone();
  for(size_t i = 0; i < road.size(); i++) {
    std::tuple<int,int> pxpos = road[i];
    int x = std::get<0>(pxpos);
    int y = std::get<1>(pxpos);
    //std::cout << x << " - " << y << "\n";
    img_draw.at<uint8_t>(y,x) = 0;
  }
  cv::imshow("road_image", img_draw);
  //cv::waitKey(0);
}

static void onMouse(int event, int x, int y, int, void*)
{
  if(event != cv::EVENT_LBUTTONDOWN)
    return;

  std::cout << "You clicked: " << x << " - " << y << "\n";
  segmentRoad(x, y, road);
  redrawImage();
}




void createPolyFromPlane(const pcl::ModelCoefficients::Ptr coefficients,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr poly_cloud)
{
  double a = coefficients->values[0];
  double b = coefficients->values[1];
  double c = coefficients->values[2];
  double d = coefficients->values[3];

  double x[4] = {-20, 20, 20, -20};
  double z[4] = {  0,  0, 100, 100};
  for(int i = 0; i < 4; i++) {
    double y = (a*x[i] + c*z[i] + d) / (-b);
    pcl::PointXYZ pt(x[i], y, z[i]);
    poly_cloud->points.push_back(pt);
  }
}

void triangulate(const double (&cam_params)[5], double x, double y, double disp, cv::Mat& pt3d) {
  double f = cam_params[0];
  double cx = cam_params[2];
  double cy = cam_params[3];
  double b = cam_params[4];

  // add - in front for x and y just to set the positive y to skies
  pt3d.at<double>(0) = -(x - cx) * b / disp;
  pt3d.at<double>(1) = -(y - cy) * b / disp;
  pt3d.at<double>(2) = f * b / disp;
  pt3d.at<double>(3) = 1.0;
}

void renderPointCloud(const std::vector<std::string>& imagelist, const std::vector<std::string>& disp_imglist,
                      const std::string& gt_file, const std::string& cparams_file,
                      const std::string& source_folder, const std::string& disp_folder, const std::string& output_folder)
{
  double cam_params[5];
  core::FormatHelper::readCameraParams(cparams_file, cam_params);

  // meadow on right
  size_t start_frame = 350;
  //size_t start_frame = 359;
  //size_t start_frame = 363;
  //size_t start_frame = 371;
  //size_t start_frame = 510;
  // car in front
  //size_t start_frame = 995;
  // traffic sign
  //size_t start_frame = 442;
  //size_t start_frame = 620;

  bool done_looping = false;
  while(!done_looping) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  //std::vector<std::tuple<int,int>> points_img;

  // clear the global variables
  cloud_normals->clear();
  pts_map.clear();
  road.clear();

  ifstream gt_fp(gt_file);
  cv::Mat Rt = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_64F);
  //cv::Mat img_left;
  cv::Mat mat_disp;
  //double min_disp = 40.0;
  double min_disp = 30.0;
  //int img_step = 1;
  //for(size_t i = 0; i < end_frame; i++) {
  //for(size_t i = start_frame; i < end_frame; i++) {

  //std::map<std::tuple<int,int>, int> pts_map;

  std::cout << "Frame: " << start_frame << "\n";
  size_t pt_cnt = 0;
  for(size_t i = start_frame; i < start_frame+1; i++) {
    //core::FormatHelper::readNextMatrixKitti(gt_fp, Rt);
    //if(!(i % img_step == 0) || i < start_frame) continue;
    std::cout << "----------------\n" << source_folder + imagelist[2*i] << "\n";
    img_left = cv::imread(source_folder + imagelist[2*i], CV_LOAD_IMAGE_GRAYSCALE);
    //string fname = output_folder + imagelist[i].substr(9, imagelist[i].size()-13) + "_disp.png";
    //std::string fname = output_folder + disp_imglist[i];
    img_disp = cv::imread(disp_folder + "/img/" + disp_imglist[i], CV_LOAD_IMAGE_GRAYSCALE);
    std::string mat_fname = disp_folder + "mat/" + disp_imglist[i].substr(0,11) + ".yml";
    std::cout << "reading: " << mat_fname << "\n";
    cv::FileStorage mat_file(mat_fname, cv::FileStorage::READ);
    mat_file["disparity_matrix"] >> mat_disp;
    for(int y = 0; y < mat_disp.rows; y++) {
      for(int x = 0; x < mat_disp.cols; x++) {
        float disp = mat_disp.at<float>(y,x);
        if(disp > min_disp) {
          triangulate(cam_params, x, y, disp, pt3d);
          //pt3d = Rt * pt3d;
          //std::cout << pt3d << "\n\n";
          pcl::PointXYZ pt;
          pt.x = pt3d.at<double>(0);
          pt.y = pt3d.at<double>(1);
          pt.z = pt3d.at<double>(2);
          if(pt.x < -3.0 || pt.x > 6.0) continue;
          if(pt.y < -5.0 || pt.y > 0.0) continue;

          point_cloud->points.push_back(pt);
          pcl::PointXYZRGB point;
          point.x = pt3d.at<double>(0);
          point.y = pt3d.at<double>(1);
          point.z = pt3d.at<double>(2);
          point.r = img_left.at<uint8_t>(y,x);
          point.g = img_left.at<uint8_t>(y,x);
          point.b = img_left.at<uint8_t>(y,x);
          point_cloud_rgb->points.push_back(point);
          //points_img.push_back(std::make_tuple(x,y));
          pts_map.insert(std::make_pair(std::make_tuple(x,y), pt_cnt));
          pt_cnt++;
        }
      }
    }
    //imshow("disparity", img_disp);
    //waitKey(0);
  }
  std::cout << "Point cloud size: " << point_cloud->points.size() << "\n";
  point_cloud->width = point_cloud->points.size();
  point_cloud->height = 1;
  point_cloud_rgb->width = point_cloud->points.size();
  point_cloud_rgb->height = 1;
  pcl::io::savePCDFile("pc_orig_rgb.pcd", *point_cloud_rgb);

  //// with smoothing
  //// Create a KD-Tree
  //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  //// Output has the PointNormal type in order to store the normals calculated by MLS
  ////pcl::PointCloud<pcl::PointNormal> normals;
  ////pcl::PointCloud<pcl::PointNormal>::Ptr normals (new pcl::PointCloud<pcl::PointNormal>);
  //// Init object (second point type is for the normals, even if unused)
  //pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
  //mls.setComputeNormals(true);
  //// Set parameters
  //mls.setInputCloud (point_cloud);
  //mls.setPolynomialFit(true);
  //mls.setSearchMethod(tree);
  //mls.setSearchRadius(0.2);       // 0.15 -
  //// Reconstruct
  //mls.process(*cloud_normals);
  //for(int i = 0; i < cloud_normals->points.size(); i++) {
  //  for(int j = 0; j < 3; j++) {
  //    cloud_normals->points[i].normal[j] = - cloud_normals->points[i].normal[j];
  //  }
  //}

  // without smoothing
  // Create the normal estimation class, and pass the input dataset to it
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  //pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (point_cloud);
  // Create an empty kdtree representation, and pass it to the normal estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  // Output datasets
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  // Use all neighbors in a sphere of radius 3cm
  //ne.setRadiusSearch (0.03);
  ne.setRadiusSearch(0.1);         // the best - 0.1
  //ne.setRadiusSearch(0.08);         // the best - 0.1, works good for 0.08-0.09
  // Compute the features
  ne.compute (*normals);
  //pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields(*point_cloud, *normals, *cloud_normals);

  pcl::io::savePCDFile("pc_with_normals.pcd", *cloud_normals);
  // we need placeholders for threading support
  cv::namedWindow("image");
  cv::namedWindow("road_image");
  cv::imshow("image", img_disp);
  //cv::imshow("image", img_left);
  cv::setMouseCallback("image", onMouse);
  //cv::waitKey(0);
  while(true) {
    int key = cv::waitKey(0);
    //std::cout << key << "\n";
    if(key == 27 || key == 1048603) {
      done_looping = true;
      break;
    }
    else if(key == 65362) {
      start_frame++;
      break;
    }
    else if(key == 65364) {
      start_frame--;
      break;
    }
  }

  } // end big loop
  //while(key != 27 && key != 1048603) {
  //      key = cv::waitKey(0);
  //}
}



int main(int argc, char** argv)
{
  std::string config_file;
  std::string imagelistfn, disp_imagelistfn;
  std::string cam_params_file;
  std::string source_folder;
  std::string disp_folder, gt_file;
  std::string output_folder;

  try {
    po::options_description generic("Generic options");
    generic.add_options()
      ("help", "produce help message")
      ("config,c", po::value<std::string>(&config_file)->default_value("config.txt"), "config filename")
      ;
    po::options_description config("Config file options");
    config.add_options()
      ("camera_params,p", po::value<std::string>(&cam_params_file)->default_value("camera_params.txt"),
       "camera params file")
      ("source_folder,s", po::value<std::string>(&source_folder), "folder with source")
      ("output_folder,o", po::value<std::string>(&output_folder), "folder for output")
      ("disp_folder,d", po::value<std::string>(&disp_folder), "folder for output")
      ("groundtruth,g", po::value<std::string>(&gt_file), "groundtruth file")
      ("imglist,l", po::value<std::string>(&imagelistfn), "file with image list")
      ("disp_imglist,dl", po::value<std::string>(&disp_imagelistfn), "file with image list");

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config);

    po::options_description config_file_options;
    config_file_options.add(config);
    po::variables_map vm;
    //po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    notify(vm);
    if(vm.count("help")) {
      std::cout << generic;
      std::cout << config;
      return 0;
    }

    ifstream ifs(config_file.c_str());
    if (!ifs) {
      std::cout << "can not open config file: " << config_file << "\n";
      std::cout << generic;
      std::cout << config;
      return 0;
    }
    else {
      po::store(parse_config_file(ifs, config_file_options), vm);
      notify(vm);
    }
    std::cout << "Configuring done, using:" << endl;

    if(vm.count("camera_params")) {
      std::cout << "Camera params: ";
      std::cout << cam_params_file << endl;
    }
    if(vm.count("source_folder")) {
      std::cout << "Source folder: ";
      std::cout << source_folder << endl;
    }
    if(vm.count("output_folder")) {
      std::cout << "Output folder: ";
      std::cout << output_folder << endl;
    }
    if(vm.count("imglist")) {
      std::cout << "Image list file: ";
      std::cout << imagelistfn << endl;
    }
  }
  catch(std::exception& e) {
    std::cout << e.what() << "\n";
    return 1;
  }

  if(imagelistfn == "")
  {
    std::cout << "error: no xml image list given." << endl;
    return -1;
  }
  if(source_folder == "")
  {
    std::cout << "error: no output or source folder given." << endl;
    return -1;
  }

  std::vector<std::string> imagelist, disp_imagelist;
  bool ok = core::FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
  {
    std::cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
    return -1;
  }
  ok = core::FormatHelper::readStringList(disp_imagelistfn, disp_imagelist);

  renderPointCloud(imagelist, disp_imagelist, gt_file, cam_params_file, source_folder, disp_folder, output_folder);

  return 0;
}
