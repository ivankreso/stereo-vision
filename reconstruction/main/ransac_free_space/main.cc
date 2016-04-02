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
#include <opencv2/imgproc/imgproc.hpp>
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
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/filters/passthrough.h>
#include <vtkPlaneSource.h>
#include <vtkImplicitPlaneWidget.h>

#include "../../../core/format_helper.h"
#include "../../../core/math_helper.h"

cv::Mat img_left, img_disp;
pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
std::map<std::tuple<int,int>, int> pts_map;
std::vector<std::tuple<int,int>> road;


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

void triangulate(const double (&cam_params)[5], double x, double y, double disp, cv::Mat& pt3d)
{
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


void getSignedDistancesToModel(const Eigen::Vector4f& plane, const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                               std::vector<double>& distances)
{
  distances.assign(pc->points.size(), 0.0);
  double a = plane[0];
  double b = plane[1];
  double c = plane[2];
  double d = plane[3];
  double norm = std::sqrt(a*a + b*b + c*c);
  for(size_t i = 0; i < pc->points.size(); i++) {
    pcl::PointXYZ pt = pc->points[i];
    double p = a*pt.x + b*pt.y + c*pt.z + d;
    distances[i] = p / norm;
  }
}

void estimateFreeSpace(const std::vector<std::string>& imagelist, const std::vector<std::string>& disp_imglist,
                      const std::string& gt_file, const std::string& cparams_file,
                      const std::string& source_folder, const std::string& disp_folder, const std::string& output_folder)
{
  double cam_params[5];
  core::FormatHelper::readCameraParams(cparams_file, cam_params);

  // KITTI 07
  // meadow on right
  size_t frame_num = 350;
  //size_t frame_num = 395;
  //size_t frame_num = 363;
  //size_t frame_num = 371;
  //size_t frame_num = 510;
  // car in front
  //size_t frame_num = 995;
  // traffic sign
  //size_t frame_num = 442;
  //size_t frame_num = 620;

  // KITTI 02
  //size_t frame_num = 1820;
  //size_t frame_num = 184;

  bool done_looping = false;
  while(!done_looping) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr road_patch_pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  std::vector<std::tuple<int,int>> pc2img;
  std::vector<std::tuple<int,int>> patch2img;
  std::vector<int> small_obstacles;
  std::vector<int> super_obstacles;

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
  double min_disp = 1.0;
  //int img_step = 1;
  //for(size_t i = 0; i < end_frame; i++) {
  //for(size_t i = frame_num; i < end_frame; i++) {

  //std::map<std::tuple<int,int>, int> pts_map;

  std::cout << "Frame: " << frame_num << "\n";
  size_t pt_cnt = 0;
  for(size_t i = frame_num; i < frame_num+1; i++) {
    //core::FormatHelper::readNextMatrixKitti(gt_fp, Rt);
    //if(!(i % img_step == 0) || i < frame_num) continue;
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

          //if(pt.x < -3.0 || pt.x > 6.0) continue;
          //if(pt.y < -5.0 || pt.y > 0.0) continue;

          point_cloud->points.push_back(pt);
          pcl::PointXYZRGB point;
          point.x = pt3d.at<double>(0);
          point.y = pt3d.at<double>(1);
          point.z = pt3d.at<double>(2);
          point.r = img_left.at<uint8_t>(y,x);
          point.g = img_left.at<uint8_t>(y,x);
          point.b = img_left.at<uint8_t>(y,x);
          point_cloud_rgb->points.push_back(point);
          pc2img.push_back(std::make_tuple(x,y));
          pts_map.insert(std::make_pair(std::make_tuple(x,y), pt_cnt));
          pt_cnt++;
        }
      }
    }
    //imshow("disparity", img_disp);
    //waitKey(0);
  }
  std::cout << "Point cloud size: " << point_cloud->points.size() << "\n";
  //std::cout << "Point cloud size: " << point_cloud_rgb->points.size() << "\n";
  point_cloud->width = point_cloud->points.size();
  point_cloud->height = 1;
  point_cloud_rgb->width = point_cloud_rgb->points.size();
  point_cloud_rgb->height = 1;

  pcl::io::savePCDFile("pc_orig_rgb.pcd", *point_cloud_rgb);

  //// with smoothing - fuckes up the camera orientation
  //// Create a KD-Tree
  //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  //// Output has the PointNormal type in order to store the normals calculated by MLS
  ////pcl::PointCloud<pcl::PointNormal> normals;
  ////pcl::PointCloud<pcl::PointNormal>::Ptr normals (new pcl::PointCloud<pcl::PointNormal>);
  //// Init object (second point type is for the normals, even if unused)
  //pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
  //mls.setComputeNormals(true);
  //// Set parameters
  //mls.setInputCloud(point_cloud);
  //mls.setPolynomialFit(true);
  //mls.setSearchMethod(tree);
  //mls.setSearchRadius(0.2);       // 0.15 -
  //// Reconstruct
  //mls.process(*cloud_normals);
  //for(int i = 0; i < cloud_normals->points.size(); i++) {
  //  point_cloud->points[i].x = cloud_normals->points[i].x;
  //  point_cloud->points[i].y = cloud_normals->points[i].y;
  //  point_cloud->points[i].z = cloud_normals->points[i].z;
  //  //point_cloud_rgb->points[i].x = cloud_normals->points[i].x;
  //  //point_cloud_rgb->points[i].y = cloud_normals->points[i].y;
  //  //point_cloud_rgb->points[i].z = cloud_normals->points[i].z;
  //}
  ////for(int i = 0; i < cloud_normals->points.size(); i++) {
  ////  for(int j = 0; j < 3; j++) {
  ////    cloud_normals->points[i].normal[j] = - cloud_normals->points[i].normal[j];
  ////  }
  ////}

  for(size_t i = 0; i < point_cloud->points.size(); i++) {
    pcl::PointXYZ pt = point_cloud->points[i];
    if(pt.x > -1.5 && pt.x < 3.5 && pt.z > 4.0 && pt.z < 20.0 && pt.y > -3.0 && pt.y < 0.0) {
      road_patch_pc->points.push_back(pt);
      patch2img.push_back(pc2img[i]);
    }
  }
  road_patch_pc->width = road_patch_pc->points.size();
  road_patch_pc->height = 1;
  pcl::io::savePCDFile("road_patch_pc.pcd", *road_patch_pc);

  //cv::Rect plane_patch(460, 300)
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100000);
  seg.setDistanceThreshold(0.1);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  seg.setInputCloud(road_patch_pc->makeShared());
  seg.segment(*inliers, *coefficients);

  std::vector<double> distances;
  Eigen::Vector4f model_coeff;
  for(int i = 0; i < 4; i++)
    model_coeff[i] = coefficients->values[i];
  //pcl::SampleConsensusModelPlane<pcl::PointXYZ> plane_model(point_cloud);
  //plane_model.getDistancesToModel(model_coeff, distances);
  getSignedDistancesToModel(model_coeff, point_cloud, distances);
  // color the obstacles
  //for(size_t i = 0; i < point_cloud->points.size(); i++) {
  //  //if(distances[i] < -2.0)
  //  //  std::cout << distances[i] << " - dist\n";
  //  if(distances[i] < -0.3) {
  //    point_cloud_rgb->points[i].r = 255.0;
  //    point_cloud_rgb->points[i].g = 0.0;
  //    point_cloud_rgb->points[i].b = 0.0;
  //    super_obstacles.push_back(i);
  //  }
  //  else if(distances[i] < -0.05 && distances[i] >= -0.3) {
  //    point_cloud_rgb->points[i].r = 0.0;
  //    point_cloud_rgb->points[i].g = 255.0;
  //    point_cloud_rgb->points[i].b = 0.0;
  //    small_obstacles.push_back(i);
  //  }
  //  //double dist = getDistanceFromPlane(point_cloud->points[i], coefficients);
  //}


  // render point cloud
  pcl::visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setBackgroundColor(0, 0, 0);
  //pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr(&point_cloud);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> hcolor(point_cloud_rgb);
  viewer.addPointCloud<pcl::PointXYZRGB>(point_cloud_rgb, hcolor, "reconstruction");
  pcl::PointCloud<pcl::PointXYZ>::Ptr poly_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  createPolyFromPlane(coefficients, poly_cloud);
  viewer.addPolygon<pcl::PointXYZ>(poly_cloud, 0, 0, 180, "poly1");
  //viewer.addPolygon<pcl::PointXYZ>(poly_cloud, 100, 100, 100, "poly1");
  viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                     pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, "poly1");
  viewer.addCoordinateSystem(1.0);
  viewer.initCameraParameters();
  //Main loop
  while (!viewer.wasStopped()) {
    //viewer.spinOnce(100);        // 100
    viewer.spin();        // 100
    //boost::this_thread::sleep(boost::posix_time::microseconds(10000));
  }
  viewer.close();

  //// we need placeholders for threading support
  cv::namedWindow("image");
  cv::namedWindow("patch_image");
  cv::namedWindow("free_space_image");
  //cv::namedWindow("road_image");
  //cv::imshow("image", img_disp);

  cv::imshow("image", img_left);
  cv::Mat img_patch = img_left.clone();
  for(int i = 0; i < patch2img.size(); i++) {
    int x = std::get<0>(patch2img[i]);
    int y = std::get<1>(patch2img[i]);
    img_patch.at<uint8_t>(y,x) = 0.0;
  }
  cv::imshow("patch_image", img_patch);

  //cv::Mat img_obstacles = img_left.clone();
  cv::Mat img_obstacles;
  cv::cvtColor(img_left, img_obstacles, cv::COLOR_GRAY2RGB);
  for(int i = 0; i < small_obstacles.size(); i++) {
    int x = std::get<0>(pc2img[small_obstacles[i]]);
    int y = std::get<1>(pc2img[small_obstacles[i]]);
    //img_obstacles.at<uint8_t>(y,x) = 0.0;
    img_obstacles.at<cv::Vec3b>(y,x)[0] = 0.0;
    img_obstacles.at<cv::Vec3b>(y,x)[1] = 0.0;
    img_obstacles.at<cv::Vec3b>(y,x)[2] = 255.0;
  }
  for(int i = 0; i < super_obstacles.size(); i++) {
    int x = std::get<0>(pc2img[super_obstacles[i]]);
    int y = std::get<1>(pc2img[super_obstacles[i]]);
    //img_obstacles.at<uint8_t>(y,x) = 0.0;
    img_obstacles.at<cv::Vec3b>(y,x)[0] = 0.0;
    img_obstacles.at<cv::Vec3b>(y,x)[1] = 255.0;
    img_obstacles.at<cv::Vec3b>(y,x)[2] = 0.0;
  }
  cv::imshow("free_space_image", img_obstacles);
  //std::stringstream strnum;
  //strnum << std::setw(5) << std::setfill('0') << frame_num;
  //cv::imwrite("out/image_" + strnum.str() + ".png", img_obstacles);
  //cv::waitKey(10);
  frame_num++;
  while(true) {
    int key = cv::waitKey(0);
    //std::cout << key << "\n";
    if(key == 27 || key == 1048603) {
      done_looping = true;
      break;
    }
    else if(key == 65362) {
      frame_num++;
      break;
    }
    else if(key == 65364) {
      frame_num--;
      break;
    }
  }

  } // end big loop
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

  estimateFreeSpace(imagelist, disp_imagelist, gt_file, cam_params_file, source_folder, disp_folder, output_folder);

  return 0;
}
