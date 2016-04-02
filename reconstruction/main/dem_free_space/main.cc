#include <vector>
#include <iostream>
#include <string>
#include <tuple>
#include <deque>
#include <map>
#include <set>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

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
#include "../../base/dem.h"
#include "../../base/dem_voting.h"
#include "../../base/math_helper.h"
#include "../../base/draw_helper.h"


void estimateFreeSpace(const std::vector<std::string>& imagelist, const std::vector<std::string>& disp_imglist,
                      const std::string& gt_file, const std::string& cparams_file,
                      const std::string& source_folder, const std::string& disp_folder,
                      const std::string& output_folder)
{
  double cam_params[5];
  core::FormatHelper::readCameraParams(cparams_file, cam_params);

  cv::Mat img_left, img_disp;
  //pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
  //std::map<std::tuple<int,int>, int> pts_map;
  //std::vector<std::tuple<int,int>> road;
  
  //size_t frame_num = 0;
  // KITTI 07
  // meadow on right
  //size_t frame_num = 344;
  //size_t frame_end = 440;
  int img_cnt = 0;

  //size_t frame_num = 344;
  //size_t frame_num = 420;
  //size_t frame_num = 363;
  //size_t frame_num = 371;
  //size_t frame_num = 510;
  // car in front
  //size_t frame_num = 995;
  // traffic sign
  //size_t frame_num = 442;
  //size_t frame_num = 620;
  // cars
  //size_t frame_num = 99;

  // KITTI 02
  size_t frame_num = 1820;
  size_t frame_end = 1975;

  bool done_looping = false;
  while(!done_looping && frame_num <= frame_end) {

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr road_patch_pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
  std::vector<std::tuple<int,int>> pc2img;
  std::vector<std::tuple<int,int>> patch2img;
  std::vector<int> small_obstacles;
  std::vector<int> super_obstacles;

  // clear the global variables
  //cloud_normals->clear();
  //pts_map.clear();
  //road.clear();

  ifstream gt_fp(gt_file);
  cv::Mat Rt = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_64F);
  //cv::Mat img_left;
  cv::Mat mat_disp;
  //double min_disp = 40.0;
  //double min_disp = 30.0;
  double min_disp = 1.0;
  //int img_step = 1;
  //for(size_t i = 0; i < end_frame; i++) {
  //for(size_t i = frame_num; i < end_frame; i++) {

  //std::map<std::tuple<int,int>, int> pts_map;

  std::cout << "Frame: " << frame_num << "\n";
  //for(size_t i = frame_num; i < frame_num+1; i++) {
  for(size_t i = img_cnt; i < img_cnt+1; i++) {
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
          recon::MathHelper::triangulate(cam_params, x, y, disp, pt3d);
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
  //pcl::io::savePCDFile("pc_orig_rgb.pcd", *point_cloud_rgb);

  for(size_t i = 0; i < point_cloud->points.size(); i++) {
    pcl::PointXYZ pt = point_cloud->points[i];
    if(pt.x > -3.5 && pt.x < 1.5 && pt.z > 4.0 && pt.z < 15.0 && pt.y > 0.0 && pt.y < 3.0) {
      road_patch_pc->points.push_back(pt);
      patch2img.push_back(pc2img[i]);
    }
  }
  road_patch_pc->width = road_patch_pc->points.size();
  road_patch_pc->height = 1;
  //pcl::io::savePCDFile("road_patch_pc.pcd", *road_patch_pc);

  //cv::Rect plane_patch(460, 300)
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  //seg.setMaxIterations(100000);
  seg.setMaxIterations(10000);
  seg.setDistanceThreshold(0.1);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  seg.setInputCloud(road_patch_pc->makeShared());
  seg.segment(*inliers, *coefficients);

  Eigen::Vector4d plane_model;
  std::cout << "Plane equation:\n"; 
  for(int i = 0; i < 4; i++) {
    // invert the normal on positive Y size
    //coefficients->values[i] = - coefficients->values[i];
    plane_model[i] = coefficients->values[i];
  }
  // force normal to always look up to the skies
  if(plane_model[3] < 0.0)
    plane_model = - plane_model;
  std::cout << "Ground plane:\n" << plane_model << "\n";

  //Eigen::Matrix4d transform2dem;
  //setTransform2DEM(plane_model, transform2dem);
  //// transform the point to the groud plane coord system
  //pcl::transformPointCloud();

  //pcl::SampleConsensusModelPlane<pcl::PointXYZ> plane_model(point_cloud);
  //plane_model.getDistancesToModel(model_coeff, distances);
  std::vector<double> distances;
  //recon::MathHelper::getSignedDistancesToModel(plane_model, point_cloud, distances);
  recon::MathHelper::getDistancesToModel(plane_model, point_cloud, distances);
  for(size_t i = 0; i < point_cloud->points.size(); i++) {
    if(distances[i] > SUPER_OBSTACLE_THR) {
      point_cloud_rgb->points[i].r = 255;
      point_cloud_rgb->points[i].g = 0;
      point_cloud_rgb->points[i].b = 0;
      super_obstacles.push_back(i);
    }
    else if(distances[i] > OBSTACLE_THR && distances[i] <= SUPER_OBSTACLE_THR) {
      point_cloud_rgb->points[i].r = 0;
      point_cloud_rgb->points[i].g = 255;
      point_cloud_rgb->points[i].b = 0;
      small_obstacles.push_back(i);
    }
  }

  // 14 x 40, papers-13x40
  // 20 x 20 cm cell size
  int cells_x = 70;
  int cells_z = 75; // 15
  //int cells_z = 100; // 20
  //int cells_x = 140;
  //int cells_z = 300;
  double elevation_limit = 2.0;
  //recon::DEM dem(-7.0, 7.0, 0.0, 30.0, cells_x, cells_z, elevation_limit);
  std::vector<double> class_thresh = { OBSTACLE_THR, SUPER_OBSTACLE_THR };
  int num_classes = 3;
  recon::DEMvoting dem(-7.0, 7.0, 0.0, 15.0, cells_x, cells_z, elevation_limit, num_classes, class_thresh);

  dem.update(point_cloud, distances);
  std::vector<Eigen::Vector3d> centers;
  std::vector<double> center_elev;
  std::vector<int> class_ids;
  for(size_t i = 0; i < cells_z; i++) {
    for(size_t j = 0; j < cells_x; j++) {
      Eigen::Vector3d center;
      int class_id;
      double elev = dem.getCell(j, i, plane_model, center, class_id);
      centers.push_back(center);
      class_ids.push_back(class_id);
      //center_elev.push_back(elev);
    }
  }

  // project centers to image
  std::vector<Eigen::Vector2d> dem_projs;
  for(size_t i = 0; i < centers.size(); i++) {
    Eigen::Vector2d proj;
    recon::MathHelper::projectPoint(cam_params, centers[i], proj);
    dem_projs.push_back(proj);
  }

  // draw them
  cv::Mat img_dem;
  cv::cvtColor(img_left, img_dem, cv::COLOR_GRAY2RGB);

  //drawDEM(dem, dem_projs, center_elev, img_dem);
  recon::DrawHelper::drawDEM(dem, dem_projs, class_ids, img_dem);

  //// we need placeholders for threading support
  cv::namedWindow("image_DEM");
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
    int b = img_obstacles.at<cv::Vec3b>(y,x)[0] - 100;
    int g = img_obstacles.at<cv::Vec3b>(y,x)[1] + 50;
    int r = img_obstacles.at<cv::Vec3b>(y,x)[2] - 100;
    img_obstacles.at<cv::Vec3b>(y,x)[0] = std::max(0, b);
    img_obstacles.at<cv::Vec3b>(y,x)[1] = std::min(255, g);
    img_obstacles.at<cv::Vec3b>(y,x)[2] = std::max(0, r);
  }
  for(int i = 0; i < super_obstacles.size(); i++) {
    int x = std::get<0>(pc2img[super_obstacles[i]]);
    int y = std::get<1>(pc2img[super_obstacles[i]]);
    //img_obstacles.at<uint8_t>(y,x) = 0.0;
    int b = img_obstacles.at<cv::Vec3b>(y,x)[0] - 100;
    int g = img_obstacles.at<cv::Vec3b>(y,x)[1] - 100;
    int r = img_obstacles.at<cv::Vec3b>(y,x)[2] + 50;
    img_obstacles.at<cv::Vec3b>(y,x)[0] = std::max(0, b);
    img_obstacles.at<cv::Vec3b>(y,x)[1] = std::max(0, g);
    img_obstacles.at<cv::Vec3b>(y,x)[2] = std::min(255, r);

    b = img_dem.at<cv::Vec3b>(y,x)[0] - 100;
    g = img_dem.at<cv::Vec3b>(y,x)[1] - 100;
    r = img_dem.at<cv::Vec3b>(y,x)[2] + 50;
    img_dem.at<cv::Vec3b>(y,x)[0] = std::max(0, b);
    img_dem.at<cv::Vec3b>(y,x)[1] = std::max(0, g);
    img_dem.at<cv::Vec3b>(y,x)[2] = std::min(255, r);
  }
  cv::imshow("image_DEM", img_dem);
  //cv::waitKey(0);
  cv::imshow("free_space_image", img_obstacles);
  std::stringstream strnum;
  //strnum << std::setw(5) << std::setfill('0') << frame_num;
  strnum << std::setw(5) << std::setfill('0') << img_cnt++;
  cv::resize(img_dem, img_dem, cv::Size(1240, 376));
  cv::imwrite("out/dem_" + strnum.str() + ".jpg", img_dem);
  //cv::imwrite("out/ransac_patch_" + strnum.str() + ".png", img_patch);
  //cv::imwrite("out/obstacles_" + strnum.str() + ".png", img_obstacles);

  cv::Mat cvH;
  Eigen::Matrix3d H;
  //cv::Size warp_sz(750, 850);
  cv::Size warp_sz(750, 750);
  //cv::Size warp_sz(700, 1100);
  Eigen::Vector3d Q1, Q2;
  recon::MathHelper::getTopViewHomography(cam_params, img_left.cols, img_left.rows, warp_sz.width, warp_sz.height,
                                          plane_model, H);
  cv::eigen2cv(H, cvH);
  cv::Mat img_warp;
  //cv::warpPerspective(img_left, img_warp, cvH, img_left.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
  //cv::warpPerspective(img_left, img_warp, cvH, img_left.size());
  //cv::warpPerspective(img_left, img_warp, cvH, warp_sz);
  cv::warpPerspective(img_dem, img_warp, cvH, warp_sz);
  cv::imshow("img_warp", img_warp);
  cv::imwrite("out/top_view_" + strnum.str() + ".jpg", img_warp);
  cv::waitKey(10);
  //cv::waitKey(0);

  //pcl::visualization::PCLVisualizer viewer("3D Viewer");
  //viewer.setBackgroundColor(0, 0, 0);
  //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> hcolor(point_cloud_rgb);
  //viewer.addPointCloud<pcl::PointXYZRGB>(point_cloud_rgb, hcolor, "reconstruction");
  //pcl::PointCloud<pcl::PointXYZ>::Ptr poly_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  //pcl::PointCloud<pcl::PointXYZ>::Ptr plane2(new pcl::PointCloud<pcl::PointXYZ>);
  //recon::MathHelper::createPolyFromPlane(plane_model, poly_cloud);
  //pcl::PointXYZ pt1(0.0, -15.0, 0.0);
  //pcl::PointXYZ pt2(0.0, -15.0, 100.0);
  //pcl::PointXYZ pt3(0.0, 5.0, 100.0);
  //pcl::PointXYZ pt4(0.0, 5.0, 0.0);
  //plane2->points.push_back(pt1);
  //plane2->points.push_back(pt2);
  //plane2->points.push_back(pt3);
  //plane2->points.push_back(pt4);
  //viewer.addPolygon<pcl::PointXYZ>(poly_cloud, 0, 0, 200, "poly1");
  ////viewer.addPolygon<pcl::PointXYZ>(plane2, 200, 0, 0, "poly2");
  ////viewer.addPolygon<pcl::PointXYZ>(poly_cloud, 100, 100, 100, "poly1");
  //recon::MathHelper::debugTopViewHomography(cam_params, img_left.cols, img_left.rows, warp_sz.width, warp_sz.height,
  //                                          plane_model, viewer);
  //viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
  //                                   pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, "poly1");
  ////viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
  ////                                   pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, "poly2");
  //viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
  //                                   pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, "line1");
  //viewer.addCoordinateSystem(1.0);
  //viewer.initCameraParameters();
  ////Main loop
  //while (!viewer.wasStopped()) {
  //  //viewer.spinOnce(100);        // 100
  //  viewer.spin();        // 100
  //  //boost::this_thread::sleep(boost::posix_time::microseconds(10000));
  //}
  //viewer.close();

  frame_num++;
  //while(true) {
  //  int key = cv::waitKey(0);
  //  //std::cout << key << "\n";
  //  if(key == 27 || key == 1048603) {
  //    done_looping = true;
  //    break;
  //  }
  //  else if(key == 65362) {
  //    frame_num++;
  //    break;
  //  }
  //  else if(key == 65364) {
  //    frame_num--;
  //    break;
  //  }
  //}

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
