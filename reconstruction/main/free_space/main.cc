#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <tuple>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/filters/passthrough.h>
#include <vtkPlaneSource.h>
#include <vtkImplicitPlaneWidget.h>


#include "../../../core/format_helper.h"
#include "../../../core/math_helper.h"


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

  //for(size_t frame = 0; frame < imagelist.size()/2; frame++) {
  //size_t start_frame = frame;
  //pcl::PointCloud<pcl::PointXYZI> point_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

  std::vector<std::tuple<int,int>> points_img;

  ifstream gt_fp(gt_file);
  cv::Mat Rt = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_64F);
  cv::Mat img_left;
  cv::Mat mat_disp;
  //double min_disp = 5.0;
  double min_disp = 40.0;
  int img_step = 1;
  //size_t start_frame = 100;
  //size_t start_frame = 315;
  //size_t start_frame = 27;
  // meadow on right
  size_t start_frame = 349;
  // car in front
  //size_t start_frame = 995;
  // traffic sign
  //size_t start_frame = 442;
  //size_t start_frame = 620;
  //for(size_t i = 0; i < end_frame; i++) {
  //for(size_t i = start_frame; i < end_frame; i++) {
  for(size_t i = start_frame; i < start_frame+1; i++) {
    //core::FormatHelper::readNextMatrixKitti(gt_fp, Rt);
    if(!(i % img_step == 0) || i < start_frame) continue;
    std::cout << "----------------\n" << source_folder + imagelist[2*i] << "\n";
    img_left = cv::imread(source_folder + imagelist[2*i], CV_LOAD_IMAGE_GRAYSCALE);
    //string fname = output_folder + imagelist[i].substr(9, imagelist[i].size()-13) + "_disp.png";
    //std::string fname = output_folder + disp_imglist[i];
    //img_disp = imread(disp_folder + disp_imglist[i], CV_LOAD_IMAGE_GRAYSCALE);
    std::string mat_fname = disp_folder + "mat/" + disp_imglist[i].substr(0,11) + ".yml";
    std::cout << "reading: " << mat_fname << "\n";
    cv::FileStorage mat_file(mat_fname, cv::FileStorage::READ);
    mat_file["disparity_matrix"] >> mat_disp;
    for(int y = 0; y < mat_disp.rows; y++) {
      for(int x = 0; x < mat_disp.cols; x++) {
        float disp = mat_disp.at<float>(y,x);
        if(disp > min_disp) {
          triangulate(cam_params, x, y, disp, pt3d);
          //std::cout << Rt << "\n";
          //std::cout << pt3d << "\n";
          //std::cout << Rt * pt3d << "\n";
          pt3d = Rt * pt3d;
          //std::cout << pt3d << "\n\n";
          pcl::PointXYZRGB point;
          //pcl::PointXYZI point;
          point.x = pt3d.at<double>(0);
          point.y = pt3d.at<double>(1);
          point.z = pt3d.at<double>(2);
          //point.intensity = img_left.at<uint8_t>(x,y);
          point.r = img_left.at<uint8_t>(y,x);
          point.g = img_left.at<uint8_t>(y,x);
          point.b = img_left.at<uint8_t>(y,x);
          point_cloud->points.push_back(point);
          points_img.push_back(std::make_tuple(y,x));
        }
      }
    }
    //imshow("disparity", img_disp);
    //waitKey(0);
  }
  std::cout << "Point cloud size: " << point_cloud->points.size() << "\n";
  point_cloud->width = point_cloud->points.size();
  point_cloud->height = 1;

  // Create the filtering object
  //pcl::PassThrough<pcl::PointXYZRGB> pass;
  //pass.setInputCloud(point_cloud);
  //pass.setFilterFieldName("y");
  //pass.setFilterLimits(-10.0, 1.0);
  ////pass.setFilterLimitsNegative(true);
  //pass.filter(*point_cloud);
  //pass.setFilterFieldName("x");
  //pass.setFilterLimits(-10.0, 10.0);
  //pass.filter(*point_cloud);

  // Create the segmentation object
  //pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  //// Optional
  //seg.setOptimizeCoefficients(true);
  //// Mandatory
  ////seg.setModelType(pcl::SACMODEL_PLANE);
  //seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);

  //seg.setMethodType(pcl::SAC_RANSAC);
  ////seg.setMaxIterations(100000);
  //seg.setMaxIterations(1000000);
  ////seg.setDistanceThreshold(0.15);
  //seg.setDistanceThreshold(0.01);
  ////seg.setDistanceThreshold(0.2);
  ////seg.setDistanceThreshold(0.01);
  //pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  //pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  //seg.setInputCloud(point_cloud->makeShared());
  //seg.segment(*inliers, *coefficients);
  //// RANSAC fitting
  ////paint the inlier points
  //for(size_t j = 0; j < inliers->indices.size(); j++) {
  //  //std::cout << j << "\n";
  //  int idx = inliers->indices[j];
  //  point_cloud->points[idx].r = 255;
  //  point_cloud->points[idx].g = 0;
  //  point_cloud->points[idx].b = 0;
  //  int r, c;
  //  std::tie(r,c) = points_img[idx];
  //  img_left.at<uint8_t>(r,c) = 255;
  //}
  
  std::vector<int> inliers;
  // created RandomSampleConsensus object and compute the appropriated model
  //pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>(point_cloud));
  //pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_p);
  //ransac.setDistanceThreshold(0.1);
  //ransac.computeModel();
  //ransac.getInliers(inliers);

  pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZRGB>::Ptr model_p(
        new pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZRGB>(point_cloud));
  //pcl::SampleConsensusModelPerpendicularPlane<pcl::PointXYZRGB> model_p(point_cloud);
  model_p->setAxis(Eigen::Vector3f(0.0, 1.0, 0.0));
  //model_p->setEpsAngle(pcl::deg2rad(15.0));
  //model_p->setEpsAngle(pcl::deg2rad(5.0));
  model_p->setEpsAngle(pcl::deg2rad(3.0));

  //pcl::SampleConsensusModelNormalPlane<pcl::PointXYZRGB, pcl::Normal>::Ptr model_p(
  //      new pcl::SampleConsensusModelNormalPlane<pcl::PointXYZRGB, pcl::Normal>(point_cloud));
  //model_p->setNormalDistanceWeight(0.1);
  //// Estimate surface normals
  //pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
  //pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
  //n.setInputCloud(point_cloud);
  ////std::vector<int> indices;
  ////n.setIndices(boost::make_shared<std::vector<int>>(indices));
  //pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  //n.setSearchMethod(tree);
  ////n.setRadiusSearch(0.02);    // Use 2cm radius to estimate the normals
  //n.setRadiusSearch(0.1);    // Use 2cm radius to estimate the normals
  //n.compute(*normals);
  //model_p->setInputNormals(normals);

  mesh it up
  pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_p);
  //ransac.setDistanceThreshold(0.1);
  ransac.setDistanceThreshold(0.07);
  ransac.computeModel();
  ransac.getInliers(inliers);

  // RANSAC fitting
  //paint the inlier points
  for(size_t j = 0; j < inliers.size(); j++) {
    //std::cout << j << "\n";
    int idx = inliers[j];
    point_cloud->points[idx].r = 255;
    point_cloud->points[idx].g = 0;
    point_cloud->points[idx].b = 0;
    int r, c;
    std::tie(r,c) = points_img[idx];
    img_left.at<uint8_t>(r,c) = 255;
  }
  //cv::imshow("left_img", img_left);
  std::cout << "saving: /home/kivan/Projects/outputs/KITTI/ransac/" << imagelist[2*start_frame] << "\n";
  cv::imwrite("/home/kivan/Projects/outputs/KITTI/ransac/" + imagelist[2*start_frame], img_left);
  //cv::waitKey(0);
  //} // end of big for loop

  //// Create the filtering object: downsample the dataset using a leaf size of 1cm
  //pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  //sor.setInputCloud(point_cloud);
  //sor.setLeafSize(0.02f, 0.02f, 0.02f);
  //sor.filter(*pc_filtered);
  //std::cerr << "PointCloud after filtering: " << pc_filtered->width * pc_filtered->height << " data points." << std::endl;
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_planar(new pcl::PointCloud<pcl::PointXYZRGB>);
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
  // Create the filtering object
  //pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  //int i = 0, nr_points = (int)pc_filtered->points.size ();
  //// While 30% of the original cloud is still there
  //while(pc_filtered->points.size () >= 1.0 * nr_points)
  //{
  //  // Segment the largest planar component from the remaining cloud
  //  seg.setInputCloud(pc_filtered);
  //  seg.segment(*inliers, *coefficients);
  //  if(inliers->indices.size() == 0)
  //  {
  //    std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
  //    break;
  //  }

  //  std::std::cout << "Filtered size: " << pc_filtered->points.size() << "\n";
  //  // Extract the inliers
  //  extract.setInputCloud(pc_filtered);
  //  extract.setIndices(inliers);
  //  extract.setNegative(false);
  //  extract.filter(*cloud_planar);
  //  std::cerr << "PointCloud representing the planar component: " << cloud_planar->width * cloud_planar->height
  //            << " data points." << std::endl;

  //  // Create the filtering object
  //  extract.setNegative(true);
  //  extract.filter(*cloud_tmp);
  //  pc_filtered.swap(cloud_tmp);
  //  i++;
  //}



  pcl::visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setBackgroundColor(0, 0, 0);
  //pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr(&point_cloud);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> hcolor(point_cloud);
  viewer.addPointCloud<pcl::PointXYZRGB>(point_cloud, hcolor, "reconstruction");

  //pcl::PointCloud<pcl::PointXYZ>::Ptr poly_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  //createPolyFromPlane(coefficients, poly_cloud);
  //viewer.addPolygon<pcl::PointXYZ>(poly_cloud, 0, 0, 180, "poly1");
  ////viewer.addPolygon<pcl::PointXYZ>(poly_cloud, 100, 100, 100, "poly1");
  //viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
  //                                   pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, "poly1");

  //viewer.addPolygon<pcl::PointXYZ>(poly_cloud, 0.0, 255.0, 0.0, "poly1");

  //vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer.addCoordinateSystem(1.0);
  viewer.initCameraParameters();


  // TODO
  //plot the point distances from fitted plane
  //try ti fit another plane with inliers only

  //Main loop
  while (!viewer.wasStopped()) {
    //viewer.spinOnce(100);        // 100
    viewer.spin();        // 100
    //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    //boost::this_thread::sleep(boost::posix_time::microseconds(10000));
  }
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
