#include <vector>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;


#include <boost/program_options.hpp>
using namespace boost;
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <vtkPlaneSource.h>
#include <vtkImplicitPlaneWidget.h>

#include "../../../core/format_helper.h"
#include "../../../core/math_helper.h"


vtkSmartPointer<vtkPolyData> createPlane(const pcl::ModelCoefficients& coefficients)
{
  vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New ();
  //plane->SetCenter(0,0,0);
  //plane->SetPoint1(10, 0, 0);
  //plane->SetPoint2(0, 0, 10);

  plane->SetNormal (coefficients.values[0], coefficients.values[1], coefficients.values[2]);
  double norm_sqr = coefficients.values[0] * coefficients.values[0]
                  + coefficients.values[1] * coefficients.values[1]
                  + coefficients.values[2] * coefficients.values[2];


  plane->Push(-coefficients.values[3] / sqrt(norm_sqr));
  plane->SetResolution(200, 200);
  plane->Update();

  double pt1[3], pt2[3], orig[3];
  plane->GetPoint1(pt1);
  plane->GetPoint2(pt2);
  plane->GetOrigin(orig);
  // TODO: buged
  double scale = 10.0;
  for(int i = 0; i < 3; i++) {
    pt1[i] = scale * (pt1[i] - orig[i]);
    pt2[i] = scale * (pt2[i] - orig[i]);
  }
  plane->SetPoint1(pt1);
  plane->SetPoint2(pt2);
  plane->Update();
  return (plane->GetOutput());

  //vtkSmartPointer<vtkImplicitPlaneWidget> inf_plane = vtkSmartPointer<vtkImplicitPlaneWidget>::New ();
  //inf_plane->SetNormal(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
  //inf_plane->SetOrigin(0, 0, 0);
  //vtkSmartPointer<vtkPolyData> pplane = vtkSmartPointer<vtkPolyData>::New();
  //inf_plane->SetEnabled(1);
  //inf_plane->GetPolyData(pplane);
  //return (pplane);
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

  pt3d.at<double>(0) = (x - cx) * b / disp;
  pt3d.at<double>(1) = (y - cy) * b / disp;
  pt3d.at<double>(2) = f * b / disp;
  pt3d.at<double>(3) = 1.0;
}

void renderPointCloud(const vector<string>& imagelist, const vector<string>& disp_imglist,
                      const string& gt_file, const string& cparams_file,
                      const string& source_folder, const string& disp_folder, const string& output_folder)
{
  double cam_params[5];
  core::FormatHelper::readCameraParams(cparams_file, cam_params);

  //pcl::PointCloud<pcl::PointXYZI> point_cloud;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

  ifstream gt_fp(gt_file);
  Mat Rt = Mat::eye(4, 4, CV_64F);
  Mat pt3d = Mat::zeros(4, 1, CV_64F);
  Mat img_left;
  Mat img_disp;
  //for(size_t i = 0; i < disp_imglist.size(); i++) {
  //for(size_t i = 0; i < 30; i++) {
  double min_disp = 40.0;
  int img_step = 1;
  //size_t start_frame = 27;
  //size_t end_frame = 28;
  size_t start_frame = 100;
  size_t end_frame = 120;
  for(size_t i = 0; i < end_frame; i++) {
    core::FormatHelper::readNextMatrixKitti(gt_fp, Rt);
    if(!(i % img_step == 0) || i < start_frame) continue;
    cout << source_folder + imagelist[2*i] << "\n";
    img_left = imread(source_folder + imagelist[2*i], CV_LOAD_IMAGE_GRAYSCALE);

    //string fname = output_folder + imagelist[i].substr(9, imagelist[i].size()-13) + "_disp.png";
    string fname = output_folder + disp_imglist[i];
    img_disp = imread(disp_folder + disp_imglist[i], CV_LOAD_IMAGE_GRAYSCALE);
    for(int y = 0; y < img_disp.rows; y++) {
      for(int x = 0; x < img_disp.cols; x++) {
        double disp = img_disp.at<uint8_t>(y,x);
        if(disp > min_disp) {
          triangulate(cam_params, x, y, disp, pt3d);
          //cout << Rt << "\n";
          //cout << pt3d << "\n";
          //cout << Rt * pt3d << "\n";
          pt3d = Rt * pt3d;
          //cout << pt3d << "\n\n";
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
        }
      }
    }
    //imshow("disparity", img_disp);
    //waitKey(0);
  }
  cout << "Point cloud size: " << point_cloud->points.size() << "\n";
  point_cloud->width = point_cloud->points.size();
  point_cloud->height = 1;


  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100000);
  seg.setDistanceThreshold(0.1);

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

  //  std::cout << "Filtered size: " << pc_filtered->points.size() << "\n";
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

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  seg.setInputCloud(point_cloud->makeShared());
  seg.segment(*inliers, *coefficients);
  // paint the inlier points
  //for(size_t j = 0; j < inliers->indices.size(); j++) {
  //  //cout << j << "\n";
  //  int idx = inliers->indices[j];
  //  point_cloud->points[idx].r = 255;
  //  point_cloud->points[idx].g = 0;
  //  point_cloud->points[idx].b = 0;
  //}

  pcl::visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setBackgroundColor(0, 0, 0);
  //pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud_ptr(&point_cloud);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> hcolor(point_cloud);
  viewer.addPointCloud<pcl::PointXYZRGB>(point_cloud, hcolor, "reconstruction");

  //vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
  //vtkSmartPointer<vtkPolyData> plane = createPlane(*coefficients);
  //viewer.addModelFromPolyData(plane, "plane");
  //viewer.addPlane(*coefficients, "def_plane");

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
  string config_file;
  string imagelistfn, disp_imagelistfn;
  string cam_params_file;
  string source_folder;
  string disp_folder, gt_file;
  string output_folder;

  try {
    po::options_description generic("Generic options");
    generic.add_options()
      ("help", "produce help message")
      ("config,c", po::value<string>(&config_file)->default_value("config.txt"), "config filename")
      ;
    po::options_description config("Config file options");
    config.add_options()
      ("camera_params,p", po::value<string>(&cam_params_file)->default_value("camera_params.txt"),
       "camera params file")
      ("source_folder,s", po::value<string>(&source_folder), "folder with source")
      ("output_folder,o", po::value<string>(&output_folder), "folder for output")
      ("disp_folder,d", po::value<string>(&disp_folder), "folder for output")
      ("groundtruth,g", po::value<string>(&gt_file), "groundtruth file")
      ("imglist,l", po::value<string>(&imagelistfn), "file with image list")
      ("disp_imglist,dl", po::value<string>(&disp_imagelistfn), "file with image list")
      ;

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config);

    po::options_description config_file_options;
    config_file_options.add(config);
    po::variables_map vm;
    //po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    notify(vm);
    if(vm.count("help")) {
      cout << generic;
      cout << config;
      return 0;
    }

    ifstream ifs(config_file.c_str());
    if (!ifs) {
      cout << "can not open config file: " << config_file << "\n";
      cout << generic;
      cout << config;
      return 0;
    }
    else {
      po::store(parse_config_file(ifs, config_file_options), vm);
      notify(vm);
    }
    cout << "Configuring done, using:" << endl;

    if(vm.count("camera_params")) {
      cout << "Camera params: ";
      cout << cam_params_file << endl;
    }
    if(vm.count("source_folder")) {
      cout << "Source folder: ";
      cout << source_folder << endl;
    }
    if(vm.count("output_folder")) {
      cout << "Output folder: ";
      cout << output_folder << endl;
    }
    if(vm.count("imglist")) {
      cout << "Image list file: ";
      cout << imagelistfn << endl;
    }
  }
  catch(std::exception& e) {
    cout << e.what() << "\n";
    return 1;
  }

  if(imagelistfn == "")
  {
    cout << "error: no xml image list given." << endl;
    return -1;
  }
  if(source_folder == "")
  {
    cout << "error: no output or source folder given." << endl;
    return -1;
  }

  vector<string> imagelist, disp_imagelist;
  bool ok = core::FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
  {
    cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
    return -1;
  }
  ok = core::FormatHelper::readStringList(disp_imagelistfn, disp_imagelist);

  renderPointCloud(imagelist, disp_imagelist, gt_file, cam_params_file, source_folder, disp_folder, output_folder);

  return 0;
}
