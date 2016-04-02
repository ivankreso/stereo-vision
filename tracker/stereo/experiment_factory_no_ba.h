#ifndef STEREO_ODOMETRY_EXPERIMENT_FACTORY_NO_BA_
#define STEREO_ODOMETRY_EXPERIMENT_FACTORY_NO_BA_

#include <string>
#include <fstream>
#include <boost/program_options.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/features2d/features2d.hpp>

#include "stereo_tracker_base.h"
#include "../../stereo_odometry/base/visual_odometry_ransac.h"
#include "../mono/tracker_base.h"
#include "../mono/tracker_bfm_cv.h"
#include "../mono/tracker_bfm.h"
#include "../mono/tracker_stm.h"
#include "../mono/tracker_bfm_1_to_n.h"
#include "../refiner/feature_refiner_klt.h"
#include "../stereo/stereo_tracker_base.h"
#include "../stereo/stereo_tracker_sim.h"
#include "../stereo/stereo_tracker_bfm.h"
#include "../stereo/stereo_tracker.h"
#include "../stereo/stereo_tracker_artificial.h"
#include "../stereo/stereo_tracker_refiner.h"
#include "../detector/feature_detector_harris_cv.h"
#include "../detector/feature_detector_harris_freak.h"
#include "../detector/feature_detector_uniform.h"

namespace track
{

namespace ExperimentFactory
{

static void create_experiment(const std::string& config_file,
                              const double* cam_params,
                              std::string& output_folder,
                              FeatureDetectorBase** feature_detector,
                              TrackerBase** mono_tracker,
                              StereoTrackerBase** stereo_tracker,
                              visodom::VisualOdometryBase** visodom);

}

static void ExperimentFactory::create_experiment(const std::string& config_file,
                                                 const double* cam_params,
                                                 std::string& output_folder,
                                                 FeatureDetectorBase** feature_detector,
                                                 TrackerBase** mono_tracker,
                                                 StereoTrackerBase** stereo_tracker,
                                                 visodom::VisualOdometryBase** visodom)
{
  namespace po = boost::program_options;

  assert(*feature_detector == nullptr);
  assert(*mono_tracker == nullptr);
  assert(*stereo_tracker == nullptr);
  assert(*visodom == nullptr);

  std::string odometry_method_name;
  int ransac_iters;
  bool libviso_weighting = true;

  std::string stereo_tracker_name;
  std::string refiner_tracker_name;
  int max_disparity;
  int stereo_wsz;
  double ncc_threshold_stereo;
  bool estimate_subpixel = true;

  std::string mono_tracker_name;
  std::string stm_tracker_name;
  int max_features;
  double ncc_threshold_mono;
  int hamming_threshold;
  int ncc_patch_size;
  int search_wsz;
  double stm_q;
  double stm_a;
  bool match_with_oldest = true;

  std::string detector_name;
  int harris_block_sz;
  int harris_filter_sz;
  double harris_k;
  double harris_thr;
  int harris_margin;
  bool freak_norm_scale;
  bool freak_norm_orient;
  double freak_pattern_scale;
  int freak_num_octaves;

  size_t start_pos = config_file.find_last_of('/');
  output_folder = "./results" + config_file.substr(start_pos, config_file.size()-start_pos-4) + "/";

  std::cout << "Using experiment config: " << config_file << '\n';
  std::cout << "Using output folder: " << output_folder << '\n';

  try {
    po::options_description config("Config file");
    config.add_options()
      //("output_folder", po::value<std::string>(&output_folder))
      ("odometry_method", po::value<std::string>(&odometry_method_name))
      ("ransac_iters", po::value<int>(&ransac_iters))
      ("libviso_weighting", po::value<bool>(&libviso_weighting))

      ("tracker", po::value<std::string>(&stereo_tracker_name))
      ("refiner_tracker", po::value<std::string>(&refiner_tracker_name))
      ("max_disparity", po::value<int>(&max_disparity))
      ("stereo_wsz", po::value<int>(&stereo_wsz))
      ("ncc_threshold_s", po::value<double>(&ncc_threshold_stereo))
      ("estimate_subpixel", po::value<bool>(&estimate_subpixel))

      ("tracker_mono", po::value<std::string>(&mono_tracker_name))
      ("tracker_stm", po::value<std::string>(&stm_tracker_name))
      ("max_features", po::value<int>(&max_features))
      ("ncc_threshold_m", po::value<double>(&ncc_threshold_mono))
      ("hamming_threshold", po::value<int>(&hamming_threshold))
      ("ncc_patch_size", po::value<int>(&ncc_patch_size))
      ("search_wsz", po::value<int>(&search_wsz))
      ("stm_q", po::value<double>(&stm_q))
      ("stm_a", po::value<double>(&stm_a))
      ("match_with_oldest", po::value<bool>(&match_with_oldest))

      ("detector", po::value<std::string>(&detector_name))
      ("harris_block_sz", po::value<int>(&harris_block_sz))
      ("harris_filter_sz", po::value<int>(&harris_filter_sz))
      ("harris_k", po::value<double>(&harris_k))
      ("harris_thr", po::value<double>(&harris_thr))
      ("harris_margin", po::value<int>(&harris_margin))
      ("freak_norm_scale", po::value<bool>(&freak_norm_scale))
      ("freak_norm_orient", po::value<bool>(&freak_norm_orient))
      ("freak_pattern_scale", po::value<double>(&freak_pattern_scale))
      ("freak_num_octaves", po::value<int>(&freak_num_octaves))
      ;

    po::options_description config_file_options;
    config_file_options.add(config);
    po::variables_map vm;
    //po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    //po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    //notify(vm);
    //if(vm.count("help")) {
    //  cout << generic;
    //  cout << config;
    //  return 0;
    //}
    std::ifstream ifs(config_file);
    if(!ifs) {
      std::cout << "can not open config file: " << config_file << "\n";
      throw -1;
    }
    else {
      po::store(parse_config_file(ifs, config_file_options, true), vm);
      notify(vm);
    }
    std::cout << "Configuring done!\n";
  }
  catch(std::exception& e) {
    std::cout << e.what() << '\n';
    throw e.what();
  }

  visodom::VisualOdometryRansac::parameters param;
  // libviso defaults
  param.ransac_iters     = 200;
  param.inlier_threshold = 1.5;
  // my camera
  param.calib.f = cam_params[0];  // focal length in pixels
  param.calib.cu = cam_params[2]; // principal point (u-coordinate) in pixels
  param.calib.cv = cam_params[3]; // principal point (v-coordinate) in pixels
  param.base = cam_params[4];
  param.ransac_iters = ransac_iters;           // def: 100
  param.reweighting = libviso_weighting;
  if(odometry_method_name == "VisualOdometryRansac") {
    //*visodom = new libviso::VisualOdometryStereo(param);
    *visodom = new visodom::VisualOdometryRansac(param);
  }
  else throw 1;

  std::cout << "Using detector: " << detector_name << '\n' << "Using mono tracker: "
            << mono_tracker_name << '\n';
  // create detector
  if(detector_name == "FeatureDetectorHarrisCV") {
    *feature_detector = new track::FeatureDetectorHarrisCV(harris_block_sz, harris_filter_sz,
                                                           harris_k, harris_thr, harris_margin);
  }
  else if(detector_name == "FeatureDetectorHarrisFREAK") {
    FeatureDetectorHarrisCV* detector_base = new track::FeatureDetectorHarrisCV(harris_block_sz,
        harris_filter_sz, harris_k, harris_thr, harris_margin);
    cv::FREAK* extractor = new cv::FREAK(freak_norm_orient, freak_norm_scale, freak_pattern_scale,
                                         freak_num_octaves);
    *feature_detector = new track::FeatureDetectorHarrisFREAK(detector_base, extractor);
  }
  else if(detector_name == "FeatureDetectorUniform") {
    //if(detector_base_name == "FeatureDetectorHarrisFREAK") {
    //  FeatureDetectorHarrisCV* detector_base1 = new track::FeatureDetectorHarrisCV(harris_block_sz,
    //      harris_filter_sz, harris_k, harris_thr, harris_margin);
    //  cv::FREAK* extractor = new cv::FREAK(freak_norm_orient, freak_norm_scale, freak_pattern_scale,
    //                                       freak_num_octaves);
    //  FeatureDetectorBase* detector_base = new track::FeatureDetectorHarrisFREAK(detector_base1, extractor);
    //}
    FeatureDetectorBase* detector_base = new track::FeatureDetectorHarrisCV(harris_block_sz, harris_filter_sz,
                                                           harris_k, harris_thr, harris_margin);
    *feature_detector = new track::FeatureDetectorUniform(detector_base, 15, 5, 15);
  }
  else
    std::cout << "[ExperimentFactory]: No detector...\n";

  // create mono tracker
  if(mono_tracker_name == "TrackerBFM")
    *mono_tracker = new track::TrackerBFM(**feature_detector, max_features, ncc_threshold_mono,
                                          ncc_patch_size, search_wsz, match_with_oldest);
  else if(mono_tracker_name == "TrackerBFMcv")
    *mono_tracker = new track::TrackerBFMcv(**feature_detector, max_features, search_wsz,
                                            hamming_threshold);
  else if(mono_tracker_name == "TrackerSTM") {
    TrackerBase* stm_tracker;
    if(stm_tracker_name == "TrackerBFM")
      stm_tracker = new track::TrackerBFM(**feature_detector, max_features, ncc_threshold_mono,
                                         ncc_patch_size, search_wsz);
    else throw "Error";
    *mono_tracker = new track::TrackerSTM(stm_tracker, stm_q, stm_a);
  }
  else if(mono_tracker_name == "TrackerBFM1toN") {
    *mono_tracker = new track::TrackerBFM1toN(**feature_detector, max_features, ncc_threshold_mono,
                                              ncc_patch_size, search_wsz, match_with_oldest);
  }
  else
    std::cout << "[ExperimentFactory]: No mono tracker...\n";
  
  // create stereo tracker
  std::cout << "Using stereo tracker: " << stereo_tracker_name << '\n';
  if(stereo_tracker_name == "StereoTracker") {
    *stereo_tracker = new track::StereoTracker(**mono_tracker, max_disparity, stereo_wsz, 
                                               ncc_threshold_stereo, estimate_subpixel);
  }
  else if(stereo_tracker_name == "StereoTrackerRefiner") {
    StereoTrackerBase* refiner_tracker;
    if(refiner_tracker_name == "StereoTracker")
      refiner_tracker = new track::StereoTracker(**mono_tracker, max_disparity, stereo_wsz, 
                                                  ncc_threshold_stereo, estimate_subpixel);
    else throw "Error";
    *stereo_tracker = new track::StereoTrackerRefiner(refiner_tracker,
                                                      new track::refiner::FeatureRefinerKLT, true);
  } 
  else if(stereo_tracker_name == "StereoTrackerBFM") {
    *stereo_tracker = new track::StereoTrackerBFM(*feature_detector, max_features, ncc_threshold_stereo, 
                                                  ncc_patch_size, search_wsz);
  }
  else if(stereo_tracker_name == "StereoTrackerLibviso") {
    throw "Error";
  }
  else throw "Error";


  // TODO: add resume folder here and output name from filename
  struct stat st = {0};
  if(stat(output_folder.c_str(), &st) == -1)
    mkdir(output_folder.c_str(), 0700);
}


}


#endif
