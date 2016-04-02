#ifndef STEREO_ODOMETRY_EXPERIMENT_FACTORY_
#define STEREO_ODOMETRY_EXPERIMENT_FACTORY_

#include <string>
#include <fstream>
#include <boost/program_options.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "stereo_tracker_base.h"
#include "../../stereo_egomotion/base/egomotion_base.h"
#include "../../stereo_egomotion/base/egomotion_ransac.h"
#include "../../stereo_egomotion/base/egomotion_libviso.h"
#include "../mono/tracker_base.h"
#include "../mono/tracker_bfm_cv.h"
#include "../mono/tracker_bfm.h"
#include "../mono/tracker_stm.h"
//#include "../mono/tracker_bfm_1_to_n.h"
#include "../refiner/feature_refiner_klt.h"
#include "../stereo/stereo_tracker_base.h"
#include "../stereo/stereo_tracker_sim.h"
//#include "../stereo/stereo_tracker_libviso.h"
#include "../stereo/stereo_tracker_bfm.h"
#include "../stereo/stereo_tracker.h"
#include "../stereo/stereo_tracker_orb.h"
#include "../stereo/stereo_tracker_freak.h"
#include "../stereo/stereo_tracker_artificial.h"
#include "../stereo/stereo_tracker_refiner.h"
//#include "../../../tracker/stereo/tracker_refiner_libviso.h"
#include "../detector/feature_detector_harris_cv.h"
//#include "../detector/feature_detector_harris_freak.h"
#include "../detector/feature_detector_uniform.h"
#include "../detector/feature_detector_agast.h"
//#include "../../stereo_egomotion/extern/libviso2/src/viso_stereo.h"
#include "../../optimization/bundle_adjustment/bundle_adjuster.h"
//#include "../../optimization/bundle_adjustment/bundle_adjuster_multiframe.h"
//#include "../../optimization/bundle_adjustment/bundle_adjuster_2frame.h"
//#include "../../optimization/bundle_adjustment/bundle_adjuster_mfi.h"

namespace track
{

namespace ExperimentFactory
{

bool create_experiment(const std::string config_file,
                       const std::string deformation_field_path,
                       const double* cam_params,
                       const int img_rows, const int img_cols,
                       std::string& output_folder,
                       FeatureDetectorBase** feature_detector,
                       TrackerBase** mono_tracker,
                       StereoTrackerBase** stereo_tracker,
                       egomotion::EgomotionBase** egomotion,
                       bool& use_bundle_adjustment,
                       optim::BundleAdjusterBase** ba_optim);

bool CreateValidationORB(int patch_size, int num_levels,
                         double scale_factor, int max_dist_stereo, int max_dist_mono,
                         std::shared_ptr<track::StereoTrackerBase>* stereo_tracker);

bool CreateValidationHarrisFreak(
    const int block_sz,
    const int filter_sz,
    const double k,
    const double thr,
    const int margin,
    const int max_features,
    const int max_tracks,
    const bool freak_normalize_orientation,
    const double freak_pattern_scale,
    const int max_xdiff,
    const int max_dist_stereo,
    const int max_dist_mono,
    std::shared_ptr<track::StereoTrackerBase>* stereo_tracker);

bool CreateValidationAgastFreak(
    const int agast_thr,
    const std::string agast_type,
    const bool freak_normalize_orientation,
    const float freak_pattern_scale,
    const int max_xdiff,
    const int max_dist_stereo,
    const int max_dist_mono,
    std::shared_ptr<track::StereoTrackerBase>* stereo_tracker);
}

bool ExperimentFactory::create_experiment(const std::string config_file,
                                          const std::string deformation_field_path,
                                          const double* cam_params,
                                          const int img_rows, const int img_cols,
                                          std::string& output_folder,
                                          FeatureDetectorBase** feature_detector,
                                          TrackerBase** mono_tracker,
                                          StereoTrackerBase** stereo_tracker,
                                          egomotion::EgomotionBase** egomotion,
                                          bool& use_bundle_adjustment,
                                          optim::BundleAdjusterBase** ba_optim)
{
  namespace po = boost::program_options;

  assert(*feature_detector == nullptr);
  assert(*mono_tracker == nullptr);
  assert(*stereo_tracker == nullptr);
  assert(*egomotion == nullptr);

  std::string egomotion_method_name;
  int ransac_iters;
  double ransac_threshold;
  std::string loss_function_type;
  double robust_loss_scale;
  bool use_weighting = false;
  bool use_deformation_field = false;

  std::string stereo_tracker_name;
  int max_disparity;
  int max_xdiff = 0;
  int orb_patch_size = 0;
  int orb_num_levels = 0;
  float orb_scale_factor = 0;
  int orb_max_dist_mono = 0;
  int orb_max_dist_stereo = 0;
  int freak_max_dist_mono = 0;
  int freak_max_dist_stereo = 0;

  std::string refiner_tracker_name;
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
  int horizontal_bins;
  int vertical_bins;
  int features_per_bin;
  int harris_block_sz;
  int harris_filter_sz;
  double harris_k;
  double harris_thr;
  int harris_margin;
  bool freak_norm_scale;
  bool freak_norm_orient;
  double freak_pattern_scale;
  int freak_num_octaves;

  use_bundle_adjustment = false;
  std::string bundle_adjuster_name;
  int ba_num_frames;
  std::string ba_type;

  try {
    po::options_description config("Config file");
    config.add_options()
      //("output_folder", po::value<std::string>(&output_folder))
      ("egomotion_method", po::value<std::string>(&egomotion_method_name)->required())
      ("ransac_iters", po::value<int>(&ransac_iters)->required())
      ("ransac_threshold", po::value<double>(&ransac_threshold)->required())
      ("loss_function_type", po::value<std::string>(&loss_function_type))
      ("robust_loss_scale", po::value<double>(&robust_loss_scale))
      ("use_weighting", po::value<bool>(&use_weighting)->required())
      ("use_deformation_field", po::value<bool>(&use_deformation_field)->required())

      ("tracker", po::value<std::string>(&stereo_tracker_name)->required())
      ("max_features", po::value<int>(&max_features)->required())
      ("max_disparity", po::value<int>(&max_disparity)->required())

      ("max_xdiff", po::value<int>(&max_xdiff))
      ("orb_patch_size", po::value<int>(&orb_patch_size))
      ("orb_num_levels", po::value<int>(&orb_num_levels))
      ("orb_scale_factor", po::value<float>(&orb_scale_factor))
      ("orb_max_dist_stereo", po::value<int>(&orb_max_dist_stereo))
      ("orb_max_dist_mono", po::value<int>(&orb_max_dist_mono))

      ("refiner_tracker", po::value<std::string>(&refiner_tracker_name))
      ("stereo_wsz", po::value<int>(&stereo_wsz))
      ("ncc_threshold_s", po::value<double>(&ncc_threshold_stereo))
      ("estimate_subpixel", po::value<bool>(&estimate_subpixel)->default_value(false))

      ("tracker_mono", po::value<std::string>(&mono_tracker_name)->default_value(""))
      ("ncc_threshold_m", po::value<double>(&ncc_threshold_mono)->default_value(0))
      ("hamming_threshold", po::value<int>(&hamming_threshold)->default_value(0))
      ("ncc_patch_size", po::value<int>(&ncc_patch_size)->default_value(0))
      ("search_wsz", po::value<int>(&search_wsz)->default_value(0))
      ("tracker_stm", po::value<std::string>(&stm_tracker_name))
      ("stm_q", po::value<double>(&stm_q))
      ("stm_a", po::value<double>(&stm_a))
      ("match_with_oldest", po::value<bool>(&match_with_oldest))

      ("detector", po::value<std::string>(&detector_name))
      ("horizontal_bins", po::value<int>(&horizontal_bins))
      ("vertical_bins", po::value<int>(&vertical_bins))
      ("features_per_bin", po::value<int>(&features_per_bin))
      ("harris_block_sz", po::value<int>(&harris_block_sz))
      ("harris_filter_sz", po::value<int>(&harris_filter_sz))
      ("harris_k", po::value<double>(&harris_k))
      ("harris_thr", po::value<double>(&harris_thr))
      ("harris_margin", po::value<int>(&harris_margin))

      ("freak_norm_scale", po::value<bool>(&freak_norm_scale))
      ("freak_norm_orient", po::value<bool>(&freak_norm_orient))
      ("freak_pattern_scale", po::value<double>(&freak_pattern_scale))
      ("freak_num_octaves", po::value<int>(&freak_num_octaves))
      ("freak_max_dist_stereo", po::value<int>(&freak_max_dist_stereo))
      ("freak_max_dist_mono", po::value<int>(&freak_max_dist_mono))

      ("use_bundle_adjustment", po::value<bool>(&use_bundle_adjustment)->default_value(false))
      ("bundle_adjuster", po::value<std::string>(&bundle_adjuster_name))
      ("ba_num_frames", po::value<int>(&ba_num_frames))
      ("ba_type", po::value<std::string>(&ba_type))
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

  size_t start_pos = config_file.find_last_of('/');
  if (output_folder.empty())
    output_folder = "./results/" + config_file.substr(start_pos, config_file.size() - start_pos - 4)
                    + "/";

  std::cout << "Using experiment config: " << config_file << '\n';
  std::cout << "Using output folder: " << output_folder << '\n';

  if (egomotion_method_name == "EgomotionLibviso") {
    egomotion::EgomotionLibviso::parameters param;
    // libviso defaults
    //param.ransac_iters     = 200;
    //params.inlier_threshold = 1.5;
    param.inlier_threshold = ransac_threshold;
    // my camera
    param.calib.f = cam_params[0];  // focal length in pixels
    param.calib.cu = cam_params[2]; // principal point (u-coordinate) in pixels
    param.calib.cv = cam_params[3]; // principal point (v-coordinate) in pixels
    param.base = cam_params[4];
    param.ransac_iters = ransac_iters;           // def: 100
    param.reweighting = use_weighting;
    std::cout << "Feature weighting = " << use_weighting << "\n";
    std::cout << "Deformation field = " << use_deformation_field << "\n";

    *egomotion = new egomotion::EgomotionLibviso(param);
    //if (!use_deformation_field) {
    //  *egomotion = new egomotion::EgomotionLibviso(param);
    //}
    //else if (deformation_field_path.size() > 0 && use_deformation_field) {
    //  std::cout << "Using deformation field: " << deformation_field_path << '\n';
    //  *egomotion = new egomotion::EgomotionLibviso(param, deformation_field_path,
    //                                               img_rows, img_cols);
    //}
    //else {
    //  throw 1;
    //}
    // FIX for StereoTracker
    //use_deformation_field = false;
    //throw 1;
  }
  else if (egomotion_method_name == "EgomotionRansac") {
    egomotion::EgomotionRansac::Parameters params;
    params.ransac_iters = ransac_iters;
    params.inlier_threshold = ransac_threshold;
    params.loss_function_type = loss_function_type;
    params.robust_loss_scale = robust_loss_scale;
    params.use_weighting = use_weighting;
    params.calib.f = cam_params[0];
    params.calib.cx = cam_params[2];
    params.calib.cy = cam_params[3];
    params.calib.b = cam_params[4];
    *egomotion = new egomotion::EgomotionRansac(params);
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
    throw 1;
    //FeatureDetectorHarrisCV* detector_base = new track::FeatureDetectorHarrisCV(harris_block_sz,
    //    harris_filter_sz, harris_k, harris_thr, harris_margin);
    //cv::FREAK* extractor = new cv::FREAK(freak_norm_orient, freak_norm_scale, freak_pattern_scale,
    //                                     freak_num_octaves);
    //// TODO: add FeatureDetectorUniform as detector_base
    //*feature_detector = new track::FeatureDetectorHarrisFREAK(detector_base, extractor);
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
    *feature_detector = new track::FeatureDetectorUniform(detector_base, horizontal_bins,
                                                          vertical_bins, features_per_bin);
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
    throw 1;
    //*mono_tracker = new track::TrackerBFM1toN(**feature_detector, max_features, ncc_threshold_mono,
    //                                          ncc_patch_size, search_wsz, match_with_oldest);
  }
  
  // create stereo tracker
  std::cout << "Using stereo tracker: " << stereo_tracker_name << '\n';
  if(stereo_tracker_name == "StereoTrackerFREAK") {
    //*stereo_tracker = new track::StereoTrackerORB(max_features, 100, 1.0, 200, 40, 21, 1.1, 1, 30, 30);
    throw 1;
    //*stereo_tracker = new track::StereoTrackerFREAK(max_features, max_xdiff, 1.0, max_disparity,
    //                                                40, orb_patch_size, orb_scale_factor,
    //                                                orb_num_levels, orb_max_dist_stereo,
    //                                                orb_max_dist_mono);
  }
  else if(stereo_tracker_name == "StereoTrackerORB") {
    //*stereo_tracker = new track::StereoTrackerORB(max_features, 100, 1.0, 200, 40, 21, 1.1, 1, 30, 30);
    *stereo_tracker = new track::StereoTrackerORB(max_features, max_xdiff, 1.0, max_disparity,
                                                  40, orb_patch_size, orb_scale_factor,
                                                  orb_num_levels, orb_max_dist_stereo,
                                                  orb_max_dist_mono);
  }
  else if(stereo_tracker_name == "StereoTracker") {
    *stereo_tracker = new track::StereoTracker(**mono_tracker, max_disparity, stereo_wsz,
                                               ncc_threshold_stereo, estimate_subpixel,
                                               use_deformation_field, deformation_field_path);
  }
  else if(stereo_tracker_name == "StereoTrackerRefiner") {
    StereoTrackerBase* refiner_tracker;
    if(refiner_tracker_name == "StereoTracker")
      refiner_tracker = new track::StereoTracker(**mono_tracker, max_disparity, stereo_wsz,
                                                 ncc_threshold_stereo, estimate_subpixel,
                                                 use_deformation_field, deformation_field_path);
    else return false;
    *stereo_tracker = new track::StereoTrackerRefiner(refiner_tracker,
                                                      new track::refiner::FeatureRefinerKLT, true);
  } 
  else if(stereo_tracker_name == "StereoTrackerBFM") {
    *stereo_tracker = new track::StereoTrackerBFM(*feature_detector, max_features,
                                                  ncc_threshold_stereo, ncc_patch_size, search_wsz);
  }
  else if(stereo_tracker_name == "StereoTrackerLibviso") {
    throw 1;
    //*stereo_tracker = new track::StereoTrackerLibviso();
  }
  else {
    std::cout << "[ExperimentFactory] Unknown stereo tracker\n";
    return false;
  }

  struct stat st = {0};
  if(stat("./results/", &st) == -1)
    mkdir("./results/", 0700);
  if(stat(output_folder.c_str(), &st) == -1)
    mkdir(output_folder.c_str(), 0700);

  if(!use_bundle_adjustment)
    return true;

  std::string ba_output_folder = output_folder.substr(0, output_folder.size()-1) + "_ba/";
  //std::string ba2_output_folder = output_folder.substr(0, output_folder.size()-1) + "_ba_multiframe/";
  if(stat(ba_output_folder.c_str(), &st) == -1)
    mkdir(ba_output_folder.c_str(), 0700);
  //if(stat(ba2_output_folder.c_str(), &st) == -1)
  //  mkdir(ba2_output_folder.c_str(), 0700);

  //optim::SBAbase::BAType adjuster_type;
  //if(ba_type == "motion")
  //  adjuster_type = optim::SBAbase::kMotion;
  //else if(ba_type == "structure_and_motion")
  //  adjuster_type = optim::SBAbase::kStructureAndMotion;
  //else
  //  throw 1;
  if(bundle_adjuster_name == "BundleAdjuster") {
    std::vector<double> params = { robust_loss_scale };
    *ba_optim = new optim::BundleAdjuster(ba_num_frames, max_features, loss_function_type, params,
                                          use_weighting);
  }
  else {
    std::cout << "[ExperimentFactory] Unknown BA method\n";
    return false;
  }
  //else if(bundle_adjuster_name == "BundleAdjusterMultiframe")
  //  *ba_optim = new optim::BundleAdjusterMultiframe(ba_num_frames, max_features, adjuster_type, use_weighting);
  //else if(bundle_adjuster_name == "BundleAdjuster2frame")
  //  *ba_optim = new optim::BundleAdjuster2frame(ba_num_frames, adjuster_type, use_weighting);
  //else if(bundle_adjuster_name == "BundleAdjusterMFI")
  //  *ba_optim = new optim::BundleAdjusterMFI(adjuster_type, use_weighting);
  ////*ba_optim = new optim::FeatureHelperSBA(ba_num_frames, max_features, optim::SBAbase::kStructureAndMotion,
  ////                                        ba_use_weighting);

  return true;
}

bool ExperimentFactory::CreateValidationHarrisFreak(
    const int block_sz,
    const int filter_sz,
    const double k,
    const double thr,
    const int margin,
    const int max_features,
    const int max_tracks,
    const bool freak_normalize_orientation,
    const double freak_pattern_scale,
    const int max_xdiff,
    const int max_dist_stereo,
    const int max_dist_mono,
    std::shared_ptr<track::StereoTrackerBase>* stereo_tracker) {
  std::shared_ptr<track::FeatureDetectorBase> left_detector =
    std::make_shared<track::FeatureDetectorHarrisCV>(block_sz, filter_sz, k, thr,
                                                     margin, max_features);
  std::shared_ptr<track::FeatureDetectorBase> right_detector = left_detector;
  *stereo_tracker = std::make_shared<track::StereoTrackerFREAK>(
      left_detector, right_detector, max_tracks, max_xdiff, 0, 140, 40,
      max_dist_stereo, max_dist_mono, freak_normalize_orientation, freak_pattern_scale);
  return true;
}

bool ExperimentFactory::CreateValidationAgastFreak(
    const int agast_thr,
    const std::string agast_type,
    const bool freak_normalize_orientation,
    const float freak_pattern_scale,
    const int max_xdiff,
    const int max_dist_stereo,
    const int max_dist_mono,
    std::shared_ptr<track::StereoTrackerBase>* stereo_tracker) {
  int max_tracks = 10000;
  //int max_xdiff = 100;
  int max_disparity = 140;

  std::shared_ptr<track::FeatureDetectorBase> left_detector =
      std::make_shared<track::FeatureDetectorAGAST>(agast_thr, true, agast_type);
  std::shared_ptr<track::FeatureDetectorBase> right_detector =
      std::make_shared<track::FeatureDetectorAGAST>(agast_thr, false, agast_type);
  *stereo_tracker = std::make_shared<track::StereoTrackerFREAK>(
      left_detector, right_detector, max_tracks, max_xdiff, 0, max_disparity, 40, max_dist_stereo,
      max_dist_mono, freak_normalize_orientation, freak_pattern_scale);

  return true;
}

bool ExperimentFactory::CreateValidationORB(int patch_size, int num_levels,
                         double scale_factor, int max_dist_stereo, int max_dist_mono,
                         std::shared_ptr<track::StereoTrackerBase>* stereo_tracker) {
  int max_features = 4096;
  int max_xdiff = 100;
  int max_disparity = 120;
  *stereo_tracker = std::make_shared<track::StereoTrackerORB>(max_features, max_xdiff, 0,
      max_disparity, 40, patch_size, scale_factor, num_levels, max_dist_stereo, max_dist_mono);

  return true;
}


}


#endif
