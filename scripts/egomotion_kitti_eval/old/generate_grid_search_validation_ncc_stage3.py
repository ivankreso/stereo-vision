#!/usr/bin/python

horizontal_bins = [10, 17]
vertical_bins = [10, 5]
fpb = [40, 50, 60, 70, 80, 90, 100]
harris_thr = ["1e-06", "1e-07"]

fp_runscript = open("/mnt/ssd/kivan/cv-stereo/scripts/eval_batch/run_grid_search_ncc_stage3.sh", 'w')
fp_runscript.write("#!/bin/bash\n\n")

cnt = 0
for i in range(len(horizontal_bins)):
    for j in range(len(fpb)):
        for k in range(len(harris_thr)):
            cnt += 1
            filepath = "/home/kivan/Projects/cv-stereo/config_files/experiments/kitti/validation/ncc_stage3/ncc_validation_stage3_" + str(cnt) + ".txt"
            print(filepath)
            fp = open(filepath, 'w')
            fp.write("odometry_method = VisualOdometryRansac\n")
            fp.write("ransac_iters    = 1000\n")
            fp.write("use_deformation_field = false\n\n")
            fp.write("tracker         = StereoTracker\n")
            fp.write("max_disparity   = 160\n")
            fp.write("stereo_wsz      = 11\n")
            fp.write("ncc_threshold_s = 0.7\n\n")
            fp.write("tracker_mono    = TrackerBFM\n")
            fp.write("max_features    = 4096\n")
            fp.write("ncc_threshold_m = 0.8\n")
            fp.write("ncc_patch_size  = 15\n")
            fp.write("search_wsz      = 230\n\n")
            fp.write("detector  = FeatureDetectorUniform\n")
            fp.write("horizontal_bins   = " + str(horizontal_bins[i]) + "\n")
            fp.write("vertical_bins     = " + str(vertical_bins[i]) + "\n")
            fp.write("features_per_bin  = " + str(fpb[j]) + "\n")
            fp.write("harris_block_sz   = 3\n")
            fp.write("harris_filter_sz  = 1\n")
            fp.write("harris_k          = 0.04\n")
            fp.write("harris_thr        = " + harris_thr[k] + "\n")
            fp.write("harris_margin     = 15\n\n")
            fp.write("use_bundle_adjustment   = false")
            fp.close()
            fp_runscript.write('./run_kitti_tracker_validation.sh "' + filepath + '"\n')
fp_runscript.close()
