#!/usr/bin/python

patch_size = [11, 15, 21]
detector = ["FeatureDetectorHarrisCV", "FeatureDetectorUniform"]
filter_size = [1, 3]
max_disparity = [140, 160, 180, 200]

fp_runscript = open("/mnt/ssd/kivan/cv-stereo/scripts/eval_batch/run_batch_grid_search_stage2.sh", 'w')
fp_runscript.write("#!/bin/bash\n\n")

cnt = 0
for i in range(len(patch_size)):
    for j in range(len(detector)):
        for k in range(len(filter_size)):
            for l in range(len(max_disparity)):
                cnt += 1
                filepath = "/home/kivan/Projects/cv-stereo/config_files/experiments/kitti/validation_stage2/param_validation_stage2_" + str(cnt) + ".txt"
                print(filepath)
                fp = open(filepath, 'w')
                fp.write("odometry_method = VisualOdometryRansac\n")
                fp.write("ransac_iters    = 1000\n\n")
                fp.write("tracker         = StereoTracker\n")
                fp.write("max_disparity   = " + str(max_disparity[l]) + "\n")
                fp.write("stereo_wsz      = " + str(patch_size[i]) + "\n")
                fp.write("ncc_threshold_s = 0.7\n\n")
                fp.write("tracker_mono    = TrackerBFM\n")
                fp.write("max_features    = 5000\n")
                fp.write("ncc_threshold_m = 0.8\n")
                fp.write("ncc_patch_size  = " + str(patch_size[i]) + "\n")
                fp.write("search_wsz      = 230\n\n")
                fp.write("detector  = " + detector[j] + "\n")
                fp.write("harris_block_sz   = 3\n")
                fp.write("harris_filter_sz  = " + str(filter_size[k]) + "\n")
                fp.write("harris_k          = 0.04\n")
                fp.write("harris_thr        = 1e-05\n")
                fp.write("harris_margin     = " + str(patch_size[i]) + "\n\n")
                fp.write("use_bundle_adjustment   = false")
                fp.close()

                fp_runscript.write('./run_kitti_evaluation_dinodas.sh "' + filepath + '"\n')
fp_runscript.close()
