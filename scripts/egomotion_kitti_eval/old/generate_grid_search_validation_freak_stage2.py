#!/usr/bin/python

hamming_threshold = [50, 60]
pattern_scale = [4.0, 6.0, 8.0, 10.0]

fp_runscript = open("/mnt/ssd/kivan/cv-stereo/scripts/eval_batch/run_batch_validation.sh", 'w')
fp_runscript.write("#!/bin/bash\n\n")

cnt = 0
for i in range(len(hamming_threshold)):
    for j in range(len(pattern_scale)):
        cnt += 1
        filepath = "/home/kivan/Projects/cv-stereo/config_files/experiments/kitti/validation_freak/freak_tracker_validation_stage2_" + str(cnt) + ".txt"
        print(filepath)
        fp = open(filepath, 'w')
        fp.write("odometry_method = VisualOdometryRansac\n")
        fp.write("use_deformation_field = false\n")
        fp.write("ransac_iters    = 1000\n\n")
        fp.write("tracker         = StereoTracker\n")
        fp.write("max_disparity   = 160\n")
        fp.write("stereo_wsz      = 15\n")
        fp.write("ncc_threshold_s = 0.7\n\n")
        fp.write("tracker_mono    = TrackerBFMcv\n")
        fp.write("max_features    = 5000\n")
        fp.write("search_wsz      = 230\n\n")
        fp.write("hamming_threshold   = " + str(hamming_threshold[i]) + "\n\n")

        fp.write("detector  = FeatureDetectorHarrisFREAK\n")
        fp.write("harris_block_sz   = 3\n")
        fp.write("harris_filter_sz  = 1\n")
        fp.write("harris_k          = 0.04\n")
        fp.write("harris_thr        = 1e-06\n")
        fp.write("harris_margin     = 15\n\n")
        fp.write("freak_norm_scale    = false\n")
        fp.write("freak_norm_orient   = false\n")
        fp.write("freak_pattern_scale = " + str(pattern_scale[j]) + "\n")
        fp.write("freak_num_octaves   = 0\n")

        fp.write("use_bundle_adjustment   = false")
        fp.close()

        fp_runscript.write('./run_kitti_evaluation_dinodas.sh "' + filepath + '"\n')

fp_runscript.close()
