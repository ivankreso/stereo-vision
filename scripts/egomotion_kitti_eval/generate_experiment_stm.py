#!/usr/bin/python

stm_a = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
stm_q = [0.9, 0.95, 1.0, 1.05, 1.1]

fp_runscript = open("/home/kivan/source/cv-stereo/scripts/egomotion_kitti_eval/run_validation_stm.sh", 'w')
fp_runscript.write("#!/bin/zsh\n\n")
fp_runscript.write("export OMP_NUM_THREADS=8\n")
prefix = "/home/kivan/source/cv-stereo/config_files/experiments/kitti/validation/stm/stm_"
cnt = 0
for i in range(len(stm_q)):
    for j in range(len(stm_a)):
        cnt += 1
        #filepath = prefix + str(cnt) + ".txt"
        filepath = prefix + str(stm_q[i]) + "_" + str(stm_a[j]) + ".txt"
        print(filepath)
        fp = open(filepath, 'w')
        fp.write("egomotion_method      = EgomotionRansac\n")
        fp.write("ransac_iters          = 1000\n")
        fp.write("ransac_threshold      = 1.5\n")
        fp.write("loss_function_type    = Squared\n")
        fp.write("use_weighting         = false\n")
        fp.write("use_deformation_field = false\n\n")

        fp.write("tracker         = StereoTracker\n")
        fp.write("max_disparity   = 160\n")
        fp.write("stereo_wsz      = 11\n")
        fp.write("ncc_threshold_s = 0.7\n")
        fp.write("estimate_subpixel = true\n\n")

        fp.write("tracker_mono    = TrackerSTM\n")
        fp.write("tracker_stm     = TrackerBFM\n")
        fp.write("stm_q           = " + str(stm_q[i]) + "\n")
        fp.write("stm_a           = " + str(stm_a[j]) + "\n")
        fp.write("max_features    = 4096\n")
        fp.write("ncc_threshold_m = 0.8\n")
        fp.write("ncc_patch_size  = 15\n")
        fp.write("search_wsz      = 230\n\n")

        fp.write("detector          = FeatureDetectorUniform\n")
        fp.write("horizontal_bins   = 10\n")
        fp.write("vertical_bins     = 10\n")
        fp.write("features_per_bin  = 50\n")
        fp.write("harris_block_sz   = 3\n")
        fp.write("harris_filter_sz  = 1\n")
        fp.write("harris_k          = 0.04\n")
        fp.write("harris_thr        = 1e-07\n")
        fp.write("harris_margin     = 15\n\n")

        fp.write("use_bundle_adjustment   = true\n")
        fp.write("bundle_adjuster         = BundleAdjuster\n")
        fp.write("ba_num_frames           = 6\n")
        fp.close()

        fp_runscript.write('./run_evaluation_i7.sh "' + filepath + '"\n')
fp_runscript.close()
