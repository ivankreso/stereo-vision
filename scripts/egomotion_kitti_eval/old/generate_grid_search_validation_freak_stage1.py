#!/usr/bin/python

#freak_norm_scale = ["true", "false"]
#freak_norm_orient = ["true", "false"]
freak_norm_scale = ["false"]
freak_norm_orient = ["false"]
freak_pattern_scale = [8.0, 12.0, 16.0]
freak_num_octaves   = [0, 1]

#harris_k = [0.04, 0.06]
harris_k = [0.04]
harris_thr = [0.000001, 0.0000001, 0.00000001]

fp_runscript = open("/mnt/ssd/kivan/cv-stereo/scripts/eval_batch/run_batch_validation.sh", 'w')
fp_runscript.write("#!/bin/bash\n\n")

cnt = 0
for i in range(len(freak_norm_scale)):
    for j in range(len(freak_norm_orient)):
        for k in range(len(freak_pattern_scale)):
            for l in range(len(freak_num_octaves)):
                for m in range(len(harris_k)):
                    for n in range(len(harris_thr)):
                        cnt += 1
                        filepath = "/home/kivan/Projects/cv-stereo/config_files/experiments/kitti/validation_freak/tracker_validation_freak_" + str(cnt) + ".txt"
                        print(filepath)
                        fp = open(filepath, 'w')
                        fp.write("odometry_method = VisualOdometryRansac\n")
                        fp.write("ransac_iters    = 1000\n")
                        fp.write("use_deformation_field = false\n\n")
                        fp.write("tracker         = StereoTracker\n")
                        fp.write("max_disparity   = 160\n")
                        fp.write("stereo_wsz      = 15\n")
                        fp.write("ncc_threshold_s = 0.7\n\n")
                        fp.write("tracker_mono    = TrackerBFMcv\n")
                        fp.write("max_features    = 5000\n")
                        fp.write("search_wsz      = 230\n\n")
                        fp.write("hamming_threshold   = 50\n\n")

                        fp.write("detector  = FeatureDetectorHarrisFREAK\n")
                        fp.write("harris_block_sz   = 3\n")
                        fp.write("harris_filter_sz  = 1\n")
                        fp.write("harris_k          = " + str(harris_k[m]) + "\n")
                        fp.write("harris_thr        = " + str(harris_thr[n]) + "\n")
                        fp.write("harris_margin     = 15\n\n")
                        fp.write("freak_norm_scale    = " + str(freak_norm_scale[i]) + "\n")
                        fp.write("freak_norm_orient   = " + str(freak_norm_orient[j]) + "\n")
                        fp.write("freak_pattern_scale = " + str(freak_pattern_scale[k]) + "\n")
                        fp.write("freak_num_octaves   = " +  str(freak_num_octaves[l]) + "\n")

                        fp.write("use_bundle_adjustment   = false")
                        fp.close()

                        fp_runscript.write('./run_kitti_evaluation_dinodas.sh "' + filepath + '"\n')
fp_runscript.close()
