#!/usr/bin/python

ransac_iters = [1000, 5000]
ncc_threshold_s = [0.6, 0.7, 0.8]
ncc_threshold_m = [0.6, 0.7, 0.8]

#search_wsz = [180, 200, 230]

harris_block_sz = [3, 5]
harris_k = [0.04, 0.06]
harris_thr = [0.00001, 0.000001, 0.0000001]

fp_runscript = open("/mnt/ssd/kivan/cv-stereo/scripts/eval_batch/run_batch_validation.sh", 'w')
fp_runscript.write("#!/bin/bash\n\n")

cnt = 0
for i in range(len(ransac_iters)):
    for j in range(len(ncc_threshold_s)):
        for k in range(len(ncc_threshold_m)):
            for l in range(len(harris_block_sz)):
                for m in range(len(harris_k)):
                    for n in range(len(harris_thr)):
                        cnt += 1
                        #filepath = "/home/kivan/Projects/cv-stereo/config_files/experiments/kitti/validation/tracker_validation_ncc_" + str(cnt) + ".txt"
                        filepath = "/home/kivan/Projects/cv-stereo/config_files/experiments/kitti/validation2/tracker_validation_ncc_" + str(cnt) + ".txt"
                        print(filepath)
                        fp = open(filepath, 'w')
                        fp.write("odometry_method = VisualOdometryRansac\n")
                        fp.write("ransac_iters    = " + str(ransac_iters[i]) + "\n\n")
                        fp.write("tracker         = StereoTracker\n")
                        fp.write("max_disparity   = 160\n")
                        fp.write("stereo_wsz      = 15\n")
                        fp.write("ncc_threshold_s = " + str(ncc_threshold_s[j]) + "\n\n")
                        fp.write("tracker_mono    = TrackerBFM\n")
                        fp.write("max_features    = 5000\n")
                        fp.write("ncc_threshold_m = " + str(ncc_threshold_m[k]) + "\n")
                        fp.write("ncc_patch_size  = 15\n")
                        fp.write("search_wsz      = 230\n\n")
                        fp.write("detector  = FeatureDetectorHarrisCV\n")
                        fp.write("harris_block_sz   = " + str(harris_block_sz[l]) + "\n")
                        fp.write("harris_filter_sz  = 1\n")
                        fp.write("harris_k          = " + str(harris_k[m]) + "\n")
                        fp.write("harris_thr        = " + str(harris_thr[n]) + "\n")
                        fp.write("harris_margin     = 15\n\n")
                        fp.write("use_bundle_adjustment   = false")
                        fp.close()

                        fp_runscript.write('./run_kitti_evaluation_dinodas.sh "' + filepath + '"\n')
fp_runscript.close()
