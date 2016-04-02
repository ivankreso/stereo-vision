#!/usr/bin/python

#loss_type = ["Cauchy", "Huber"]
#loss_type = ["SoftLOneLoss", "Huber"]
loss_type = ["Cauchy"]
#robust_loss_scale = [0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 2.5]
#robust_loss_scale = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
robust_loss_scale = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 10.0]

fp_runscript = open("/home/kivan/Projects/cv-stereo/scripts/egomotion_kitti_eval/run_validation_robust_loss.sh", 'w')
fp_runscript.write("#!/bin/bash\n\n")
prefix = "/home/kivan/Projects/cv-stereo/config_files/experiments/kitti/validation/robust_loss_3/"
cnt = 0
for i in range(len(loss_type)):
    for j in range(len(robust_loss_scale)):
        cnt += 1
        #filepath = prefix + str(cnt) + ".txt"
        filepath = prefix + str(loss_type[i]) + "_" + str(robust_loss_scale[j]) + ".txt"
        print(filepath)
        fp = open(filepath, 'w')
        fp.write("egomotion_method      = EgomotionRansac\n")
        fp.write("ransac_iters          = 1000\n")
        fp.write("ransac_threshold      = 1.5\n")
        fp.write("loss_function_type    = " + loss_type[i] + "\n")
        fp.write("robust_loss_scale     = " + str(robust_loss_scale[j]) + "\n")
        fp.write("use_weighting         = false\n")
        fp.write("use_deformation_field = false\n\n")

        fp.write("tracker         = StereoTracker\n")
        fp.write("max_disparity   = 160\n")
        fp.write("stereo_wsz      = 11\n")
        fp.write("ncc_threshold_s = 0.7\n")
        fp.write("estimate_subpixel = true\n\n")

        fp.write("tracker_mono    = TrackerBFM\n")
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

        fp.write("use_bundle_adjustment   = false\n")
        fp.close()

        fp_runscript.write('./run_kitti_evaluation_dinodas.sh "' + filepath + '" &\n')
fp_runscript.close()
