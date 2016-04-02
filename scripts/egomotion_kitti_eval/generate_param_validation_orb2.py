#!/usr/bin/python

patch_size = [25, 31, 35]
scale_factor = [1.15, 1.2, 1.3]
num_levels = [4, 8, 10]
max_dist = [35, 40, 45, 50]

fp_runscript = open("/home/kivan/source/cv-stereo/scripts/egomotion_kitti_eval/run_validation_orb2.sh", 'w')
fp_runscript.write("#!/bin/bash\n\n")

cnt = 0
for i in range(len(patch_size)):
    for j in range(len(num_levels)):
        for k in range(len(scale_factor)):
            for l in range(len(max_dist)):
                cnt += 1
                filepath = "/home/kivan/source/cv-stereo/config_files/experiments/kitti/validation/orb/validation_orb2_" + str(cnt) + ".txt"
                print(filepath)
                fp = open(filepath, 'w')
                fp.write("egomotion_method      = EgomotionRansac\n")
                fp.write("ransac_iters          = 1000\n")
                fp.write("ransac_threshold      = 1.5\n")
                fp.write("loss_function_type    = Squared\n")
                fp.write("use_weighting         = false\n")
                fp.write("use_deformation_field = false\n")
                fp.write("tracker               = StereoTrackerORB\n")
                fp.write("max_features          = 5000\n")
                fp.write("max_xdiff             = 100\n")
                fp.write("max_disparity         = 160\n")
                fp.write("orb_patch_size        = " + str(patch_size[i]) + "\n")
                fp.write("orb_num_levels        = " + str(num_levels[j]) + "\n")
                fp.write("orb_scale_factor      = " + str(scale_factor[k]) + "\n")
                fp.write("orb_max_dist_stereo   = " + str(max_dist[l]) + "\n")
                fp.write("orb_max_dist_mono     = " + str(max_dist[l]) + "\n")

                fp.write("use_bundle_adjustment   = true\n")
                fp.write("bundle_adjuster         = BundleAdjuster\n")
                fp.write("ba_num_frames           = 6\n")
                fp.close()

                fp_runscript.write('./run_evaluation_i7.sh "' + filepath + '"\n')
fp_runscript.close()
