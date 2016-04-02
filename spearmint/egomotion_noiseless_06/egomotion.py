import libegomotion

def main(job_id, params):
    print 'Job #%d' % job_id
    print params
    loss = libegomotion.run(
        "/home/kivan/source/cv-stereo/config_files/config_kitti_06.txt",
        int(params['patch_size'][0]), int(params['num_levels'][0]),
        float(params['scale_factor'][0]), int(params['max_dist_stereo'][0]),
        int(params['max_dist_mono'][0]))
    print loss
    return loss
