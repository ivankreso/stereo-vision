import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def calculate_histogram(data, min_val, hist_size):
    min_samples = 30
    num_pts = data.shape[0]
    H = [[] for x in range(hist_size)]
    for i in range(num_pts):
        H[int(data[i,0] - min_val)].append(data[i,1])
    means = np.zeros(hist_size)
    sigmas = np.zeros(hist_size)
    for i in range(hist_size):
        if (len(H[i]) > 0):
            means[i] = np.mean(H[i])
        #if (len(H[i]) > 1):
        if (len(H[i]) > min_samples):
            sigmas[i] = np.std(H[i])
    return means, sigmas


def plot_distribution(errors, min_val, max_val):
    hist_size = max_val - min_val
    means, sigmas = calculate_histogram(errors, min_val, hist_size)
    #plt.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
    x_pts = np.linspace(min_val+0.5, max_val, hist_size)
    plt.gca().fill_between(x_pts, means-3*sigmas, means+3*sigmas, color="#dddddd")
    plt.plot(x_pts, means, 'b-', lw=2)
    #plt.plot(means, 'r--', lw=2)
    #plt.plot(means, 'ro')
    #print("Mean = ", means)
    #print("Variance = ", sigmas)

data_folder = "/home/kivan/source/cv-stereo/build/stereo_test/release/stats_01/"
errors_response = np.array(np.loadtxt(data_folder + 'errors_response.txt'))
errors_matching = np.array(np.loadtxt(data_folder + 'errors_matching.txt'))
errors_disparity = np.array(np.loadtxt(data_folder + 'errors_disparity.txt'))

min_disp = 0
max_disp = 120
fig_disp1 = plt.figure("disparity-error")
plot_distribution(errors_disparity, min_disp, max_disp)
plt.xlabel('disparity', fontsize=20)
plt.ylabel('reproj error', fontsize=20)

min_response = 50
max_response = 700
fig_resp1 = plt.figure("feature_response-error")
plot_distribution(errors_response, min_response, max_response)
plt.xlabel('feature_response', fontsize=20)
plt.ylabel('reproj error', fontsize=20)

min_distance = 0
max_distance = 140
fig_match1 = plt.figure("matching_distance-error")
plot_distribution(errors_matching, min_distance, max_distance)
plt.xlabel('matching_distance', fontsize=20)
plt.ylabel('reproj error', fontsize=20)

# plot error points
#fig_disp2 = fig3 = plt.figure(4)
#plt.plot(errors_disparity[:,0], errors_disparity[:,1], marker='.', ls='', color='b', label="x - disparity")
#plt.legend(fontsize=20)
#fig_resp2 = plt.figure(5)
#plt.plot(errors_response[:,0], errors_response[:,1], marker='.', ls='', color='r', label="x - feature response")
#plt.legend(fontsize=20)
#fig_match2 = plt.figure(6)
#plt.plot(errors_matching[:,0], errors_matching[:,1], marker='.', ls='', color='b', label="x - matching distances")
#plt.legend(fontsize=20)

fig_disp1.savefig("disparity_error_distribution.pdf", bbox_inches='tight')
fig_resp1.savefig("response_error_distribution.pdf", bbox_inches='tight')
fig_match1.savefig("matching_error_distribution.pdf", bbox_inches='tight')
#fig_disp2.savefig("disparity_error_points.pdf", bbox_inches='tight')
#fig_match2.savefig("matching_error_points.pdf", bbox_inches='tight')
#fig_resp2.savefig("response_error_points.pdf", bbox_inches='tight')

#plt.plot([0,120], [1.0,1.0], ls="--", color="k")
#fig_scale.suptitle('Scale error', fontsize=18)
#plt.xlabel('time frames', fontsize=30)
#plt.ylabel('mean reprojection error', fontsize=30)

#fig_scale = plt.figure(figsize=(12,8))
#plt.plot(error_gt, marker='o', color='r', label="GT")
#plt.plot(error_gt_refiner, marker='o', color='g', label="GT refined")
#plt.plot(error_vo, marker='o', color='b', label="VO")
#plt.plot(error_vo_refiner, marker='o', color='k', label="VO refined")
##plt.plot([0,120], [1.0,1.0], ls="--", color="k")
##fig_scale.suptitle('Scale error', fontsize=18)
#plt.xlabel('time frames', fontsize=30)
#plt.ylabel('mean reprojection error', fontsize=30)
#plt.legend(fontsize=24)

plt.show()
