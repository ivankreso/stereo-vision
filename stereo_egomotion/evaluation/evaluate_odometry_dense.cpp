#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <tuple>
#include <limits>
#include <algorithm>
#include <iomanip>

#include <dirent.h>
#include <sys/stat.h>

#include "mail.h"
#include "matrix.h"

typedef std::tuple<std::string,double,double>    ResultTuple;

using namespace std;

struct errors {
  int32_t first_frame;
  float   r_err;
  float   t_err;
  float   len;
  errors (int32_t first_frame,float r_err,float t_err,float len) :
    first_frame(first_frame),r_err(r_err),t_err(t_err),len(len) {}
};

vector<Matrix> loadPoses(string file_name) {
  vector<Matrix> poses;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return poses;
  while (!feof(fp)) {
    Matrix P = Matrix::eye(4);
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                   &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                   &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
      poses.push_back(P);
    }
  }
  fclose(fp);
  return poses;
}

vector<Matrix> loadPoses(string file_name, int n) {
  vector<Matrix> poses;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return poses;
  int cnt = 0;
  while (!feof(fp)) {
    Matrix P = Matrix::eye(4);
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                   &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                   &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
      poses.push_back(P);
      cnt++;
      if(cnt == n)
        break;
    }
  }
  if(cnt != n)
    throw 1;

  fclose(fp);
  return poses;
}

vector<float> trajectoryDistances(vector<Matrix> &poses) {
  vector<float> dist;
  dist.push_back(0);
  for (int32_t i = 1; i < poses.size(); i++) {
    Matrix P1 = poses[i-1];
    Matrix P2 = poses[i];
    float dx = P1.val[0][3]-P2.val[0][3];
    float dy = P1.val[1][3]-P2.val[1][3];
    float dz = P1.val[2][3]-P2.val[2][3];
    dist.push_back(sqrt(dx*dx+dy*dy+dz*dz));
  }
  return dist;
}

float getLength(std::vector<float>& dist, int start, int end)
{
  float sum_dist = 0.0;
  for(int i = start; i <= end; i++)
    sum_dist += dist[i];
  return sum_dist;
}

inline float rotationError(Matrix &pose_error) {
  float a = pose_error.val[0][0];
  float b = pose_error.val[1][1];
  float c = pose_error.val[2][2];
  float d = 0.5*(a+b+c-1.0);
  return acos(max(min(d,1.0f),-1.0f));
}

inline float translationError(Matrix &pose_error) {
  float dx = pose_error.val[0][3];
  float dy = pose_error.val[1][3];
  float dz = pose_error.val[2][3];
  return sqrt(dx*dx+dy*dy+dz*dz);
}

vector<errors> calcSequenceErrors(int gt_start_frame, vector<Matrix> &poses_gt,vector<Matrix> &poses_result) {

  // error vector
  vector<errors> err;

  // parameters
  int32_t step_size = 1; // every frame
  
  // pre-compute distances (from ground truth as reference)
  std::vector<float> dist = trajectoryDistances(poses_gt);
  std::cout << "Max dist = " << dist.back() << "\n";
 
  // for all start positions do
  //for (int32_t first_frame=0; first_frame<poses_gt.size(); first_frame+=step_size) {
  for(int first_frame = 0; first_frame < poses_result.size(); first_frame += step_size) {
    // for all segment lengths do
    //for (int32_t i=0; i<num_lengths; i++) {
    //for(int i = first_frame + 1; i < poses_gt.size(); i++) {
    for(int i = first_frame + 1; i < poses_result.size(); i++) {
      // compute rotational and translational errors
      Matrix pose_delta_gt     = Matrix::inv(poses_gt[gt_start_frame + first_frame]) * poses_gt[gt_start_frame + i];
      Matrix pose_delta_result = Matrix::inv(poses_result[first_frame]) * poses_result[i];
      Matrix pose_error        = Matrix::inv(pose_delta_result) * pose_delta_gt;
      //std::cout << "Frame dist = " << i - first_frame << ":\n";
      //std::cout << "Delta GT:\n" << pose_delta_gt << "\nDelta VO:\n" << pose_delta_result << "\n\n\n";
      //std::cout << pose_error << '\n';
      float r_err = rotationError(pose_error);
      float t_err = translationError(pose_error);
      
      // makes no sense to divide with this len
      float len = getLength(dist, first_frame, i);      
      //err.push_back(errors(first_frame, r_err/len, t_err/len, len));
      // write to file
      err.push_back(errors(first_frame, r_err, t_err, len));
    }
  }

  // return error vector
  return err;
}

void saveSequenceErrors (vector<errors> &err,string file_name) {

  // open file  
  FILE *fp;
  fp = fopen(file_name.c_str(),"w");
 
  // write to file
  for (vector<errors>::iterator it=err.begin(); it!=err.end(); it++)
    fprintf(fp,"%d %f %f %f\n", it->first_frame, it->r_err, it->t_err, it->len);
  
  // close file
  fclose(fp);
}

void saveStats(std::vector<errors> err, std::string dir, double& t_error, double& r_error) {
  float t_err = 0;
  float r_err = 0;

  // for all errors do => compute sum of t_err, r_err
  for (vector<errors>::iterator it=err.begin(); it!=err.end(); it++) {
    t_err += it->t_err;
    r_err += it->r_err;
  }

  float num = err.size();
  //// open file  
  //FILE *fp = fopen((dir + "/stats.txt").c_str(),"w");
  //// save errors
  //fprintf(fp,"%f %f\n",t_err/num,r_err/num);
  //
  //// close file
  //fclose(fp);

  t_error = 100.0 * t_err / num;
  r_error = 100.0 * r_err / num;
}

bool eval(int start, int end, int start_frame, const std::string& directory,
          const std::string& result_name, std::vector<ResultTuple>& results)
{
  // ground truth and result directories
  //string gt_dir         = "data/odometry/poses/";
  string gt_file         = "/home/kivan/Projects/cv-stereo/data/GT/Tsukuba/tsukuba_gt_crop.txt";
  //string gt_file         = "/home/kivan/Projects/cv-stereo/data/GT/KITTI/poses/07.txt";
  //string gt_file         = "/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/tsukuba_gt_crop.txt";
  string result_dir     = directory + "/" + result_name + "/";

  //string error_dir      = result_dir + "/errors";
  //string plot_path_dir  = result_dir + "/plot_path";
  //string plot_error_dir = result_dir + "/plot_error";
  // create output directories
  //system(("mkdir " + error_dir).c_str());
  //system(("mkdir " + plot_path_dir).c_str());
  //system(("mkdir " + plot_error_dir).c_str());
  
  // total errors
  std::vector<errors> total_err;

  // for all sequences do
  for(int32_t i = start; i <= end; i++) {
  //for (int32_t i=0; i<11; i++) {
    // file name
    char file_name[256];
    sprintf(file_name,"%02d.txt",i);
    
    std::cout << "Loading: " << result_dir + file_name << "\n";
    // read ground truth and result poses
    vector<Matrix> poses_gt     = loadPoses(gt_file);
    vector<Matrix> poses_result = loadPoses(result_dir + file_name);
   
    // check for errors
    //if (poses_gt.size()==0 || poses_result.size()!=poses_gt.size()) {
    //  return false;
    //}
    if(poses_gt.size() == 0) {
      return false;
    }

    // compute sequence errors    
    std::vector<errors> seq_err = calcSequenceErrors(start_frame, poses_gt, poses_result);
    saveSequenceErrors(seq_err, result_dir + "/errors_" + file_name);
    
    // add to total errors
    total_err.insert(total_err.end(),seq_err.begin(),seq_err.end());
    
    // for first half => plot trajectory and compute individual stats
    //if (i<=15) {
    //  // save + plot bird's eye view trajectories
    //  savePathPlot(poses_gt,poses_result,plot_path_dir + "/" + file_name);
    //  vector<int32_t> roi = computeRoi(poses_gt,poses_result);
    //  plotPathPlot(plot_path_dir,roi,i);

    //  // save + plot individual errors
    //  char prefix[16];
    //  sprintf(prefix,"%02d",i);
    //  saveErrorPlots(seq_err,plot_error_dir,prefix);
    //  plotErrorPlots(plot_error_dir,prefix);
    //}
  }
  
  // save + plot total errors + summary statistics
  if(total_err.size() > 0) {
    //char prefix[16];
    //sprintf(prefix,"avg");
    //saveErrorPlots(total_err,plot_error_dir,prefix);
    //plotErrorPlots(plot_error_dir,prefix);
    double t_error, r_error;
    saveStats(total_err, result_dir, t_error, r_error);
    std::cout << result_name << " errors:\n";
    printf("T error = %f\nR error = %f\n",t_error, r_error);
    ResultTuple result;
    results.push_back(std::make_tuple(result_name, t_error, r_error));
  }

  // success
  return true;
}

int main (int32_t argc,char *argv[]) {
  if(argc != 6) {
    cout << "Usage: ./eval_odometry start_num end_num start_frame dir_name out_filename" << endl;
    return 1;
  }

  // read arguments
  int start = std::stoi(argv[1]);
  int end = std::stoi(argv[2]);
  int start_frame = std::stoi(argv[3]);
  std::string directory = argv[4];
  std::string out_filename = argv[5];
  //std::string directory = "./results/";

  std::vector<std::string> dirs, result_names;
  DIR *dir;
  struct dirent* ent;
  //struct stat st;
  dir = opendir(directory.c_str());
  while ((ent = readdir(dir)) != NULL) {
    const string dir_name = ent->d_name;
    const string full_dir_name = directory + "/" + dir_name;
    if(dir_name[0] == '.')
      continue;
    //std::cout << dir_name << '\n';

    result_names.push_back(dir_name);
    dirs.push_back(full_dir_name);
  }
  closedir(dir);

  // run evaluation
  std::vector<ResultTuple> results;
  std::string stats_file = directory + "/../" + out_filename + ".txt";
  std::ofstream stats_fp(stats_file);
  for(size_t i = 0; i < dirs.size(); i++) {
    std::cout << "\nEvaluating: " + directory + result_names[i] << '\n';
    bool success = eval(start, end, start_frame, directory, result_names[i], results);
    if(!success) throw "error";
  }
  std::sort(std::begin(results), std::end(results), 
    [](ResultTuple const &t1, ResultTuple const &t2) {return std::get<1>(t1) < std::get<1>(t2);});

  for(size_t i = 0; i < results.size(); i++)
    stats_fp << std::left << std::setfill(' ') << std::setw(50) << std::get<0>(results[i])
             << std::get<1>(results[i]) << "\t" << std::get<2>(results[i]) << "\n";
    //stats_fp << std::get<0>(results[i]) << "\t\t" << std::get<1>(results[i])
    //         << "\t\t" << std::get<2>(results[i]) << "\n";


  return 0;
}

