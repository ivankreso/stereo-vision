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

typedef std::tuple<std::string,double,double> ResultTuple;

using namespace std;

struct errors {
  int32_t frame;
  float   r_err;
  float   t_err;
  errors(int32_t frame, float r_err, float t_err) :
    frame(frame), r_err(r_err), t_err(t_err) {}
};

std::vector<Matrix> loadPoses(string file_name) {
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

std::vector<errors> CalcSequenceErrors(std::vector<Matrix> &poses_gt, std::vector<Matrix> &poses_result)
{
  // error vector
  std::vector<errors> err;
  // parameters
  int32_t step_size = 1; // every frame
  //// pre-compute distances (from ground truth as reference)
  //std::vector<float> dist = trajectoryDistances(poses_gt);
  //std::cout << "Max dist = " << dist.back() << "\n";

  // for all start positions do
  for(int i = 1; i < poses_result.size(); i += step_size) {
    // compute rotational and translational errors
    Matrix pose_delta_gt     = Matrix::inv(poses_gt[i-1]) * poses_gt[i];
    Matrix pose_delta_result = Matrix::inv(poses_result[i-1]) * poses_result[i];
    Matrix pose_error        = Matrix::inv(pose_delta_result) * pose_delta_gt;
    //std::cout << "Delta GT:\n" << pose_delta_gt << "\nDelta VO:\n" << pose_delta_result << "\n\n\n";
    //std::cout << pose_error << '\n';
    float r_err = rotationError(pose_error);
    float t_err = translationError(pose_error);

    // write to file
    err.push_back(errors(i, r_err, t_err));
  }
  // return error vector
  return err;
}

void SaveSequenceErrors(std::vector<errors> &err, const std::string file_name) {
  // open file  
  FILE *fp;
  fp = fopen(file_name.c_str(),"w");

  // write to file
  for(std::vector<errors>::iterator it=err.begin(); it!=err.end(); it++)
    fprintf(fp,"%d %f %f\n", it->frame, it->r_err, it->t_err);

  // close file
  fclose(fp);
}

void GetStats(std::vector<errors> err, double& t_error, double& r_error) {
  float t_err = 0;
  float r_err = 0;

  // for all errors do => compute sum of t_err, r_err
  for(std::vector<errors>::iterator it=err.begin(); it!=err.end(); it++) {
    t_err += it->t_err;
    r_err += it->r_err;
  }

  float num = err.size();

  t_error = 100.0 * t_err / num;
  r_error = 100.0 * r_err / num;
}

bool Eval(const std::string& gt_fname, const std::string& vo_fname) {
  // total errors
  std::vector<errors> total_err;

  std::cout << "Loading: " << vo_fname << "\n";
  // read ground truth and result poses
  std::vector<Matrix> poses_gt     = loadPoses(gt_fname);
  std::vector<Matrix> poses_result = loadPoses(vo_fname);

  // check for errors
  if(poses_gt.size() == 0 || poses_result.size() != poses_gt.size())
    throw 1;

  // compute sequence errors
  std::vector<errors> seq_err = CalcSequenceErrors(poses_gt, poses_result);
  SaveSequenceErrors(seq_err, "errors.txt");

  // add to total errors
  total_err.insert(total_err.end(),seq_err.begin(),seq_err.end());

  // save + plot total errors + summary statistics
  if(total_err.size() > 0) {
    double t_error, r_error;
    GetStats(total_err, t_error, r_error);
    printf("T error = %f\nR error = %f\n",t_error, r_error);
  }
  else throw 1;

  // success
  return true;
}

int main (int32_t argc,char *argv[]) {
  if(argc != 3) {
    cout << "Usage: ./eval_odometry gt_file vo_file" << endl;
    return 1;
  }

  // read arguments
  std::string gt_file = argv[1];
  std::string vo_file = argv[2];
  Eval(gt_file, vo_file);

  return 0;
}

