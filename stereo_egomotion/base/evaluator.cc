#include "evaluator.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <tuple>
#include <limits>
#include <algorithm>
#include <iomanip>

#include <dirent.h>
#include <sys/stat.h>

namespace egomotion {

typedef std::tuple<std::string,double,double> ResultTuple;

using namespace std;
using namespace libviso;

namespace {

float lengths[] = {100,200,300,400,500,600,700,800};
int32_t num_lengths = 8;

struct errors {
  int32_t first_frame;
  float   r_err;
  float   t_err;
  float   len;
  float   speed;
  errors (int32_t first_frame,float r_err,float t_err,float len,float speed) :
    first_frame(first_frame),r_err(r_err),t_err(t_err),len(len),speed(speed) {}
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

vector<float> trajectoryDistances(const std::vector<Matrix>& poses) {
  vector<float> dist;
  dist.push_back(0);
  for (int32_t i=1; i<poses.size(); i++) {
    Matrix P1 = poses[i-1];
    Matrix P2 = poses[i];
    float dx = P1.val[0][3]-P2.val[0][3];
    float dy = P1.val[1][3]-P2.val[1][3];
    float dz = P1.val[2][3]-P2.val[2][3];
    dist.push_back(dist[i-1]+sqrt(dx*dx+dy*dy+dz*dz));
  }
  return dist;
}

int32_t lastFrameFromSegmentLength(vector<float> &dist,int32_t first_frame,float len) {
  for (int32_t i=first_frame; i<dist.size(); i++)
    if (dist[i]>dist[first_frame]+len)
      return i;
  return -1;
}

inline float rotationError(const Matrix& pose_error) {
  float a = pose_error.val[0][0];
  float b = pose_error.val[1][1];
  float c = pose_error.val[2][2];
  float d = 0.5*(a+b+c-1.0);
  return acos(max(min(d,1.0f),-1.0f));
}

inline float translationError(const Matrix& pose_error) {
  float dx = pose_error.val[0][3];
  float dy = pose_error.val[1][3];
  float dz = pose_error.val[2][3];
  return sqrt(dx*dx+dy*dy+dz*dz);
}

std::vector<errors> CalcSequenceErrors(const std::vector<Matrix>& poses_gt,
                                       const std::vector<Matrix>& poses_result) {
  // error vector
  std::vector<errors> err;

  // parameters
  int32_t step_size = 10; // every second
  
  // pre-compute distances (from ground truth as reference)
  vector<float> dist = trajectoryDistances(poses_gt);
 
  // for all start positions do
  for (int32_t first_frame=0; first_frame<poses_gt.size(); first_frame+=step_size) {
  
    // for all segment lengths do
    for (int32_t i=0; i<num_lengths; i++) {
    
      // current length
      float len = lengths[i];
      
      // compute last frame
      int32_t last_frame = lastFrameFromSegmentLength(dist,first_frame,len);
      
      // continue, if sequence not long enough
      if (last_frame==-1)
        continue;

      // compute rotational and translational errors
      Matrix pose_delta_gt     = Matrix::inv(poses_gt[first_frame])*poses_gt[last_frame];
      Matrix pose_delta_result = Matrix::inv(poses_result[first_frame])*poses_result[last_frame];
      Matrix pose_error        = Matrix::inv(pose_delta_result)*pose_delta_gt;
      float r_err = rotationError(pose_error);
      float t_err = translationError(pose_error);
      
      // compute speed
      float num_frames = (float)(last_frame-first_frame+1);
      float speed = len/(0.1*num_frames);
      
      // write to file
      err.push_back(errors(first_frame,r_err/len,t_err/len,len,speed));
    }
  }

  // return error vector
  return err;
}

void GetStats(std::vector<errors> errors, double& t_error, double& r_error) {
  double t_err = 0;
  double r_err = 0;

  // for all errors do => compute sum of t_err, r_err
  //for(std::vector<errors>::iterator it=err.begin(); it!=err.end(); it++) {
  for (const auto& e : errors) {
    t_err += e.t_err;
    r_err += e.r_err;
  }

  double num = errors.size();

  t_error = 100.0 * t_err / num;
  r_error = 100.0 * r_err / num;
}

} // empty namespace

void Evaluator::Eval(const std::string& gt_fname, const std::vector<libviso::Matrix>& egomotion_poses,
                     double& trans_error, double& rot_error) {
  // read ground truth and result poses
  std::vector<Matrix> poses_gt = loadPoses(gt_fname);

  // check for errors
  if(poses_gt.size() == 0 || egomotion_poses.size() != poses_gt.size())
    throw 1;

  // compute sequence errors
  std::vector<errors> errors = CalcSequenceErrors(poses_gt, egomotion_poses);

  if(errors.size() > 0)
    GetStats(errors, trans_error, rot_error);
  else throw 1;
}

libviso::Matrix Evaluator::EigenMatrixToLibvisoMatrix(const Eigen::Matrix4d& mat1) {
  libviso::Matrix mat2 = libviso::Matrix(mat1.rows(), mat1.cols());
  for (int i = 0; i < mat1.rows(); i++)
    for (int j = 0; j < mat1.cols(); j++)
      mat2.val[i][j] = mat1(i,j);
  return mat2;
}

void Evaluator::EigenVectorToLibvisoVector(const std::vector<Eigen::Matrix4d>& vec1,
                                           std::vector<libviso::Matrix>& vec2) {
  vec2.clear();
  for (const auto& rt : vec1)
    vec2.push_back(EigenMatrixToLibvisoMatrix(rt));
}

void Evaluator::Eval(const std::string& gt_fname, const std::vector<Eigen::Matrix4d>& egomotion_poses,
          double& trans_error, double& rot_error) {
  std::vector<libviso::Matrix> libviso_poses;
  EigenVectorToLibvisoVector(egomotion_poses, libviso_poses);
  Eval(gt_fname, libviso_poses, trans_error, rot_error);
}

}   // namespace egomotion
