#include "math_helper.h"

namespace vo {

using namespace std;
using namespace cv;


void MathHelper::getReprojError(const deque<vector<KeyPoint>>& features_left,
                                const deque<vector<KeyPoint>>& features_right,
                                const double base, const Mat& C, const Mat& Rt,
                                double& err_left, double& err_right)
{
   assert(features_left.size() == features_right.size());
   vector<Mat> pts3d;
   Mat pt(4, 1, CV_64F);
   double d;
   double f = C.at<double>(0,0);
   double cu = C.at<double>(0,2);
   double cv = C.at<double>(1,2);
   // cut last row (0, 0, 0, 1)
   //Mat Rt_3x4(Rt.clone(), Range(0,3), Range(0,4));
   Mat C_3x4;
   Mat ccol = Mat::zeros(3, 1, CV_64F);
   hconcat(C, ccol, C_3x4);
   Mat Tb = Mat::eye(4, 4, CV_64F);
   Tb.at<double>(0,3) = -base;

   // triangulate points in coord system one frame before last
   size_t last = features_left.size() - 1;
   for(size_t i = 0; i < features_left[last-1].size(); i++) {
      d = max(features_left[last-1][i].pt.x - features_right[last-1][i].pt.x, 1.f);
      pt.at<double>(0,0) = (features_left[last-1][i].pt.x - cu) * base / d;
      pt.at<double>(1,0) = (features_left[last-1][i].pt.y - cv) * base / d;
      pt.at<double>(2,0) = f * base / d;
      pt.at<double>(3,0) = 1.0;
      // clone because opencv Mat will not create new object instance if objects left and right of = are same size and same type
      pts3d.push_back(pt.clone());
   }

   err_left = 0.0;
   err_right = 0.0;
   double dx, dy;
   Mat pt2d_h, pt2d;
   for(size_t i = 0; i < pts3d.size(); i++) {
      // transform 3d point to coord system in last frame with Rt
      // then project 3d point to image plane with camera matrix C

      // error for left camera
      pt2d_h = C_3x4 * Rt * pts3d[i];
      pt2d = pt2d_h / pt2d_h.at<double>(0,2);
      //      cout << i << ". pt proj: (" << features_left[1][i].pt.x << ", " << features_left[1][i].pt.y << ")  ->  ";
      //      cout << "(" << pt2d.at<double>(0,0) << ", " << pt2d.at<double>(0,1) << ", " << pt2d.at<double>(0,2) << ")\n";
      // calculate reprojection error for current frame
      dx = features_left[last][i].pt.x - pt2d.at<double>(0,0);
      dy = features_left[last][i].pt.y - pt2d.at<double>(0,1);
      err_left += dx*dx + dy*dy;

      // error for right camera - need to shift 3d point for baseline with Tb
      //pts3d[i].at<double>(0,0) -= base;
      pt2d_h = C_3x4 * Tb * Rt * pts3d[i];
      pt2d = pt2d_h / pt2d_h.at<double>(0,2);
      dx = features_right[last][i].pt.x - pt2d.at<double>(0,0);
      dy = features_right[last][i].pt.y - pt2d.at<double>(0,1);
      err_right += dx*dx + dy*dy;

      // caluclate for previous frame - this one should be zero for this triangulation method - sanity check
//      pt2d_h = C * pts3d[i](Range(0,3), Range::all());
//      pt2d = pt2d_h / pt2d_h.at<double>(0,2);
//      cout << pt2d << endl;
//      dx = features_left[last-1][i].pt.x - pt2d.at<double>(0,0);
//      dy = features_left[last-1][i].pt.y - pt2d.at<double>(0,1);
//      cout << "dx: " << dx << "\ndy: " << dy << endl;
   }
   //cout << "right error: " << err_right / pts3d.size() << endl;
   err_left /= pts3d.size();
   err_right /= pts3d.size();
}

double MathHelper::getReprojError(Point2f& px, Point3f& pt, Mat& C)
{
   Mat pt3d(3, 1, CV_64F);
   pt3d.at<double>(0,0) = pt.x;
   pt3d.at<double>(1,0) = pt.y;
   pt3d.at<double>(2,0) = pt.z;
   Mat proj = C * pt3d;
   proj.at<double>(0) /= proj.at<double>(2);
   proj.at<double>(1) /= proj.at<double>(2);
   double dx = proj.at<double>(0) - px.x;
   double dy = proj.at<double>(1) - px.y;
   return sqrt(dx*dx + dy*dy);
}


inline double SIGN(double x) {return (x >= 0.0) ? +1.0 : -1.0;}
inline double NORM(double a, double b, double c, double d) {return sqrt(a * a + b * b + c * c + d * d);}


void MathHelper::matToQuat(const Mat& m, double quat[4])
{
   double tr = m.at<double>(0,0) + m.at<double>(1,1) + m.at<double>(2,2);

   if (tr > 0) {
      double S = sqrt(tr+1.0) * 2; // S=4*quat[0]
      quat[0] = 0.25 * S;
      quat[1] = (m.at<double>(2,1) - m.at<double>(1,2)) / S;
      quat[2] = (m.at<double>(0,2) - m.at<double>(2,0)) / S;
      quat[3] = (m.at<double>(1,0) - m.at<double>(0,1)) / S;
   } else if ((m.at<double>(0,0) > m.at<double>(1,1))&(m.at<double>(0,0) > m.at<double>(2,2))) {
      double S = sqrt(1.0 + m.at<double>(0,0) - m.at<double>(1,1) - m.at<double>(2,2)) * 2; // S=4*quat[1]
      quat[0] = (m.at<double>(2,1) - m.at<double>(1,2)) / S;
      quat[1] = 0.25 * S;
      quat[2] = (m.at<double>(0,1) + m.at<double>(1,0)) / S;
      quat[3] = (m.at<double>(0,2) + m.at<double>(2,0)) / S;
   } else if (m.at<double>(1,1) > m.at<double>(2,2)) {
      double S = sqrt(1.0 + m.at<double>(1,1) - m.at<double>(0,0) - m.at<double>(2,2)) * 2; // S=4*quat[2]
      quat[0] = (m.at<double>(0,2) - m.at<double>(2,0)) / S;
      quat[1] = (m.at<double>(0,1) + m.at<double>(1,0)) / S;
      quat[2] = 0.25 * S;
      quat[3] = (m.at<double>(1,2) + m.at<double>(2,1)) / S;
   } else {
      double S = sqrt(1.0 + m.at<double>(2,2) - m.at<double>(0,0) - m.at<double>(1,1)) * 2; // S=4*quat[3]
      quat[0] = (m.at<double>(1,0) - m.at<double>(0,1)) / S;
      quat[1] = (m.at<double>(0,2) + m.at<double>(2,0)) / S;
      quat[2] = (m.at<double>(1,2) + m.at<double>(2,1)) / S;
      quat[3] = 0.25 * S;
   }
}


void MathHelper::quatToMat(const double trans_vec[7], Mat& Rt)
{
   double w = trans_vec[0];
   double x = trans_vec[1];
   double y = trans_vec[2];
   double z = trans_vec[3];
   double tx = trans_vec[4];
   double ty = trans_vec[5];
   double tz = trans_vec[6];
   // left handed or right? post/pre-multiply?
   double array[16] = {1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,	 2*x*z + 2*y*w,	      tx ,
      2*x*y + 2*z*w,	    1 - 2*x*x - 2*z*z,	 2*y*z - 2*x*w,	      ty,
      2*x*z - 2*y*w,	    2*y*z + 2*x*w,	 1 - 2*x*x - 2*y*y,   tz,
      0.0, 0.0, 0.0, 1.0};

   Mat tmp = Mat(4, 4, CV_64F, &array); // with = in opencv no data is copied
   Rt = tmp.clone(); // now we copy the local data (array)
}

void MathHelper::matToQuat2(const Mat& m, double q[4])
{
   assert((m.rows == 4 && m.cols == 4) || (m.rows == 3 && m.cols == 3));
   q[0] = ( m.at<double>(0,0) + m.at<double>(1,1) + m.at<double>(2,2) + 1.0) / 4.0;
   q[1] = ( m.at<double>(0,0) - m.at<double>(1,1) - m.at<double>(2,2) + 1.0) / 4.0;
   q[2] = (-m.at<double>(0,0) + m.at<double>(1,1) - m.at<double>(2,2) + 1.0) / 4.0;
   q[3] = (-m.at<double>(0,0) - m.at<double>(1,1) + m.at<double>(2,2) + 1.0) / 4.0;
   if(q[0] < 0.0f) q[0] = 0.0f;
   if(q[1] < 0.0f) q[1] = 0.0f;
   if(q[2] < 0.0f) q[2] = 0.0f;
   if(q[3] < 0.0f) q[3] = 0.0f;
   q[0] = sqrt(q[0]);
   q[1] = sqrt(q[1]);
   q[2] = sqrt(q[2]);
   q[3] = sqrt(q[3]);
   if(q[0] >= q[1] && q[0] >= q[2] && q[0] >= q[3]) {
      q[0] *= +1.0;
      q[1] *= SIGN(m.at<double>(2,1) - m.at<double>(1,2));
      q[2] *= SIGN(m.at<double>(0,2) - m.at<double>(2,0));
      q[3] *= SIGN(m.at<double>(1,0) - m.at<double>(0,1));
   } else if(q[1] >= q[0] && q[1] >= q[2] && q[1] >= q[3]) {
      q[0] *= SIGN(m.at<double>(2,1) - m.at<double>(1,2));
      q[1] *= +1.0;
      q[2] *= SIGN(m.at<double>(1,0) + m.at<double>(0,1));
      q[3] *= SIGN(m.at<double>(0,2) + m.at<double>(2,0));
   } else if(q[2] >= q[0] && q[2] >= q[1] && q[2] >= q[3]) {
      q[0] *= SIGN(m.at<double>(0,2) - m.at<double>(2,0));
      q[1] *= SIGN(m.at<double>(1,0) + m.at<double>(0,1));
      q[2] *= +1.0;
      q[3] *= SIGN(m.at<double>(2,1) + m.at<double>(1,2));
   } else if(q[3] >= q[0] && q[3] >= q[1] && q[3] >= q[2]) {
      q[0] *= SIGN(m.at<double>(1,0) - m.at<double>(0,1));
      q[1] *= SIGN(m.at<double>(2,0) + m.at<double>(0,2));
      q[2] *= SIGN(m.at<double>(2,1) + m.at<double>(1,2));
      q[3] *= +1.0;
   } else {
      printf("coding error\n");
   }
   double r = NORM(q[0], q[1], q[2], q[3]);
   q[0] /= r;
   q[1] /= r;
   q[2] /= r;
   q[3] /= r;
}


// TODO - is it correct?
void MathHelper::matrixToQuaternion(const libviso::Matrix& m, double quat[4])
{
   double tr = m.val[0][0] + m.val[1][1] + m.val[2][2];

   if (tr > 0) {
      double S = sqrt(tr+1.0) * 2; // S=4*quat[0]
      quat[0] = 0.25 * S;
      quat[1] = (m.val[2][1] - m.val[1][2]) / S;
      quat[2] = (m.val[0][2] - m.val[2][0]) / S;
      quat[3] = (m.val[1][0] - m.val[0][1]) / S;
   } else if ((m.val[0][0] > m.val[1][1])&(m.val[0][0] > m.val[2][2])) {
      double S = sqrt(1.0 + m.val[0][0] - m.val[1][1] - m.val[2][2]) * 2; // S=4*quat[1]
      quat[0] = (m.val[2][1] - m.val[1][2]) / S;
      quat[1] = 0.25 * S;
      quat[2] = (m.val[0][1] + m.val[1][0]) / S;
      quat[3] = (m.val[0][2] + m.val[2][0]) / S;
   } else if (m.val[1][1] > m.val[2][2]) {
      double S = sqrt(1.0 + m.val[1][1] - m.val[0][0] - m.val[2][2]) * 2; // S=4*quat[2]
      quat[0] = (m.val[0][2] - m.val[2][0]) / S;
      quat[1] = (m.val[0][1] + m.val[1][0]) / S;
      quat[2] = 0.25 * S;
      quat[3] = (m.val[1][2] + m.val[2][1]) / S;
   } else {
      double S = sqrt(1.0 + m.val[2][2] - m.val[0][0] - m.val[1][1]) * 2; // S=4*quat[3]
      quat[0] = (m.val[1][0] - m.val[0][1]) / S;
      quat[1] = (m.val[0][2] + m.val[2][0]) / S;
      quat[2] = (m.val[1][2] + m.val[2][1]) / S;
      quat[3] = 0.25 * S;
   }
}


void MathHelper::matrixToMat(const libviso::Matrix& src, Mat& dst)
{
   dst = Mat::zeros(src.m, src.n, CV_64F);
   //dst = tmp.clone();
   for(int i = 0; i < src.m; i++) {
      for(int j = 0; j < src.n; j++)
         dst.at<double>(i,j) = src.val[i][j];
   }
}

void MathHelper::transMatToQuatVec(const Mat& pose_mat, Vec<double,7>& trans_vec)
{
   double quat[4];
   // calculate quaternion from rotation matrix
   matToQuat(pose_mat, quat);
   //cout << "first method: " << quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << endl;
   //matToQuat2(pose_mat, quat); // sometimes returns negative scalar which sba doesn't like 
   //cout << "second method: " << quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << endl;
   for(int i = 0; i < 4; i++)
      trans_vec.val[i] = quat[i];
   // write translation params
   for(int i = 4; i < 7; i++)
      trans_vec.val[i] = pose_mat.at<double>(i-4, 3);

   // TODO
   //trans_vec.val[5] = - trans_vec.val[5]; maybe switch x and y?
}

// TODO check this shit
void MathHelper::invTrans(const Mat& src, Mat& dst)
{
   Mat R(src, Range(0,3), Range(0,3));
   dst = Mat::eye(4, 4, CV_64F);
   //dst = tmp.clone();
   Mat RT = R.t();
   Mat t(src, Range(0,3), Range(3,4));
   Mat RTt = - RT * t;
   for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
         dst.at<double>(i,j) = RT.at<double>(i,j);
      }
   }
   for(int i = 0; i < 3; i++)
      dst.at<double>(i,3) = RTt.at<double>(i);
}

}
