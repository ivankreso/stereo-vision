#include "draw_helper.h"


namespace recon
{

namespace DrawHelper
{

void drawDEM(const recon::DEMvoting& dem, const std::vector<Eigen::Vector2d>& dem_projs,
             const std::vector<int>& class_ids, cv::Mat& img_dem)
{
  enum {GROUND, CURB, OBSTACLE};
  cv::Scalar color;
  // draw dots
  //for(size_t i = 0; i < dem_projs.size(); i++) {
  //  double height = center_elev[i];
  //  cv::Point pt(dem_projs[i][0], dem_projs[i][1]);
  //  if(pt.x >= 0 && pt.x < img_dem.cols && pt.y >= 0 && pt.y < img_dem.rows) {
  //    if(height > SUPER_OBSTACLE_THR)
  //      color = cv::Scalar(0,0,255);
  //    else if(height > OBSTACLE_THR && height <= SUPER_OBSTACLE_THR)
  //      color = cv::Scalar(255,0,0);
  //    else
  //      color = cv::Scalar(0,255,0);
  //    cv::circle(img_dem, pt, 1, color, -1);
  //  }
  //}
  // draw mesh lines
  int cells_x = dem.getSizeX();
  int cells_z = dem.getSizeZ();
  //for(int k = 0; k < 3; k++) { // just for preety colors
  for(int k = 0; k < 2; k++) { // just for preety colors
    for(size_t i = 0; i < cells_z; i++) {
      for(size_t j = 0; j < cells_x; j++) {
        if(!dem.isCellVisible(j,i)) continue;
        //std::cout << k << " - " << i << " - " << j << "\n";
        int idx = i*cells_x + j;
        //double height = center_elev[idx];
        int class_id = class_ids[idx];
        cv::Point pt(dem_projs[idx][0], dem_projs[idx][1]);
        if(pt.x >= 0 && pt.x < img_dem.cols && pt.y >= 0 && pt.y < img_dem.rows) {
          if(class_id == GROUND) {
            if(k != 0) continue;
            color = cv::Scalar(0,255,0);
          }
          else if(class_id == CURB) {
            if(k != 1) continue;
            color = cv::Scalar(255,0,0);
          }
          else if(class_id == OBSTACLE) {
            if(k != 2) continue;
            color = cv::Scalar(0,0,255);
          }
          else
            throw "Unknown class ID!\n";
          cv::Point pt2;
          int idx2;
          if(j > 0 && dem.isCellVisible(j-1, i)) {
            idx2 = i*cells_x + j - 1;
            if(class_ids[idx2] == OBSTACLE) continue;
            //if(k == 0 class_ids[idx2] != OBSTACLE) continue;
            pt2.x = dem_projs[idx2][0];
            pt2.y = dem_projs[idx2][1];
            if(pt2.x >= 0 && pt2.x < img_dem.cols && pt2.y >= 0 && pt2.y < img_dem.rows)
              cv::line(img_dem, pt, pt2, color, 1);
          }
          if(j < (cells_x-1) && dem.isCellVisible(j+1, i)) {
            idx2 = i*cells_x + j + 1;
            if(class_ids[idx2] == OBSTACLE) continue;
            pt2.x = dem_projs[idx2][0];
            pt2.y = dem_projs[idx2][1];
            if(pt2.x >= 0 && pt2.x < img_dem.cols && pt2.y >= 0 && pt2.y < img_dem.rows)
              cv::line(img_dem, pt, pt2, color, 1);
          }
          if(i > 0 && dem.isCellVisible(j, i-1)) {
            idx2 = (i-1)*cells_x + j;
            if(class_ids[idx2] == OBSTACLE) continue;
            pt2.x = dem_projs[idx2][0];
            pt2.y = dem_projs[idx2][1];
            if(pt2.x >= 0 && pt2.x < img_dem.cols && pt2.y >= 0 && pt2.y < img_dem.rows)
              cv::line(img_dem, pt, pt2, color, 1);
          }
          if(i < (cells_z-1) && dem.isCellVisible(j, i+1)) {
            idx2 = (i+1)*cells_x + j;
            if(class_ids[idx2] == OBSTACLE) continue;
            pt2.x = dem_projs[(i+1)*cells_x+j][0];
            pt2.y = dem_projs[(i+1)*cells_x+j][1];
            if(pt2.x >= 0 && pt2.x < img_dem.cols && pt2.y >= 0 && pt2.y < img_dem.rows)
              cv::line(img_dem, pt, pt2, color, 1);
          }
          //dem.printCellInfo(idx);
          //if(k == 1 && i > 60) {
          //  cv::imshow("image_DEM", img_dem);
          //  cv::waitKey(0);
          //}
        }
      }
    }
  }
}


void drawBasicDEM(const recon::DEM& dem, const std::vector<Eigen::Vector2d>& dem_projs,
                  const std::vector<double>& center_elev, cv::Mat& img_dem)
{
  cv::Scalar color;
  // draw dots
  //for(size_t i = 0; i < dem_projs.size(); i++) {
  //  double height = center_elev[i];
  //  cv::Point pt(dem_projs[i][0], dem_projs[i][1]);
  //  if(pt.x >= 0 && pt.x < img_dem.cols && pt.y >= 0 && pt.y < img_dem.rows) {
  //    if(height > SUPER_OBSTACLE_THR)
  //      color = cv::Scalar(0,0,255);
  //    else if(height > OBSTACLE_THR && height <= SUPER_OBSTACLE_THR)
  //      color = cv::Scalar(255,0,0);
  //    else
  //      color = cv::Scalar(0,255,0);
  //    cv::circle(img_dem, pt, 1, color, -1);
  //  }
  //}
  // draw mesh lines
  int cells_x = dem.getSizeX();
  int cells_z = dem.getSizeZ();
  //for(int k = 0; k < 3; k++) { // just for preety colors
  for(int k = 0; k < 2; k++) { // just for preety colors
    for(size_t i = 0; i < cells_z; i++) {
      for(size_t j = 0; j < cells_x; j++) {
        if(!dem.isCellVisible(j,i)) continue;
        //std::cout << k << " - " << i << " - " << j << "\n";
        int idx = i*cells_x + j;
        double height = center_elev[idx];
        cv::Point pt(dem_projs[idx][0], dem_projs[idx][1]);
        if(pt.x >= 0 && pt.x < img_dem.cols && pt.y >= 0 && pt.y < img_dem.rows) {
          if(height > SUPER_OBSTACLE_THR) {
            if(k != 2) continue;
            color = cv::Scalar(0,0,255);
          }
          else if(height > OBSTACLE_THR && height <= SUPER_OBSTACLE_THR) {
            if(k != 1) continue;
            color = cv::Scalar(255,0,0);
          }
          else {
            if(k != 0) continue;
            color = cv::Scalar(0,255,0);
          }
          if(j > 0 && j < (cells_x-1)) {
            cv::Point pt2;
            pt2.x = dem_projs[i*cells_x+j+1][0];
            pt2.y = dem_projs[i*cells_x+j+1][1];
            if(pt2.x >= 0 && pt2.x < img_dem.cols && pt2.y >= 0 && pt2.y < img_dem.rows)
              cv::line(img_dem, pt, pt2, color, 1);

            pt2.x = dem_projs[i*cells_x+j-1][0];
            pt2.y = dem_projs[i*cells_x+j-1][1];
            if(pt2.x >= 0 && pt2.x < img_dem.cols && pt2.y >= 0 && pt2.y < img_dem.rows)
              cv::line(img_dem, pt, pt2, color, 1);
          }
          if(i > 0 && i < (cells_z-1)) {
            cv::Point pt2;
            pt2.x = dem_projs[(i+1)*cells_x+j][0];
            pt2.y = dem_projs[(i+1)*cells_x+j][1];
            if(pt2.x >= 0 && pt2.x < img_dem.cols && pt2.y >= 0 && pt2.y < img_dem.rows)
              cv::line(img_dem, pt, pt2, color, 1);

            pt2.x = dem_projs[(i-1)*cells_x+j][0];
            pt2.y = dem_projs[(i-1)*cells_x+j][1];
            if(pt2.x >= 0 && pt2.x < img_dem.cols && pt2.y >= 0 && pt2.y < img_dem.rows)
              cv::line(img_dem, pt, pt2, color, 1);
          }
          //cv::imshow("image_DEM", img_dem);
          //cv::waitKey(0);
        }
      }
    }
  }
}

}
} // end namespace: recon
