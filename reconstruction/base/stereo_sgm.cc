#include "stereo_sgm.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <omp.h>

#include "stereo_costs.h"

//#include <Eigen/Core>

namespace recon
{

void StereoSGM::compute(cv::Mat& left_img, cv::Mat& right_img, cv::Mat& disp)
{
  // TODO
  //int p_width = params_.patch_width;
  //int p_height = params_.patch_height;
  //int margin_h = (p_width - 1) / 2;
  //int margin_v = (p_height - 1) / 2;

  int wsz = params_.window_sz;
  int mc = (wsz-1)/2;                     // margin crop size
  int img_height = left_img.rows;
  int img_width = left_img.cols;
  int height = img_height - 2*mc;
  int width = img_width - 2*mc;
  int disp_range = params_.disp_range;

  CostArray costs;
  //ACostArray aggr_costs, path_aggr_costs;
  ACostArray aggr_costs, path_aggr_costs;
  costs.resize(height);
  aggr_costs.resize(height);
  path_aggr_costs.resize(height);
  for(int i = 0; i < height; i++) {
    costs[i].resize(width);
    aggr_costs[i].resize(width);
    path_aggr_costs[i].resize(width);
    for(int j = 0; j < width; j++) {
      // CostType needs to be smaller then ACostType for int types
      costs[i][j].assign(disp_range, std::numeric_limits<CostType>::max());
      //costs[i][j].assign(disp_range, 255u);
      aggr_costs[i][j].assign(disp_range, (ACostType)0);
      path_aggr_costs[i][j].assign(disp_range, (ACostType)0);
      //costs[i][j].resize(disp_range);
      //for(int k = 0; k < disp_range; k++)
      //  costs[i][j][k] = std::numeric_limits<CostType>::max();
    }
  }

#ifdef COST_ZSAD
  cv::Mat left_means, right_means;
  StereoCosts::calcPatchMeans(left_img, left_means, wsz);
  StereoCosts::calcPatchMeans(right_img, right_means, wsz);
#endif
#ifdef COST_CENSUS
  cv::Mat left_census, right_census;
  StereoCosts::census_transform(left_img, wsz, left_census);
  StereoCosts::census_transform(right_img, wsz, right_census);
#endif

#ifdef COST_CENSUS
  //cv::Mat lcensus, rcensus;
  //left_census.convertTo(lcensus, CV_8U);
  //right_census.convertTo(rcensus, CV_8U);
  //cv::imshow("left_census", lcensus);
  //cv::imshow("right_census", rcensus);
  //cv::waitKey(0);
#endif

  //omp_set_dynamic(0);     // Explicitly disable dynamic teams
  //omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
  #pragma omp parallel for
  for(int y = 0; y < height; y++) {
    for(int d = 0; d < disp_range; d++) {
      for(int x = d; x < width; x++) {
#ifdef COST_SAD
        // SAD 1x1
        //costs[y][x][d] = std::abs(int(left_img.at<uint8_t>(iy, ix) - int(right_img.at<uint8_t>(iy, ix-d))));
        int ix = x + mc;
        int iy = y + mc;
        // SAD
        costs[y][x][d] = StereoCosts::get_cost_SAD(left_img, right_img, wsz, ix, iy, d);
#endif
#ifdef COST_ZSAD
        int ix = x + mc;
        int iy = y + mc;
        // ZSAD - 3x3, 2, 130
        costs[y][x][d] = StereoCosts::get_cost_ZSAD(left_img, right_img, left_means, right_means, wsz, ix, iy, d);
#endif
#ifdef COST_CENSUS
        // Census cost
        // Daimler: Traffic - 10, 50; Middlebury - 7, 20
        costs[y][x][d] = StereoCosts::hamming_dist<uint32_t>(left_census.at<uint32_t>(y, x),
                                                             right_census.at<uint32_t>(y, x-d));
#endif
      }
    }
  }

  // save scanline cost
  //cv::Mat cost_image = cv::Mat::zeros(disp_range, width, CV_8U);
  //for(int x = 0; x < width; x++) {
  //  for(int d = 0; d < disp_range; d++) {
  //    cost_image.at<uint8_t>(d,x) = costs[height/2][x][d];
  //    //std::cout << (int)costs[height/2][x][d] << "\n";
  //  }
  //}
  //cv::equalizeHist(cost_image, cost_image);
  //cv::imwrite("scanline_costs.png", cost_image);

  // TODO change path_aggr_costs to array
  //int path_id = 0;
  //#pragma omp parallel for schedule(dynamic,1) private(path_aggr_costs) collapse(2)

  //omp_set_num_threads(4); // Use 4 threads for all consecutive parallel regions
  //#pragma omp parallel for
  for(int x = -1; x <= 1; x++) {
    //#pragma omp parallel for
    for(int y = -1; y <= 1; y++) {
      if(x == 0 && y == 0)
        continue;
      //printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
      //printf("Cost propagation [%d , %d]\n", x, y);
      aggregate_costs(left_img, costs, x, y, path_aggr_costs);
      //int tid = omp_get_thread_num();
      //aggregate_costs(left_img, costs, x, y, path_aggr_costs[tid]);

      //#pragma omp atomic
      //sum_costs(path_aggr_costs[tid], aggr_costs);
      sum_costs(path_aggr_costs, aggr_costs);
    }
  }

  //std::cout << "Cost propagation [1,0]\n";
  //aggregate_costs(left_img, costs, 1, 0, path_aggr_costs);
  ////cv::imwrite("path_1_0.png", get_disparity_image(path_aggr_costs));
  //sum_costs(path_aggr_costs, aggr_costs);
  //std::cout << "Cost propagation [-1,0]\n";
  //init_costs((ACostType)0, path_aggr_costs);
  //aggregate_costs(left_img, costs, -1, 0, path_aggr_costs);
  ////cv::imwrite("path_-1_0.png", get_disparity_image(path_aggr_costs));
  //sum_costs(path_aggr_costs, aggr_costs);
  
  //std::cout << "Cost propagation [0,1]\n";
  ////init_costs((ACostType)0, path_aggr_costs);
  //aggregate_costs<0,1>(left_img, costs, path_aggr_costs);
  ////cv::imwrite("path_0_1.png", get_disparity_image(path_aggr_costs));
  //sum_costs(path_aggr_costs, aggr_costs);
  //
  //std::cout << "Cost propagation [0,-1]\n";
  ////init_costs((ACostType)0, path_aggr_costs);
  //aggregate_costs<0,-1>(left_img, costs, path_aggr_costs);
  ////cv::imwrite("path_0_-1.png", get_disparity_image(path_aggr_costs));
  //sum_costs(path_aggr_costs, aggr_costs);
  //
  //std::cout << "Cost propagation [1,1]\n";
  ////init_costs((ACostType)0, path_aggr_costs);
  //aggregate_costs<1,1>(left_img, costs, path_aggr_costs);
  ////cv::imwrite("path_1_1.png", get_disparity_image(path_aggr_costs));
  //sum_costs(path_aggr_costs, aggr_costs);
  //
  //std::cout << "Cost propagation [-1,-1]\n";
  ////init_costs((ACostType)0, path_aggr_costs);
  //aggregate_costs<-1,-1>(left_img, costs, path_aggr_costs);
  ////cv::imwrite("path_-1_-1.png", get_disparity_image(path_aggr_costs));
  //sum_costs(path_aggr_costs, aggr_costs);
  //
  //std::cout << "Cost propagation [1,-1]\n";
  ////init_costs((ACostType)0, path_aggr_costs);
  //aggregate_costs<1,-1>(left_img, costs, path_aggr_costs);
  ////cv::imwrite("path_1_-1.png", get_disparity_image(path_aggr_costs));
  //sum_costs(path_aggr_costs, aggr_costs);
  //
  //std::cout << "Cost propagation [-1,1]\n";
  ////init_costs((ACostType)0, path_aggr_costs);
  //aggregate_costs<-1,1>(left_img, costs, path_aggr_costs);
  ////cv::imwrite("path_-1_1.png", get_disparity_image(path_aggr_costs));
  //sum_costs(path_aggr_costs, aggr_costs);

  //disp = get_disparity_matrix(aggr_costs, mc);
  cv::Mat data_cost_image = GetDisparityImage(costs, mc);
  cv::imwrite("data_cost_img.png", data_cost_image);
  disp = get_disparity_image_uint16(aggr_costs, mc);
  cv::medianBlur(disp, disp, 3);
}

//template<int DIRX, int DIRY>
void StereoSGM::aggregate_costs(const cv::Mat& img, const CostArray& costs, int DIRX, int DIRY,
                                ACostArray& aggr_costs) {
  const int width = costs[0].size();
  const int height = costs.size();

  // Walk along the edges in a clockwise fashion
  if(DIRX > 0) {
    // Process every pixel along left most edge
    for(int y = 0; y < height; y++) {
      //aggr_costs[0][j] += costs[0][j];
      sum_vectors(costs[y][0], aggr_costs[y][0]);
    }
    for(int x = 1; x < width; x++) {
      //std::cout << "x = " << x << "\n";
      int y_start = std::max(0, 0 + DIRY * x);
      int y_stop  = std::min(height, height + DIRY * x);
      for(int y = y_start; y < y_stop; y++) {
        int gradient = static_cast<int>(std::abs(img.at<uint8_t>(y,x) - img.at<uint8_t>(y-DIRY,x-DIRX)));
        aggregate_path(aggr_costs[y-DIRY][x-DIRX], costs[y][x], aggr_costs[y][x], gradient);
      }
    }
  }
  if(DIRY > 0) {
    // Process every pixel along top most edge only if DIRX <= 0
    // Otherwise skip the top-left most pixel because we already processed
    for(int x = (DIRX <= 0 ? 0 : 1); x < width; x++) {
      //aggr_costs[0][j] += costs[0][j];
      sum_vectors(costs[0][x], aggr_costs[0][x]);
    }
    for(int y = 1; y < height; y++) {
      //std::cout << "y = " << y << "\n";
      int x_start = std::max( (DIRX <= 0 ? 0 : 1),
                              (DIRX <= 0 ? 0 : 1) + DIRX * y );
      int x_stop  = std::min( width, width + DIRX * y );
      for(int x = x_start; x < x_stop; x++) {
        int gradient = static_cast<int>(std::abs(img.at<uint8_t>(y,x) - img.at<uint8_t>(y-DIRY,x-DIRX)));
        aggregate_path(aggr_costs[y-DIRY][x-DIRX], costs[y][x], aggr_costs[y][x], gradient);
      }
    }
  }
  if(DIRX < 0) {
    // Process every pixel along right most edge only if DIRY <= 0
    // Otherwise skip the top-right most pixel because we already processed
    for(int y = (DIRY <= 0 ? 0 : 1); y < height; y++) {
      //aggr_costs[0][j] += costs[0][j];
      sum_vectors(costs[y][width-1], aggr_costs[y][width-1]);
    }
    for(int x = width-2; x >= 0; x--) {
      //std::cout << "x = " << x << "\n";
      int y_start = std::max( (DIRY <= 0 ? 0 : 1),
                              (DIRY <= 0 ? 0 : 1) - DIRY * (x - width + 1) );
      int y_stop  = std::min( height, height - DIRY * (x - width + 1) );
      for(int y = y_start; y < y_stop; y++) {
        int gradient = static_cast<int>(std::abs(img.at<uint8_t>(y,x) - img.at<uint8_t>(y-DIRY,x-DIRX)));
        aggregate_path(aggr_costs[y-DIRY][x-DIRX], costs[y][x], aggr_costs[y][x], gradient);
      }
    }
  }
  if(DIRY < 0) {
    // Process every pixel along bottom most edge only if DIRX <= 0
    // Otherwise skip the bottom-left and bottom-right most pixels because we already processed them
    for(int x = (DIRX <= 0 ? 0 : 1); x < (DIRX >= 0 ? width : width-1); x++) {
      //aggr_costs[0][j] += costs[0][j];
      sum_vectors(costs[height-1][x], aggr_costs[height-1][x]);
    }
    for(int y = height-2; y >= 0; y--) {
      //std::cout << "y = " << y << "\n";
      int x_start = std::max( (DIRX <= 0 ? 0 : 1),
                              (DIRX <= 0 ? 0 : 1) - DIRX * (y - height + 1) );
      int x_stop  = std::min( (DIRX >= 0 ? width : width - 1),
                              (DIRX >= 0 ? width : width - 1) - DIRX * (y - height + 1) );
      for(int x = x_start; x < x_stop; x++) {
        int gradient = static_cast<int>(std::abs(img.at<uint8_t>(y,x) - img.at<uint8_t>(y-DIRY,x-DIRX)));
        aggregate_path(aggr_costs[y-DIRY][x-DIRX], costs[y][x], aggr_costs[y][x], gradient);
      }
    }
  }
}

}
