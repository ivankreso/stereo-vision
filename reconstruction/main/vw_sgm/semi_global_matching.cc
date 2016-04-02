#include "semi_global_matching.h"

#include <vw/Core/Debugging.h>

using namespace vw;

AVector
vw::evaluate_path( AVector const& prior,
                   CVector const& local,
                   int path_intensity_gradient ) {
  AVector curr_cost = local;

  int32 min_prior = prior[0];
  for( int32 d = 1; d < DISP_RANGE; d++) {
    if(prior[d] < min_prior)
      min_prior = prior[d];
  }
  for ( int32 d = 0; d < DISP_RANGE; d++ ) {
    int32 e_smooth = min_prior + std::max(PENALTY1, path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2);
    e_smooth = std::min(e_smooth, prior[d]);
    if(d > 0)
      e_smooth = std::min(e_smooth, prior[d-1] + PENALTY1);
    if(d < (DISP_RANGE-1))
      e_smooth = std::min(e_smooth, prior[d+1] + PENALTY1);

    curr_cost[d] += e_smooth;
  }

  // Normalize by subtracting min of prior cost
  return elem_diff(curr_cost,min(prior));
}
//AVector
//vw::evaluate_path( AVector const& prior,
//                   CVector const& local,
//                   int path_intensity_gradient ) {
//  AVector curr_cost = local;
//  for ( int32 d = 0; d < DISP_RANGE; d++ ) {
//    int32 e_smooth = std::numeric_limits<int32>::max();
//    for ( int32 d_p = 0; d_p < DISP_RANGE; d_p++ ) {
//      if ( d_p - d == 0 ) {
//        // No penality
//        e_smooth = std::min(e_smooth,prior[d_p]);
//      } else if ( abs(d_p - d) == 1 ) {
//        // Small penality
//        e_smooth = std::min(e_smooth,prior[d_p]+PENALTY1);
//      } else {
//        // Large penality
//        //e_smooth = std::min(e_smooth,prior[d_p] + std::max(PENALTY1,
//        //                    path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
//        e_smooth = std::min(e_smooth,prior[d_p] + PENALTY2);
//      }
//    }
//    curr_cost[d] += e_smooth;
//  }
//
//  // Normalize by subtracting min of prior cost
//  //return elem_diff(curr_cost,min(prior));
//  return curr_cost;
//}

ImageView<PixelGray<uint8> >
vw::create_disparity_view( ImageView<AVector> const& accumulated_costs ) {
  ImageView<PixelGray<uint8> > disparity( accumulated_costs.cols(),
                                          accumulated_costs.rows() );
  Timer timer("\tCalculate Disparity Minimum");
  for ( size_t j = 0; j < disparity.rows(); j++ ) {
    for ( size_t i = 0; i < disparity.cols(); i++ ) {
      disparity(i,j) =
        4 * find_min_index( accumulated_costs(i,j) );
    }
  }
  return disparity;
}

ImageView<uint8>
vw::semi_global_matching_func( ImageView<uint8> const& left_image,
                               ImageView<uint8> const& right_image ) {

  Vector2i size( left_image.cols(), left_image.rows() );

  // Processing all costs. W*H*D. D= DISP_RANGE
  ImageView<CVector >
    costs( size.x(), size.y() );

  size_t buffer_size = size.x() * size.y();
  CVector temporary;
  std::fill(&temporary[0], &temporary[0]+DISP_RANGE, 255u);
  std::fill(costs.data(),costs.data()+buffer_size,
            temporary);
  {
    Timer timer("\tCost Calculation");
    for ( size_t j = 0; j < size.y(); j++ ) {
      for ( size_t d = 0; d < DISP_RANGE; d++ ) {
        for ( size_t i = d; i < size.x(); i++ ) {
          costs(i,j)[d] = abs( int(left_image(i,j)) - int(right_image(i-d,j)) );
          //std::cout << "Planes = " << left_image.planes() << "\n";
        }
      }
    }
  }

  ImageView<uint8> cost_image( left_image.cols(), DISP_RANGE );
  for ( size_t i = 0; i < left_image.cols(); i++ ) {
    for ( size_t j = 0; j < DISP_RANGE; j++ ) {
      cost_image(i,j) = costs(i,185)[j];
    }
  }
  write_image("scanline_costs.png",cost_image);

  ImageView<AVector >
    accumulated_costs( left_image.cols(), left_image.rows() ),
    dir_accumulated_costs( left_image.cols(), left_image.rows() );

  {
    Timer timer_total("\tCost Propagation");
    {
      Timer timer("\tCost Propagation [1,0]");
      std::fill(dir_accumulated_costs.data(), dir_accumulated_costs.data()+buffer_size,
                AVector());
      iterate_direction<1,0>( left_image, costs, dir_accumulated_costs );
      write_image("effect_1_0.png", create_disparity_view( dir_accumulated_costs ) );
      inplace_sum_views( accumulated_costs, dir_accumulated_costs );
    }
    {
      Timer timer("\tCost Propagation [-1,0]");
      std::fill(dir_accumulated_costs.data(), dir_accumulated_costs.data()+buffer_size,
                AVector());
      iterate_direction<-1,0>( left_image, costs, dir_accumulated_costs );
      write_image("effect_-1_0.png", create_disparity_view( dir_accumulated_costs ) );
      inplace_sum_views( accumulated_costs, dir_accumulated_costs );
    }
    {
      Timer timer("\tCost Propagation [0,1]");
      std::fill(dir_accumulated_costs.data(), dir_accumulated_costs.data()+buffer_size,
                AVector());
      iterate_direction<0,1>( left_image, costs, dir_accumulated_costs );
      write_image("effect_0_1.png", create_disparity_view( dir_accumulated_costs ) );
      inplace_sum_views( accumulated_costs, dir_accumulated_costs );
    }
    {
      Timer timer("\tCost Propagation [0,-1]");
      std::fill(dir_accumulated_costs.data(), dir_accumulated_costs.data()+buffer_size,
                AVector());
      iterate_direction<0,-1>( left_image, costs, dir_accumulated_costs );
      write_image("effect_0_-1.png", create_disparity_view( dir_accumulated_costs ) );
      inplace_sum_views( accumulated_costs, dir_accumulated_costs );
    }
    {
      Timer timer("\tCost Propagation [1,1]");
      std::fill(dir_accumulated_costs.data(), dir_accumulated_costs.data()+buffer_size,
                AVector());
      iterate_direction<1,1>( left_image, costs, dir_accumulated_costs );
      write_image("effect_1_1.png", create_disparity_view( dir_accumulated_costs ) );
      inplace_sum_views( accumulated_costs, dir_accumulated_costs );
    }
    {
      Timer timer("\tCost Propagation [-1,-1]");
      std::fill(dir_accumulated_costs.data(), dir_accumulated_costs.data()+buffer_size,
                AVector());
      iterate_direction<-1,-1>( left_image, costs, dir_accumulated_costs );
      write_image("effect_-1_-1.png", create_disparity_view( dir_accumulated_costs ) );
      inplace_sum_views( accumulated_costs, dir_accumulated_costs );
    }
    {
      Timer timer("\tCost Propagation [1,-1]");
      std::fill(dir_accumulated_costs.data(), dir_accumulated_costs.data()+buffer_size,
                AVector());
      iterate_direction<1,-1>( left_image, costs, dir_accumulated_costs );
      write_image("effect_1_-1.png", create_disparity_view( dir_accumulated_costs ) );
      inplace_sum_views( accumulated_costs, dir_accumulated_costs );
    }
    {
      Timer timer("\tCost Propagation [-1,1]");
      std::fill(dir_accumulated_costs.data(), dir_accumulated_costs.data()+buffer_size,
                AVector());
      iterate_direction<-1,1>( left_image, costs, dir_accumulated_costs );
      write_image("effect_-1_1.png", create_disparity_view( dir_accumulated_costs ) );
      inplace_sum_views( accumulated_costs, dir_accumulated_costs );
    }
  }

  return create_disparity_view( accumulated_costs );
}
