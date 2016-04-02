#ifndef __SEMI_GLOBAL_MATCHING_H__
#define __SEMI_GLBOAL_MATCHING_H__

#include <vw/Image/ImageView.h>
#include <vw/Math/Vector.h>
#include <vw/FileIO.h>

namespace vw {

#define DISP_RANGE 128
#define PENALTY1 15
#define PENALTY2 100
//#define DISP_RANGE 128
//#define PENALTY1 40
//#define PENALTY2 1000

  // Accumulation Vector type
  typedef Vector<int32,DISP_RANGE> AVector;
  // Cost Vector type
  typedef Vector<int16,DISP_RANGE> CVector;

  AVector
  evaluate_path( AVector const& prior,
                 CVector const& local,
                 int path_intensity_gradient );

  template <int DIRX, int DIRY>
  void iterate_direction( ImageView<uint8> const& left_image,
                          ImageView<CVector > const& costs,
                          ImageView<AVector >& accumulated_costs ) {
    const int32 WIDTH = costs.cols();
    const int32 HEIGHT = costs.rows();

    // Walk along the edges in a clockwise fashion
    if ( DIRX > 0 ) {
      // LEFT MOST EDGE
      // Process every pixel along this edge
      for ( int32 j = 0; j < HEIGHT; j++ ) {
        accumulated_costs(0,j) += costs(0,j);
      }
      for ( int32 i = 1; i < WIDTH; i++ ) {
        int32 jstart = std::max( 0, 0 + DIRY * i );
        int32 jstop  = std::min( HEIGHT, HEIGHT + DIRY * i );
        for ( int32 j = jstart; j < jstop; j++ ) {
          accumulated_costs(i,j) =
            evaluate_path( accumulated_costs(i-DIRX,j-DIRY),
                           costs(i,j),
                           abs(left_image(i,j)-left_image(i-DIRX,j-DIRY)) );
        }
      }
    } if ( DIRY > 0 ) {
      // TOP MOST EDGE
      // Process every pixel along this edge only if DIRX ==
      // 0. Otherwise skip the top left most pixel
      for ( int32 i = (DIRX <= 0 ? 0 : 1 ); i < WIDTH; i++ ) {
        accumulated_costs(i,0) += costs(i,0);
      }
      for ( int32 j = 1; j < HEIGHT; j++ ) {
        int32 istart = std::max( (DIRX <= 0 ? 0 : 1),
                                 (DIRX <= 0 ? 0 : 1) + DIRX * j );
        int32 istop  = std::min( WIDTH, WIDTH + DIRX * j );
        for ( int32 i = istart; i < istop; i++ ) {
          accumulated_costs(i,j) =
            evaluate_path( accumulated_costs(i-DIRX,j-DIRY),
                           costs(i,j),
                           abs(left_image(i,j)-left_image(i-DIRX,j-DIRY)) );
        }
      }
    } if ( DIRX < 0 ) {
      // RIGHT MOST EDGE
      // Process every pixel along this edge only if DIRY ==
      // 0. Otherwise skip the top right most pixel
      for ( int32 j = (DIRY <= 0 ? 0 : 1); j < HEIGHT; j++ ) {
        accumulated_costs(WIDTH-1,j) += costs(WIDTH-1,j);
      }
      for ( int32 i = WIDTH-2; i >= 0; i-- ) {
        int32 jstart = std::max( (DIRY <= 0 ? 0 : 1),
                                 (DIRY <= 0 ? 0 : 1) - DIRY * (i - WIDTH + 1) );
        int32 jstop  = std::min( HEIGHT, HEIGHT - DIRY * (i - WIDTH + 1) );
        for ( int32 j = jstart; j < jstop; j++ ) {
          accumulated_costs(i,j) =
            evaluate_path( accumulated_costs(i-DIRX,j-DIRY),
                           costs(i,j),
                           abs(left_image(i,j)-left_image(i-DIRX,j-DIRY)) );
        }
      }
    } if ( DIRY < 0 ) {
      // BOTTOM MOST EDGE
      // Process every pixel along this edge only if DIRX ==
      // 0. Otherwise skip the bottom left and bottom right pixel
      for ( int32 i = (DIRX <= 0 ? 0 : 1);
            i < (DIRX >= 0 ? WIDTH : WIDTH-1); i++ ) {
        accumulated_costs(i,HEIGHT-1) += costs(i,HEIGHT-1);
      }
      for ( int32 j = HEIGHT-2; j >= 0; j-- ) {
        int32 istart = std::max( (DIRX <= 0 ? 0 : 1),
                                 (DIRX <= 0 ? 0 : 1) - DIRX * (j - HEIGHT + 1) );
        int32 istop  = std::min( (DIRX >= 0 ? WIDTH : WIDTH - 1),
                                 (DIRX >= 0 ? WIDTH : WIDTH - 1) - DIRX * (j - HEIGHT + 1) );
        for ( int32 i = istart; i < istop; i++ ) {
          accumulated_costs(i,j) =
            evaluate_path( accumulated_costs(i-DIRX,j-DIRY),
                           costs(i,j),
                           abs(left_image(i,j)-left_image(i-DIRX,j-DIRY)) );
        }
      }
    }
  }

  // ADD two image views of vector type. Vectors are not from pixel math base
  //
  // This only works if these are ImageViews!
  template <class PixelT>
  void inplace_sum_views( ImageView<PixelT> & im1,
                          ImageView<PixelT> const& im2 ) {
    PixelT * im1_ptr      = im1.data();
    const PixelT* im2_ptr = im2.data();
    while ( im1_ptr != im1.data() + im1.cols() * im1.rows() ) {
      *im1_ptr += *im2_ptr;
      im1_ptr++;
      im2_ptr++;
    }
  }

  inline uint8 find_min_index( AVector const& v ) {
    return std::distance(v.begin(),
                         std::min_element(v.begin(),
                                          v.end()));
  }

  // Goes across all the viterbi diagrams and extracts out the minimum
  // vector.
  ImageView<PixelGray<uint8> >
  create_disparity_view( ImageView<AVector> const& accumulated_costs );

  // Invokes a 8 path version of SGM
  ImageView<uint8>
  semi_global_matching_func( ImageView<uint8> const& left_image,
                             ImageView<uint8> const& right_image );

  template <class Image1T, class Image2T>
  class SemiGlobalMatchingView : public ImageViewBase<SemiGlobalMatchingView<Image1T,Image2T> > {
    Image1T m_left_image;
    Image2T m_right_image;
  public:
    typedef uint8 pixel_type;
    typedef uint8 result_type;
    typedef ProceduralPixelAccessor<SemiGlobalMatchingView> pixel_accessor;

    SemiGlobalMatchingView( ImageViewBase<Image1T> const& left,
                            ImageViewBase<Image2T> const& right ) :
      m_left_image(left.impl()), m_right_image(right.impl()) {}

    inline int32 cols() const { return m_left_image.cols(); }
    inline int32 rows() const { return m_left_image.rows(); }
    inline int32 planes() const { return 1; }

    inline pixel_accessor origin() const { return pixel_accessor( *this, 0, 0 ); }
    inline pixel_type operator()( int32 /*i*/, int32 /*j*/, int32 /*p*/ = 0) const {
      vw_throw( NoImplErr() << "CorrelationView::operator()(....) has not been implemented." );
      return pixel_type();
    }

    // Block rasterization section that does actual work
    typedef CropView<ImageView<pixel_type> > prerasterize_type;
    inline prerasterize_type prerasterize(BBox2i const& bbox) const {
      ImageView<PixelGray<uint8> > left = crop( edge_extend(m_left_image), bbox );
      BBox2i rbbox = bbox;
      rbbox.max() += Vector2i(64,0);
      ImageView<PixelGray<uint8> > right = crop( edge_extend(m_right_image), rbbox );
      return prerasterize_type( semi_global_matching_func( left, right ),
                                -bbox.min().x(), -bbox.min().y(), cols(), rows() );
    }

    template <class DestT>
    inline void rasterize(DestT const& dest, BBox2i const& bbox) const {
      vw::rasterize(prerasterize(bbox), dest, bbox);
    }
  };

  template <class Image1T, class Image2T>
  SemiGlobalMatchingView<Image1T,Image2T>
  semi_global_matching( ImageViewBase<Image1T> const& left,
                        ImageViewBase<Image2T> const& right ) {
    typedef SemiGlobalMatchingView<Image1T,Image2T> result_type;
    return result_type( left.impl(), right.impl() );
  }
}

#endif //__SEMI_GLBOAL_MATCHING__
